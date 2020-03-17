# -*- coding: utf-8 -*-
"""
Created on 2020.02.04

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import xarray as xr
import numpy as np
from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds
from GeoApps.ConstUtils import deg2m


dimXList = ['lon', 'longitude', 'LON', 'LONGITUDE', 'geolon', 'GEOLON']
dimYList = ['lat', 'latitude' , 'LAT', 'LATITUDE' , 'geolat', 'GEOLAT']


def add_latlon_metrics(dset, dims=None):
    """
    Infer 2D metrics (latitude/longitude) from gridded data file.

    Parameters
    ----------
    dset : xarray.Dataset
        A dataset open from a file
    dims : dict
        Dimension pair in a dict, e.g., {'lat':'latitude', 'lon':'longitude'}

    Return
    -------
    dset : xarray.Dataset
        Input dataset with appropriated metrics added
    grid : xgcm.Grid
        The grid with appropriated metrics
    """
    lon, lat = None, None
    
    if dims is None:
        for dim in dimXList:
            if dim in dset:
                lon = dim
                break

        for dim in dimYList:
            if dim in dset:
                lat = dim
                break

        if lon is None or lat is None:
            raise Exception('unknown dimension names in dset, should be in '
                            + str(dimXList + dimYList))
    else:
        lon, lat = dims['lon'], dims['lat']
    
    ds = generate_grid_ds(dset, {'X':lon, 'Y':lat})
    
    coords = ds.coords
    
    if __is_periodic(coords[lon], 360.0):
        periodic = 'X'
    else:
        periodic = []
    
    grid = Grid(ds, periodic=periodic)
    
    na = np.nan
    
    if 'X' in periodic:
        dlonG = grid.diff(ds[lon        ], 'X', boundary_discontinuity=360)
        dlonC = grid.diff(ds[lon+'_left'], 'X', boundary_discontinuity=360)
    else:
        dlonG = grid.diff(ds[lon        ], 'X', boundary='fill', fill_value=na)
        dlonC = grid.diff(ds[lon+'_left'], 'X', boundary='fill', fill_value=na)
        
    dlatG = grid.diff(ds[lat        ], 'Y', boundary='fill', fill_value=na)
    dlatC = grid.diff(ds[lat+'_left'], 'Y', boundary='fill', fill_value=na)
    
    coords['dxG'], coords['dyG'] = __dll_dist(dlonG, dlatG, ds[lon], ds[lat])
    coords['dxC'], coords['dyC'] = __dll_dist(dlonC, dlatC, ds[lon], ds[lat])
    coords['rAc'] = ds['dyC'] * ds['dxC']
    
    metrics={('X',    ): ['dxG', 'dxC'], # X distances
             ('Y' ,   ): ['dyG', 'dyC'], # Y distances
             ('X', 'Y'): ['rAc']}
    
    grid._assign_metrics(metrics)
    
    return ds, grid


def add_MITgcm_missing_metrics(dset, periodic=None):
    """
    Infer missing metrics from MITgcm output files.

    Parameters
    ----------
    dset : xarray.Dataset
        A dataset open from a file

    Return
    -------
    dset : xarray.Dataset
        Input dataset with appropriated metrics added
    grid : xgcm.Grid
        The grid with appropriated metrics
    """
    coords = dset.coords
    grid   = Grid(dset, periodic=periodic)
    
    BCx = 'periodic' if grid.axes['X']._periodic else 'fill'
    BCy = 'periodic' if grid.axes['Y']._periodic else 'fill'
    
    if 'drW' not in coords: # vertical cell size at u point
        coords['drW'] = dset.hFacW * dset.drF
    if 'drS' not in coords: # vertical cell size at v point
        coords['drS'] = dset.hFacS * dset.drF
    if 'drC' not in coords: # vertical cell size at tracer point
        coords['drC'] = dset.hFacC * dset.drF
    
    if 'dxF' not in coords:
        coords['dxF'] = grid.interp(dset.dxC, 'X', boundary=BCx)
    if 'dyF' not in coords:
        coords['dyF'] = grid.interp(dset.dyC, 'Y', boundary=BCy)
    if 'dxV' not in coords:
        coords['dxV'] = grid.interp(dset.dxG, 'X', boundary=BCx)
    if 'dyU' not in coords:
        coords['dyU'] = grid.interp(dset.dyG, 'Y', boundary=BCy)
    
    if 'hFacZ' not in coords:
        coords['hFacZ'] = grid.interp(dset.hFacS, 'X', boundary=BCx)
    if 'maskZ' not in coords:
        coords['maskZ'] = coords['hFacZ']
    
    # Calculate vertical distances located on the cellboundary
    # ds.coords['dzC'] = grid.diff(ds.depth, 'Z', boundary='extrapolate')
    # Calculate vertical distances located on the cellcenter
    # ds.coords['dzT'] = grid.diff(ds.depth_left, 'Z', boundary='extrapolate')
    
    metrics = {
        ('X',)    : ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
        ('Y',)    : ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
        ('Z',)    : ['drW', 'drS', 'drC', 'drF'], # Z distances
        ('X', 'Y'): ['rAw', 'rAs', 'rA' , 'rAz']} # Areas
    
    grid._assign_metrics(metrics)
    
    return dset, grid


"""
Helper (private) methods are defined below
"""
def __dll_dist(dlon, dlat, lon, lat):
    """
    Converts lat/lon differentials into distances in meters.

    Parameters
    ----------
    dlon : xarray.DataArray
        longitude differentials
    dlat : xarray.DataArray
        latitude differentials
    lon  : xarray.DataArray
        longitude values
    lat  : xarray.DataArray
        latitude values

    Return
    -------
    dx  : xarray.DataArray
        Distance inferred from dlon
    dy  : xarray.DataArray
        Distance inferred from dlat
    """
     # cos(+/-90) is not exactly zero, add a threshold
    dx = xr.ufuncs.cos(xr.ufuncs.deg2rad(lat))
    dx = xr.where(dx<1e-15, 0, dx) * dlon * deg2m
    dy = dlat * ((lon * 0) + 1) * deg2m
    
    return dx, dy

def __is_periodic(coord, period):
    """
    Whether a given coordinate array is periodic.

    Parameters
    ----------
    coord  : xarray.DataArray
        A given coordinate e.g., longitude
    period : float
        Period used to justify the coordinate, e.g., 360 for longitude
    """
    # assume it is linear increasing
    if coord.size == 1:
        return False

    delta = coord[1] - coord[0]
	
    start = coord[-1] + delta - period;
		
    if np.abs((start - coord[0]) / delta) > 1e-4:
        return False;
		
    return True


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in GridUtils.py')
    
    
    # ds['drW'] = ds.hFacW * ds.drF #vertical cell size at u point
    # ds['drS'] = ds.hFacS * ds.drF #vertical cell size at v point
    # ds['drC'] = ds.hFacC * ds.drF #vertical cell size at tracer point
    # ds['dxF'] = grid.interp(ds.dxC,'X')
    # ds['dyF'] = grid.interp(ds.dyC,'Y')
    # ds['dxV'] = grid.interp(ds.dxG,'X')
    # ds['dyU'] = grid.interp(ds.dyG,'Y')
    
    # maskC = ds.maskC
    # maskW = ds.maskW
    # maskS = ds.maskS
    # maskZ = grid.interp(ds.hFacS, 'X')
    
    
    # # Calculate vertical distances located on the cellboundary
    # ds.coords['dzC'] = grid.diff(ds.depth, 'Z', boundary='extrapolate')
    # # Calculate vertical distances located on the cellcenter
    # ds.coords['dzT'] = grid.diff(ds.depth_left, 'Z', boundary='extrapolate')
        
    # metrics = {
    #     ('X',): ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
    #     ('Y',): ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
    #     ('Z',): ['drW', 'drS', 'drC', 'drF'], # Z distances
    #     ('X', 'Y'): ['rAw', 'rAs', 'rAc', 'rAz'] # Areas
    # }
    
    # grid = Grid(ds, metrics=metrics)
    
