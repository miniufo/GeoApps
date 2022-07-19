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


dimXList = ['lon', 'longitude', 'LON', 'LONGITUDE', 'geolon', 'GEOLON',
            'xt_ocean']
dimYList = ['lat', 'latitude' , 'LAT', 'LATITUDE' , 'geolat', 'GEOLAT',
            'yt_ocean']
dimZList = ['lev', 'level', 'LEV', 'LEVEL', 'pressure', 'PRESSURE',
            'depth', 'DEPTH']


def add_latlon_metrics(dset, dims=None, boundary=None):
    """
    Infer 2D metrics (latitude/longitude) from gridded data file.

    Parameters
    ----------
    dset : xarray.Dataset
        A dataset open from a file
    dims : dict
        Dimension pair in a dict, e.g., {'lat':'latitude', 'lon':'longitude'}
    boundary : dict
        Default boundary conditions applied to each coordinate

    Return
    -------
    dset : xarray.Dataset
        Input dataset with appropriated metrics added
    grid : xgcm.Grid
        The grid with appropriated metrics
    """
    lon, lat, lev = None, None, None
    
    if dims is None:
        for dim in dimXList:
            if dim in dset or dim in dset.coords:
                lon = dim
                break

        for dim in dimYList:
            if dim in dset or dim in dset.coords:
                lat = dim
                break

        for dim in dimZList:
            if dim in dset or dim in dset.coords:
                lev = dim
                break

        if lon is None or lat is None:
            raise Exception('unknown dimension names in dset, should be in '
                            + str(dimXList + dimYList))
    else:
        lon = dims['lon'] if 'lon' in dims else None
        lat = dims['lat'] if 'lat' in dims else None
        lev = dims['lev'] if 'lev' in dims else None
        
    if lev is None:
        ds = generate_grid_ds(dset, {'X':lon, 'Y':lat})
    else:
        ds = generate_grid_ds(dset, {'X':lon, 'Y':lat, 'Z':lev})
    
    coords = ds.coords
    
    BCx, BCy, BCz = 'extend', 'extend', 'extend'
    
    if boundary is not None:
        BCx = boundary['X'] if 'X' in boundary else 'extend'
        BCy = boundary['Y'] if 'Y' in boundary else 'extend'
        BCz = boundary['Z'] if 'Z' in boundary else 'extend'
    
    if __is_periodic(coords[lon], 360.0):
        periodic = 'X'
        
        if lev is None:
            grid = Grid(ds, periodic=periodic, boundary={'Y': BCy})
        else:
            grid = Grid(ds, periodic=periodic, boundary={'Z':BCz, 'Y': BCy})
    else:
        periodic = []
        
        if lev is None:
            grid = Grid(ds, boundary={'Y': BCy, 'X': BCx})
        else:
            grid = Grid(ds, boundary={'Z': BCz, 'Y': BCy, 'X': BCx})
    
    
    lonC = ds[lon]
    latC = ds[lat]
    lonG = ds[lon + '_left']
    latG = ds[lat + '_left']
    
    if 'X' in periodic:
        # dlonC = grid.diff(lonC, 'X', boundary_discontinuity=360)
        # dlonG = grid.diff(lonG, 'X', boundary_discontinuity=360)
        dlonC = grid.diff(lonC, 'X')
        dlonG = grid.diff(lonG, 'X')
    else:
        dlonC = grid.diff(lonC, 'X', boundary='extrapolate')
        dlonG = grid.diff(lonG, 'X', boundary='extrapolate')
    
    dlatC = grid.diff(latC, 'Y')
    dlatG = grid.diff(latG, 'Y')
    
    coords['dxG'], coords['dyG'] = __dll_dist(dlonG, dlatG, lonG, latG)
    coords['dxC'], coords['dyC'] = __dll_dist(dlonC, dlatC, lonC, latC)
    coords['dxF'] = grid.interp(coords['dxG'], 'Y')
    coords['dyF'] = grid.interp(coords['dyG'], 'X')
    coords['dxV'] = grid.interp(coords['dxG'], 'X')
    coords['dyU'] = grid.interp(coords['dyG'], 'Y')
    
    coords['rA' ] = ds['dyF'] * ds['dxF']
    coords['rAw'] = ds['dyG'] * ds['dxC']
    coords['rAs'] = ds['dyC'] * ds['dxG']
    coords['rAz'] = ds['dyU'] * ds['dxV']
    
    if lev is not None:
        levC = ds[lev].values
        tmp  = np.diff(levC)
        tmp  = np.concatenate([[(levC[0]-tmp[0])], levC])
        levG = tmp[:-1]
        delz = np.diff(tmp)
        
        ds[lev + '_left'] = levG
        coords['drF'] = xr.DataArray(delz, dims=lev, coords={lev: levC})
        coords['drG'] = xr.DataArray(np.concatenate([[delz[0]/2], delz[1:-1],
                                      [delz[-1]/2]]), dims=lev+'_left',
                                      coords={lev+'_left': levG})
        
        metrics={('X',    ): ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
                 ('Y' ,   ): ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
                 ('Z' ,   ): ['drG', 'drF'],               # Z distances
                 ('X', 'Y'): ['rAw', 'rAs', 'rA' , 'rAz']}
    else:
        metrics={('X',    ): ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
                 ('Y' ,   ): ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
                 ('X', 'Y'): ['rAw', 'rAs', 'rA' , 'rAz']}
    
    # print('lonC', lonC.dims)
    # print('latC', latC.dims)
    # print('lonG', lonG.dims)
    # print('latG', latG.dims)
    # print('')
    # print('dlonC', dlonC.dims)
    # print('dlatC', dlatC.dims)
    # print('dlonG', dlonG.dims)
    # print('dlatG', dlatG.dims)
    # print('')
    # print('dxG', coords['dxG'].dims)
    # print('dyG', coords['dyG'].dims)
    # print('dxF', coords['dxF'].dims)
    # print('dyF', coords['dyF'].dims)
    # print('dxC', coords['dxC'].dims)
    # print('dyC', coords['dyC'].dims)
    # print('dxV', coords['dxV'].dims)
    # print('dyU', coords['dyU'].dims)
    # print('')
    # print('rA' , coords['rA' ].dims)
    # print('rAz', coords['rAz'].dims)
    # print('rAw', coords['rAw'].dims)
    # print('rAs', coords['rAs'].dims)
    
    for key, value in metrics.items():
        grid.set_metrics(key, value)
    
    return ds, grid


def add_MITgcm_missing_metrics(dset, periodic=None, boundary=None, partial_cell=True):
    """
    Infer missing metrics from MITgcm output files.

    Parameters
    ----------
    dset : xarray.Dataset
        A dataset open from a file
    periodic : str
        Which coordinate is periodic
    boundary : dict
        Default boundary conditions applied to each coordinate
    partial_cell: bool
        Turn on the partial-cell or not (default is on).

    Return
    -------
    dset : xarray.Dataset
        Input dataset with appropriated metrics added
    grid : xgcm.Grid
        The grid with appropriated metrics
    """
    coords = dset.coords
    grid   = Grid(dset, periodic=periodic, boundary=boundary)
    
    if 'drW' not in coords: # vertical cell size at u point
        coords['drW'] = dset.hFacW * dset.drF if partial_cell else dset.drF
    if 'drS' not in coords: # vertical cell size at v point
        coords['drS'] = dset.hFacS * dset.drF if partial_cell else dset.drF
    if 'drC' not in coords: # vertical cell size at tracer point
        coords['drC'] = dset.hFacC * dset.drF if partial_cell else dset.drF
    if 'drG' not in coords: # vertical cell size at tracer point
        coords['drG'] = dset.Zl - dset.Zl + dset.drC.values[:-1]
        # coords['drG'] = xr.DataArray(dset.drC[:-1].values, dims='Zl',
        #                              coords={'Zl':dset.Zl.values})
    
    if 'dxF' not in coords:
        coords['dxF'] = grid.interp(dset.dxC, 'X')
    if 'dyF' not in coords:
        coords['dyF'] = grid.interp(dset.dyC, 'Y')
    if 'dxV' not in coords:
        coords['dxV'] = grid.interp(dset.dxG, 'X')
    if 'dyU' not in coords:
        coords['dyU'] = grid.interp(dset.dyG, 'Y')
    
    if 'hFacZ' not in coords:
        coords['hFacZ'] = grid.interp(dset.hFacS, 'X')
    if 'maskZ' not in coords:
        coords['maskZ'] = coords['hFacZ']
        
    if 'yA' not in coords:
        coords['yA'] = dset.drF * dset.hFacC * dset.dxF if partial_cell \
                  else dset.drF * dset.dxF
    
    # Calculate vertical distances located on the cellboundary
    # ds.coords['dzC'] = grid.diff(ds.depth, 'Z', boundary='extrapolate')
    # Calculate vertical distances located on the cellcenter
    # ds.coords['dzT'] = grid.diff(ds.depth_left, 'Z', boundary='extrapolate')
    
    metrics = {
        ('X',)    : ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
        ('Y',)    : ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
        ('Z',)    : ['drW', 'drS', 'drC', 'drF', 'drG'], # Z distances
        ('X', 'Y'): ['rAw', 'rAs', 'rA' , 'rAz'], # Areas in X-Y plane
        ('X', 'Z'): ['yA']} # Areas in X-Z plane
    
    for key, value in metrics.items():
        grid.set_metrics(key, value)
    
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
    dx = np.cos(np.deg2rad(lat)) * dlon * deg2m
    dy = (dlat + lon - lon) * deg2m
    
    # cos(+/-90) is not exactly zero, add a threshold
    dx = xr.where(dx<1e-15, 0, dx)
    
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
    
