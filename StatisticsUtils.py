# -*- coding: utf-8 -*-
"""
Created on 2020.03.03

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import xarray as xr
import numpy as np
from scipy.stats import linregress


def lag_linregress(x, y, lagx=0, lagy=0):
    """
    Calculate the lead-lag linear regression.

    Parameters
    ----------
    x : xarray.DataArray
        The x-coordinates at which to evaluate the interpolated values.
    y : xarray.DataArray
        The x-coordinates of the data points.
    lagx : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as `xp`.

    Returns
    ----------
    re : tuple of floats
        Covariance, correlation, regression slope and intercept, p-value,
        and standard error on regression between the two datasets along
        their aligned time dimension.  Lag values can be assigned to either
        of the data, with lagx shifting x, and lagy shifting y, with the
        specified lag amount. 

    Input: Two xr.Datarrays of any dimensions with the first dim being time. 
    Thus the input data could be a 1D time series, or for example, have three dimensions (time,lat,lon). 
    Datasets can be provied in any order, but note that the regression slope and intercept will be calculated
    for y with respect to x.
    """ 
    #1. Ensure that the data are properly alinged to each other. 
    x,y = xr.align(x,y)
    
    #2. Add lag information if any, and shift the data accordingly
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards. 
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). Hence it needs to be dropped
        x   = x.shift(time = -lagx).dropna(dim='time')
        #Next important step is to re-align the two datasets so that y adjusts to the changed coordinates of x
        x,y = xr.align(x,y)

    if lagy!=0:
        y   = y.shift(time = -lagy).dropna(dim='time')
        x,y = xr.align(x,y)
 
    # slope, intercept, r_value, p_value, std_err = linregress(x, y)
    slp, itc, r, p, std = xr.apply_ufunc(linregress,
                      x,
                      y,
                      dask='allowed',
                      input_core_dims=[['time'],['time']],
                      output_core_dims=[[],[],[],[],[]],
                      # exclude_dims=set(('contour',)),
                      # output_dtypes=[theta.dtype],
                      vectorize=True
                      )

    return slp, itc, r, p, std


def lag_linregress2(x, y, lagx=0, lagy=0):
    """
    Calculate the lead-lag linear regression.

    Parameters
    ----------
    x : xarray.DataArray
        The x-coordinates at which to evaluate the interpolated values.
    y : xarray.DataArray
        The x-coordinates of the data points.
    lagx : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as `xp`.
    inc : boolean
        xp is increasing or decresing.

    Returns
    ----------
    re : tuple of floats
        Covariance, correlation, regression slope and intercept, p-value,
        and standard error on regression between the two datasets along
        their aligned time dimension.  Lag values can be assigned to either
        of the data, with lagx shifting x, and lagy shifting y, with the
        specified lag amount. 

    Input: Two xr.Datarrays of any dimensions with the first dim being time. 
    Thus the input data could be a 1D time series, or for example, have three dimensions (time,lat,lon). 
    Datasets can be provied in any order, but note that the regression slope and intercept will be calculated
    for y with respect to x.
    """ 
    #1. Ensure that the data are properly alinged to each other. 
    x,y = xr.align(x,y)
    
    #2. Add lag information if any, and shift the data accordingly
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards. 
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). Hence it needs to be dropped
        x    = x.shift(time = -lagx).dropna(dim='time', how='all')
        #Next important step is to re-align the two datasets so that y adjusts to the changed coordinates of x
        x, y = xr.align(x, y)

    if lagy!=0:
        y    = y.shift(time = -lagy).dropna(dim='time', how='all')
        x, y = xr.align(x, y)

    #3. Compute data length, mean and standard deviation along time axis for further use: 
    n     = y.count(axis=0)
    n     = n.where(n!=0)
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd  = x.std(axis=0)
    ystd  = y.std(axis=0)
    
    #4. Compute covariance along time axis
    cov   =  ((x - xmean)*(y - ymean)).sum(axis=0, skipna=True)/(n)
    
    #5. Compute correlation along time axis
    cor   = cov/(xstd*ystd)
    
    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  
    
    #7. Compute P-value and standard error
    #Compute t-statistics
    tstats = cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    stderr = slope/tstats
    
    from scipy.stats import t
    pval   = t.sf(tstats, n-2)*2
    pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

    return cov,cor,slope,intercept,pval,stderr

