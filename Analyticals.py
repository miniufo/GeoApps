# -*- coding: utf-8 -*-
"""
Created on 2019.08.16

@author: MiniUFO, Emily
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr


"""
Analytical expressions defined below.
"""
def Rossby_Haurwitz_wave(lon, lat, a=6371200.0, omega=7.848e-6, K=7.848e-6,
                         phiConst=8e3, OMEGA=7.292e-5, R=4):
    """
    Calculate tendency due to advection.
    
    Parameters
    ----------
    lon : xarray.DataArray
        Longitudes in degree.
    lat : xarray.DataArray
        Latitudes in degree.
    a : float
        Radius of the sphere in meter (Earth is default).
    omega : float
        A constant.
    K : float
        A constant.
    phiConst : float
        Constant geopotential height in gpm.
    OMEGA : float
        Rotating speed of the sphere (Earth is default).
    R : int
        Zonal wave number.
    """
    lonR = np.deg2rad(lon)
    latR = np.deg2rad(lat)
    
    sin = np.sin
    cos = np.cos
    
    psi = -a**2 * (omega * sin(latR) -
                   K * cos(latR)**R * sin(latR) * cos(R*lonR))
    
    u = a * (omega * cos(latR) +
             K * cos(latR)**(R-1) * (R*sin(latR)**2-cos(latR)**2) * cos(R*lonR))
    
    v = -a*K*R * cos(latR)**(R-1) * sin(latR) * sin(R*lonR)
    
    zeta = 2 * omega * sin(latR) - \
           K*sin(latR) * cos(latR)**R * (R**2+3*R+2) * cos(R*lonR)
    
    A = omega*(2*OMEGA+omega)*cos(latR)**2/2. + \
        K**2*cos(latR)**(2*R) * ((R+1)*cos(latR)**2 + (2*R**2-R-2) -
                                 2*R**2*cos(latR)**-2) / 4.
    
    B = 2*(OMEGA+omega)*K/(R+1)/(R+2) * \
        cos(latR)**R*((R**2+2*R+2)-(R+1)**2*cos(latR)**2)
    
    C = K**2*cos(latR)**(2*R) * ((R+1)*cos(latR)**2+(R+2))/4
    
    phi = phiConst + a**2 * (A + B*cos(R*lonR) + C*cos(2*R*lonR))
    
    psi  =  psi.rename('psi' ).astype(np.float32)
    u    =    u.rename('u'   ).astype(np.float32)
    v    =    v.rename('v'   ).astype(np.float32)
    zeta = zeta.rename('zeta').astype(np.float32)
    phi  =  phi.rename('phi' ).astype(np.float32)
    
    return psi, u, v, zeta, phi


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in Analyticals')
    
    lonV = np.linspace(0, 360, 360)
    latV = np.linspace(-90, 90, 181)
    
    lon = xr.DataArray(lonV, dims='lon', coords={'lon': lonV})
    lat = xr.DataArray(latV, dims='lat', coords={'lat': latV})

