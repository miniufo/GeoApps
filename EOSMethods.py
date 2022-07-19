# -*- coding: utf-8 -*-
"""
Created on 2020.01.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import seawater
import xgcm
import xmitgcm
import xarray as xr
import matplotlib.pyplot as plt
from GeoApps.Application import Application
from .ConstUtils import g


class EOSMethods(Application):
    """
    This class is designed for EOS related methods in MITgcm.
    """
    def __init__(self, dset, grid=None, arakawa='A'):
        """
        Construct a EOS instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        arakawa : str
            The type of the grid. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        """
        super().__init__(dset, grid=grid, arakawa=arakawa)


    """
    All the EOS-related terms calculation methods.
    """
    @staticmethod
    def cal_insitu_density(THETA, SALT, PHrefC, rhoRef=999.8):
        """
        Calculate in-situ density using model outputs and seawater module.
    
        Parameters
        ----------
        THETA : DataArray
            Model-output potential temperature (degree).
        SALT  : DataArray
            Model-output salinity (PSU).
        PHrefC: DataArray
            Model-output reference pressure potential (p/rhoRef)
            (m^2 s^-2) at vertical grid cell.
        rhoRef: float or DataArray
            Model-output reference density (kg m^-3).
    
        Return
        ----------
        dens : xarray.DataArray
            In-situ density that should be exactly the same as
            RHOAnoma + rhoRef
        """
        # get MITgcm diagnostics
        PRESS = PHrefC * (rhoRef / 10000) # in unit of dbar
        
        # cal. in-situ temperature
        TEMP = xr.apply_ufunc(seawater.temp, SALT, THETA, PRESS, 0)
        
        # cal. in-situ density
        RHO  = xr.apply_ufunc(seawater.dens, SALT, TEMP , PRESS)
        
        RHO.rename('RHO')
        
        return RHO
    
    
    @staticmethod
    def cal_linear_insitu_density(THETA, SALT,
                           tRef  =20  , sRef=30, rhoRef=999.8,
                           tAlpha=2E-4, sBeta=7.4E-4):
        """
        Calculate in-situ density using model outputs and linear EOS.
    
        Parameters
        ----------
        THETA : DataArray
            Model-output potential temperature (degree).
        SALT  : DataArray
            Model-output salinity (PSU).
        tRef  : float
            Model-output reference potential temperature (degree).
        sRef  : float
            Model-output reference salinity (PSU).
        rhoRef: float
            Model-output reference density (kg m^-3).
        tAlpha: float
            Thermal expansion coefficient for seawater.
        sBeta : float
            Haline contraction coefficient for seawater.
    
        Return
        ----------
        dens : xarray.DataArray
            In-situ density that should be exactly the same as
            RHOAnoma + rhoRef
        """
        # cal. in-situ density using linear EOS
        RHO = rhoRef * (sBeta * (SALT-sRef) - tAlpha * (THETA - tRef)) + rhoRef
        
        RHO.rename('RHO')
        
        return RHO
    
    @staticmethod
    def cal_linear_buoyancy(rho, rhoRef=999.8):
        """
        Calculate buoyancy using linear EOS given potential density.
    
        Parameters
        ----------
        rho : DataArray
            Model-output in-situ density (kg m^-3).
        rhoRef: float
            Model-output reference density (kg m^-3).
    
        Return
        ----------
        dens : xarray.DataArray
            In-situ density that should be exactly the same as
            RHOAnoma + rhoRef
        """
        buoyancy = g * (rhoRef - rho) / rhoRef
        
        buoyancy.rename('buoyancy')
        
        return buoyancy
    
    
    def cal_pressure_potential(self, RHO, ETAN, PHrefC, rhoRef=999.8):
        """
        Calculate pressure potential (p/rhoConst) using model outputs.
    
        Parameters
        ----------
        RHO   : DataArray
            Model-output in-situ density (kg m^-3).
        ETAN  : DataArray
            Model-output sea surface height anomalies (m).
        PHrefC: DataArray
            Model-output reference pressure potential (p/rhoRef)
            (m^2 s^-2) at vertical grid cell.
        rhoRef: float or DataArray
            Model-output reference density (kg m^-3).
        g: float
            Gravitational acceleration (m s^-2)
    
        Return
        ----------
        PP : xarray.DataArray
            Hydrostatic pressure potential (m^2 s^-2) that should be exactly
            the same as PHIHYD + PHrefC output by MITgcm.
        """
        # get in-situ density at cell interface
        rhoF = self.grid.interp(RHO * self.coords.drF * self.coords.hFacC,
                                'Z', boundary='fill').load()
        # vertical integrate to get hydrostatic pressure potential
        pp   = self.grid.cumsum(rhoF, 'Z', boundary='fill') / rhoRef + ETAN
        
        pp *= g
        pp.rename('PRSPT')
        
        return pp
    
    @staticmethod
    def cal_bottom_pressure_potential(DEPTH, PHIBOT, rhoRef=999.8):
        """
        Calculate bottom pressure potential (p_bot/rhoConst)
        using model outputs.
    
        Parameters
        ----------
        DEPTH : DataArray
            Model-output depth (m).
        PHIBOT : DataArray
            Bottom pressure potential anomalies (m).
        rhoRef: float or DataArray
            Model-output reference density (kg m^-3).
        g: float
            Gravitational acceleration (m s^-2)
    
        Return
        ----------
        ppbot : xarray.DataArray
            Pressure potential (m^2 s^-2) at the bottom of the model.
        """
        ppbot = (PHIBOT + g * abs(DEPTH)).rename('ppbot')
        
        return ppbot


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in EOSUtils')
    
    # linear EOS test
    dset = xmitgcm.open_mdsdataset('I:/channel/', delta_t=300,
                                   prefix=['Stat3D', 'Surf'])
    
    metrics = {
        ('X',): ['dxC', 'dxG'], # X distances
        ('Y',): ['dyC', 'dyG'], # Y distances
        ('Z',): ['drF', 'drC'], # Z distances
        ('X', 'Y'): ['rA', 'rAs', 'rAw'] # Areas
    }
    
    grid=xgcm.Grid(dset, periodic=['X'], metrics=metrics)
    
    method = EOSMethods(dset)

    THETA = dset.THETA[0,:,1:-2,:].where(dset.maskC!=0).load()
    SALT  = 30
    PHrefC= dset.PHrefC.load()
    ETAN  = dset.ETAN[0,1:-2,:].where(dset.maskInC!=0).load()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    rho1 = method.cal_linear_insitu_density(THETA, SALT)
    rho2 = dset.RHOAnoma[0,:,1:-2,:].where(dset.maskC[0,1:-2,:]!=0).load() +\
            999.8
    
    (rho1 - rho2)[10,:,:].plot(ax=axes[0])

    pp1 = method.cal_pressure_potential(rho2, ETAN, PHrefC)
    pp2 = dset.PHIHYD[0,:,1:-2,:].where(dset.maskC[0,1:-2,:]!=0).load() +\
            PHrefC

    (pp1 - pp2)[10,:,:].plot(ax=axes[1])
    
    plt.tight_layout()
    
    
    
    # nonlinear EOS test
    # dset = xmitgcm.open_mdsdataset('D:/Data/MITgcm/heatBudget/daily/',
    #                                delta_t=300, prefix=['Stat'])
    
    # method = EOSMethods(dset)
    
    # THETA = dset.THETA [0].where(dset.maskC!=0).load()
    # SALT  = dset.SALT  [0].where(dset.maskC!=0).load()
    # PHrefC= dset.PHrefC.load()
    
    # rho1 = method.cal_insitu_density(THETA, SALT, PHrefC)
    # rho2 = dset.RHOAnoma[0] + 999.8
    
    # (rho1 - rho2)[10,:,:].plot()
    
    
    
    # pressure potential test
    # dset = xmitgcm.open_mdsdataset('I:/channel/',
    #                                 delta_t=300, prefix=['Stat3D', 'Surf'])
    # metrics = {
    #     ('X',): ['dxC', 'dxG'], # X distances
    #     ('Y',): ['dyC', 'dyG'], # Y distances
    #     ('Z',): ['drF', 'drC'], # Z distances
    #     ('X', 'Y'): ['rA', 'rAs', 'rAw'] # Areas
    # }
    
    # grid=xgcm.Grid(dset, periodic=['X'], metrics=metrics)

    # method = EOSMethods(dset)

    # THETA = dset.THETA[0,:,1:-2,:].where(dset.maskC!=0).load()
    # SALT  = 30
    # PHrefC= dset.PHrefC.load()
    # ETAN  = dset.ETAN[0,1:-2,:].where(dset.maskInC!=0).load()

    # rho = method.cal_linear_insitu_density(THETA, SALT)

    # pp1 = method.cal_pressure_potential(rho, ETAN, PHrefC)
    # pp2 = dset.PHIHYD[0,:,1:-2,:] + PHrefC

    # (pp1 - pp2)[20,:,:].plot()
