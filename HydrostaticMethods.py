# -*- coding: utf-8 -*-
'''
Created on 2019.12.30

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
'''
import xgcm


class HydrostaticMethods(object):
    '''
    This class is designed for calculating the hydrostatic methods.
    '''
    def __init__(self, dset, grid=None):
        '''
        Construct a Dynamics instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        '''
        self.grid   = xgcm.Grid(dset) if grid is None else grid
        self.coords = dset.coords.to_dataset().reset_coords()
        self.dset   = dset.reset_coords(drop=True)
        self.volume = dset.drF * dset.hFacC * dset.rA
        self.terms  = None


    '''
    Calculation methods.
    '''
    def cal_pressure(self, height, T0=273.15+15, p0=101325, h0=0, L0=-0.0065):
        '''
        Calculate vertical pressure using height using barometric method.
        Reference: https://physics.stackexchange.com/questions/333475/
        
        Parameters
        ----------
        height : DataArray
            Height levels (m).
        '''
        p = p0 * (T0/(T0 + L0 * (height - h0)))**5.25579
        
        return p
    
    def cal_height(self, press, temp, p0=101325, formula='barometric'):
        '''
        Calculate vertical height using pressure.
        Reference: https://physics.stackexchange.com/questions/333475/
        
        Parameters
        ----------
        press  : DataArray
            Pressure levels (Pa).
        temp   : DataArray
            Temperature (K).
        formula: str
            Possible formula, in ['barometric', 'hypsometric'].
        '''
        if formula == 'hypsometric':
            h = (((p0/press)**(1.0/5.257) -1) * temp) / 0.0065
        elif formula == 'barometric':
            h = 44330 * (1.0 - (press/p0)**(1.0/5.25579))
        
        return h


    '''
    Helper (private) methods are defined below
    '''


'''
Testing codes for each class
'''
if __name__ == '__main__':
    print('start testing in HydrostaticMethods')


