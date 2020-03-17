# -*- coding: utf-8 -*-
"""
Created on 2019.12.30

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
from GeoApps.Application import Application


class Dynamics(Application):
    """
    This class is designed for calculating the dynamical methods.
    """
    def __init__(self, dset, grid=None):
        """
        Construct a Dynamics instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        """
        super(Dynamics, self).__init__(dset, grid=grid)


    def cal_horizontal_divergence(self, u, v):
        """
        Calculate horizonal divergence as du/dx + dv/dy.
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        """
        coords = self.coords
        grid   = self.grid

        # get MITgcm diagnostics and calculate fluxes
        uflx = u * coords.dyG * coords.hFacW
        vflx = v * coords.dxG * coords.hFacS

        # difference to get flux convergence
        div = ((grid.diff(uflx, 'X', boundary=self.BCx) +
                grid.diff(vflx, 'Y', boundary=self.BCy)
              )/coords.rA/coords.hFacC).rename('div')

        return div


    def cal_vertical_vorticity(self, u, v):
        """
        Calculate vertical vorticity as dv/dx - du/dy.
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        """
        # get MITgcm diagnostics and calculate circulations
        ucir = u * self.coords.dxC
        vcir = v * self.coords.dyC

        # difference to get flux convergence
        vor = ((self.grid.diff(vcir, 'X', boundary=self.BCx) -
                self.grid.diff(ucir, 'Y', boundary=self.BCy)
              )/self.coords.rAz).rename('vor')

        return vor


    def cal_Laplacian(self, var):
        """
        Calculate Laplacian of var as %\nabla^2 q%.
        
        Parameters
        ----------
        var : xarray.DataArray
            A given variable.
        """
        der = self.grid.derivative

        lap = der(der(var, 'X', boundary=self.BCx), 'X', boundary=self.BCx) + \
              der(der(var, 'Y', boundary=self.BCy), 'Y', boundary=self.BCy)

        return lap


    def cal_strain(self, u, v):
        """
        Calculate strain as du/dx - dv/dy.
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        """
        coords = self.coords
        
        uflx = u * coords.dyG * coords.hFacW
        vflx = v * coords.dxG * coords.hFacS

        # difference to get strain
        strn = ((self.grid.diff(uflx, 'X', boundary=self.BCx) -
                 self.grid.diff(vflx, 'Y', boundary=self.BCy)
               )/coords.rA/coords.hFacC).rename('strn')

        return strn


    def cal_kinetic_energy(self, u, v):
        """
        Calculate kinetic energy.
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        """
        coords = self.coords
        
        uu = u * coords.hFacW
        vv = v * coords.hFacS

        # interpolate to get tracer-point energy
        KE = ((self.grid.interp(uu**2, 'X', boundary=self.BCx) +
               self.grid.interp(vv**2, 'Y', boundary=self.BCy)
             )/coords.hFacC * 0.5).rename('KE')

        return KE


    def cal_squared_gradient(self, var):
        """
        Calculate squared gradient magnitude.
        
        Parameters
        ----------
        var : xarray.DataArray
            A given variable to be defined at tracer point
            and thus the result.
        """
        grid = self.grid
        
        grdx = grid.derivative(var, 'X', boundary=self.BCx)
        grdy = grid.derivative(var, 'Y', boundary=self.BCy)

        # interpolate to get tracer-point gradient magnitude
        grdS = ((self.grid.interp(grdx**2, 'X', boundary=self.BCx) +
                 self.grid.interp(grdy**2, 'Y', boundary=self.BCy)
               )).rename('grdS'+var.name)
        
        return grdS


    """
    Helper (private) methods are defined below
    """


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in DiagnosticMethods')


