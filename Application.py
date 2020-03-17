# -*- coding: utf-8 -*-
"""
Created on 2020.03.15

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import xgcm


class Application(object):
    """
    This class is the base class for any geofuild applications
    built on a given `xarray.Dataset` and related `xgcm.Grid`.
    """
    def __init__(self, dset, grid=None):
        """
        Construct a Budget instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        
        Return
        ----------
        terms : xarray.Dataset
            A Dataset containing all budget terms
        """
        self.grid   = xgcm.Grid(dset) if grid is None else grid
        self.coords = dset.coords.to_dataset().reset_coords()
        self.dset   = dset
        self.terms  = None
        
        # self.dset   = dset.reset_coords(drop=True)
        # self.volume = dset.drF * dset.hFacC * dset.rA
        
        self.BCx = 'periodic' if self.grid.axes['X']._periodic else 'fill'
        self.BCy = 'periodic' if self.grid.axes['Y']._periodic else 'fill'


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in Application')
