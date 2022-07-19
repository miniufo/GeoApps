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
    def __init__(self, dset, grid=None, arakawa='A'):
        """
        Construct a Budget instance using a Dataset
        
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
        self.grid    = xgcm.Grid(dset) if grid is None else grid
        self.coords  = dset.coords.to_dataset().reset_coords()
        self.dset    = dset
        self.arakawa = arakawa
        # self.BC      = {}
        
        # self.dset   = dset.reset_coords(drop=True)
        # self.volume = dset.drF * dset.hFacC * dset.rA
        
        # if 'X' in self.grid.axes:
        #     self.BC['X'] = 'periodic' if grid.axes['X']._periodic else 'extend'
        # if 'Y' in self.grid.axes:
        #     self.BC['Y'] = 'periodic' if grid.axes['Y']._periodic else 'extend'
        # if 'Z' in self.grid.axes:
        #     self.BC['Z'] = 'periodic' if grid.axes['Z']._periodic else 'extend'


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in Application')
