# -*- coding: utf-8 -*-
"""
Created on 2019.12.13

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import numpy  as np
import xarray as xr
from utils.GridUtils import add_latlon_metrics
from utils.DiagnosticMethods import Dynamics
from utils.ContourMethods import ContourAnalysisInLatLon

N = 121

lon = 'longitude'
lat = 'latitude'

preY = np.linspace(-90, 90, N)

trName = 'pv'


# dset = open_CtlDataset('D:/Data/ERAInterim/Keff/PV/PV2.ctl')\
#    .isel(dict(time=0,lev=0))

path = 'D:/Data/ERAInterim/BKGState/OriginalData/PT/'

dset = xr.open_dataset(path + 'PV.nc').isel(time=0).load()
dset, grid = add_latlon_metrics(dset.sortby(dset[lat], ascending=True))
# dset, grid = add_latlon_metrics(dset)

# pv = dset.pv

# prof = xr.DataArray(np.sin(np.deg2rad(np.linspace(-90,90,241))),
#                     dims='latitude',
#                     coords={'latitude':np.linspace(-90,90,241)})

# dset.pv.loc[:] = pv - pv + prof

grdS = Dynamics(dset, grid).cal_squared_gradient(trName)


#%%
import numpy  as np
import xmitgcm
import xarray as xr
from xgcm import Grid
from utils.DiagnosticMethods import Dynamics
from utils.ContourMethods import ContourAnalysisInCartesian

N = 121

lon = 'XC'
lat = 'YC'

preY = np.linspace(0, 5500*399, N)

trName = 'TRAC09'

path = 'D:/Data/MITgcm/barotropicDG/BetaCartRL/Leith1_k0/'
dset = xmitgcm.open_mdsdataset(path, delta_t=300,
                               prefix=['Stat'])

# change 0 to nan
dset = dset.where(dset!=0)

metrics = {
    ('X',): ['dxC', 'dxG'],  # X distances
    ('Y',): ['dyC', 'dyG'],  # Y distances
    ('Z',): ['drF', 'drC'],  # Z distances
    # ('Z',    ): ['drW', 'drS', 'drC'], # Z distances
    ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw']  # Areas
}


grid = Grid(dset, periodic=[], metrics=metrics)
grdS = Dynamics(dset, grid).cal_squared_gradient(trName)


#%%

# test linear contour levels
cm = ContourAnalysisInLatLon(dset, trName, grid)


ctr     = cm.cal_contours(N, dims=[lat,lon])
area    = cm.cal_integral_within_contours(ctr, name='intArea')
intgrdS = cm.cal_integral_within_contours(ctr, grdS, name='intgrdS')
latEq   = cm.cal_equivalent_coords(area)

dintgrdSdA = cm.cal_gradient_wrt_area(intgrdS, area)
dqdA    = cm.cal_gradient_wrt_area(ctr, area)
Leq2    = cm.cal_sqared_equivalent_length(dintgrdSdA, dqdA)
Lmin    = cm.cal_minimum_possible_length(latEq)
nkeff   = cm.cal_normalized_Keff(Leq2, Lmin)

vs = [ctr, area, intgrdS, latEq, dintgrdSdA, dqdA, Leq2, Lmin, nkeff]

origin = xr.merge(vs)
interp = cm.interp_to_dataset(preY, latEq, origin)

# test adjusting contour levels
cm2 = ContourAnalysisInCartesian(dset, trName, grid)


ctr2     = cm2.cal_contours_at(preY, dims=[lat,lon])
area2    = cm2.cal_integral_within_contours(ctr2, name='intArea')
intgrdS2 = cm2.cal_integral_within_contours(ctr2, grdS, name='intgrdS')
latEq2   = cm2.cal_equivalent_coords(area2)

dintgrdSdA2 = cm2.cal_gradient_wrt_area(intgrdS2, area2)
dqdA2    = cm2.cal_gradient_wrt_area(ctr2, area2)
Leq22    = cm2.cal_sqared_equivalent_length(dintgrdSdA2, dqdA2)
Lmin2    = cm2.cal_minimum_possible_length(latEq2)
nkeff2   = cm2.cal_normalized_Keff(Leq22, Lmin2)

vs2 = [ctr2, area2, intgrdS2, latEq2, dintgrdSdA2, dqdA2, Leq22, Lmin2, nkeff2]

origin2 = xr.merge(vs2)
interp2 = cm2.interp_to_dataset(preY, latEq2, origin2)

