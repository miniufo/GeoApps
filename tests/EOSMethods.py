# -*- coding: utf-8 -*-
"""
Created on 2020.03.16

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import xmitgcm
import matplotlib.pyplot as plt
from GeoApps.GridUtils import add_MITgcm_missing_metrics
from GeoApps.EOSMethods import EOSMethods


#%%  linear EOS test
dset = xmitgcm.open_mdsdataset('I:/channel/', delta_t=300,
                               prefix=['Stat3D', 'Surf'])

dset, grid = add_MITgcm_missing_metrics(dset, periodic='X')

method = EOSMethods(dset)

THETA = dset.THETA[0,:,1:-2,:].where(dset.maskC!=0).load()
SALT  = 30
PHrefC= dset.PHrefC.load()
ETAN  = dset.ETAN[0,1:-2,:].where(dset.maskInC!=0).load()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

rho1 = method.cal_linear_insitu_density(THETA, SALT)
rho2 = dset.RHOAnoma[0,:,1:-2,:].where(dset.maskC[0,1:-2,:]!=0).load() + 999.8

(rho1 - rho2)[10,:,:].plot(ax=axes[0])

pp1 = method.cal_pressure_potential(rho2, ETAN, PHrefC)
pp2 = dset.PHIHYD[0,:,1:-2,:].where(dset.maskC[0,1:-2,:]!=0).load() + PHrefC

(pp1 - pp2)[10,:,:].plot(ax=axes[1])

plt.tight_layout()



#%%  nonlinear EOS test
dset = xmitgcm.open_mdsdataset('D:/Data/MITgcm/heatBudget/daily/',
                                delta_t=300, prefix=['Stat'])

method = EOSMethods(dset)

THETA = dset.THETA [0].where(dset.maskC!=0).load()
SALT  = dset.SALT  [0].where(dset.maskC!=0).load()
PHrefC= dset.PHrefC.load()

rho1 = method.cal_insitu_density(THETA, SALT, PHrefC)
rho2 = dset.RHOAnoma[0] + 999.8

(rho1 - rho2)[10,:,:].plot()



#%%  pressure potential test
dset = xmitgcm.open_mdsdataset('I:/channel/',
                                delta_t=300, prefix=['Stat3D', 'Surf'])
metrics = {
    ('X',): ['dxC', 'dxG'], # X distances
    ('Y',): ['dyC', 'dyG'], # Y distances
    ('Z',): ['drF', 'drC'], # Z distances
    ('X', 'Y'): ['rA', 'rAs', 'rAw'] # Areas
}

dset, grid = add_MITgcm_missing_metrics(dset, periodic='X')

method = EOSMethods(dset)

THETA = dset.THETA[0,:,1:-2,:].where(dset.maskC!=0).load()
SALT  = 30
PHrefC= dset.PHrefC.load()
ETAN  = dset.ETAN[0,1:-2,:].where(dset.maskInC!=0).load()

rho = method.cal_linear_insitu_density(THETA, SALT)

pp1 = method.cal_pressure_potential(rho, ETAN, PHrefC)
pp2 = dset.PHIHYD[0,:,1:-2,:] + PHrefC

(pp1 - pp2)[20,:,:].plot()

