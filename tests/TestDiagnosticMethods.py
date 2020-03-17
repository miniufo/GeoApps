# -*- coding: utf-8 -*-
"""
Created on 2019.12.30

@author: MiniUFO
"""
#%%
import matplotlib.pyplot as plt
import xmitgcm
import xgcm
from utils.DiagnosticMethods import Dynamics

#os.environ['OMP_NUM_THREADS']='2'
indir='H:/channel/'

deltaTmom=300

dset=xmitgcm.open_mdsdataset(indir, grid_dir=indir, read_grid=True,
                             delta_t=deltaTmom, prefix=['Stat3D'])

grid=xgcm.Grid(dset, periodic=['X'])
coords=dset.coords.to_dataset().reset_coords()
#dsr=ds.reset_coords(drop=True)

dyn = Dynamics(dset)


#%%
# for divergence
div1 = dyn.cal_horizontal_divergence('UVEL', 'VVEL')
div2 = dset.momHDiv

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14,6), sharey=True)

dvg1 = div1[0,0,:,:]*1e5
dvg2 = div2[0,0,:,:]*1e5

dvg1.plot(ax=ax1, vmin=-2, vmax=2, cmap='bwr')
dvg2.plot(ax=ax2, vmin=-2, vmax=2, cmap='bwr')
(dvg1-dvg2).plot(ax=ax3, cmap='bwr')

ax1.set(title='offline div', xlabel='x-coord [km]', ylabel='y-coord [km]')
ax2.set(title='online div' , xlabel='x-coord [km]', ylabel='y-coord [km]')
ax3.set(title='difference' , xlabel='x-coord [km]', ylabel='y-coord [km]')

for ax in (ax1,ax2,ax3):
    ax.get_xaxis().set_ticklabels([0,200,400,600,800,1000])
    ax.get_yaxis().set_ticklabels([0,250,500,750,1000,1250,1500,1750,2000])

plt.tight_layout()

vor1 = dyn.cal_vertical_vorticity('UVEL', 'VVEL')
vor2 = dset.momVort3

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(11,11))

dvg1[:,0].plot(ax=ax1, y='YC', linewidth=3, linestyle='-', color='r', xlim=[-1, 1], ylim=[0, 2000000])
dvg2[:,0].plot(ax=ax1, y='YC', linewidth=1, linestyle='-', color='b', xlim=[-1, 1], ylim=[0, 2000000])
ax1.set(title='east boundary')

dvg1[:,-1].plot(ax=ax2, y='YC', linewidth=3, linestyle='-', color='r', xlim=[-1, 1], ylim=[0, 2000000])
dvg2[:,-1].plot(ax=ax2, y='YC', linewidth=1, linestyle='-', color='b', xlim=[-1, 1], ylim=[0, 2000000])
ax2.set(title='west boundary')

dvg1[0,:].plot(ax=ax3, linewidth=3, linestyle='-', color='r', xlim=[0, 1000000], ylim=[-1, 1])
dvg2[0,:].plot(ax=ax3, linewidth=1, linestyle='-', color='b', xlim=[0, 1000000], ylim=[-1, 1])
ax3.set(title='south boundary')

dvg1[-1,:].plot(ax=ax4, linewidth=3, linestyle='-', color='r', xlim=[0, 1000000], ylim=[-1, 1])
dvg2[-1,:].plot(ax=ax4, linewidth=1, linestyle='-', color='b', xlim=[0, 1000000], ylim=[-1, 1])
ax4.set(title='north boundary')

plt.tight_layout()



#%%
# for vorticity
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14,6), sharey=True)

vrt1 = vor1[0,0,:,:]*1e5
vrt2 = vor2[0,0,:,:]*1e5

vrt1.plot(ax=ax1, vmin=-4, vmax=4, cmap='bwr')
vrt2.plot(ax=ax2, vmin=-4, vmax=4, cmap='bwr')
(vrt1-vrt2).plot(ax=ax3, cmap='bwr')

ax1.set(title='offline vor', xlabel='x-coord [km]', ylabel='y-coord [km]')
ax2.set(title='online vor' , xlabel='x-coord [km]', ylabel='y-coord [km]')
ax3.set(title='difference' , xlabel='x-coord [km]', ylabel='y-coord [km]')

for ax in (ax1,ax2,ax3):
    ax.get_xaxis().set_ticklabels([0,200,400,600,800,1000])
    ax.get_yaxis().set_ticklabels([0,250,500,750,1000,1250,1500,1750,2000])

plt.tight_layout()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(11,11))

vrt1[:,0].plot(ax=ax1, y='YG', linewidth=3, linestyle='-', color='r', xlim=[-4, 4], ylim=[0, 2000000])
vrt2[:,0].plot(ax=ax1, y='YG', linewidth=1, linestyle='-', color='b', xlim=[-4, 4], ylim=[0, 2000000])
ax1.set(title='east boundary')

vrt1[:,-1].plot(ax=ax2, y='YG', linewidth=3, linestyle='-', color='r', xlim=[-4, 4], ylim=[0, 2000000])
vrt2[:,-1].plot(ax=ax2, y='YG', linewidth=1, linestyle='-', color='b', xlim=[-4, 4], ylim=[0, 2000000])
ax2.set(title='west boundary')

vrt1[0,:].plot(ax=ax3, linewidth=3, linestyle='-', color='r', xlim=[0, 1000000], ylim=[-4, 4])
vrt2[0,:].plot(ax=ax3, linewidth=1, linestyle='-', color='b', xlim=[0, 1000000], ylim=[-4, 4])
ax3.set(title='south boundary')

vrt1[-1,:].plot(ax=ax4, linewidth=3, linestyle='-', color='r', xlim=[0, 1000000], ylim=[-4, 4])
vrt2[-1,:].plot(ax=ax4, linewidth=1, linestyle='-', color='b', xlim=[0, 1000000], ylim=[-4, 4])
ax4.set(title='north boundary')

plt.tight_layout()


#%%
# for kinetic energy
KE1  = dyn.cal_kinetic_energy('UVEL', 'VVEL')
KE2  = dset.momKE

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14,6), sharey=True)

KEG1 = KE1[0,0,:,:]
KEG2 = KE2[0,0,:,:]

KEG1.plot(ax=ax1, vmin=0, vmax=2, cmap='Spectral_r')
KEG2.plot(ax=ax2, vmin=0, vmax=2, cmap='Spectral_r')
(KEG1-KEG2).plot(ax=ax3, vmin=-0.01, vmax=0.01, cmap='bwr')

ax1.set(title='offline kinetic energy', xlabel='x-coord [km]', ylabel='y-coord [km]')
ax2.set(title='online kinetic energy' , xlabel='x-coord [km]', ylabel='y-coord [km]')
ax3.set(title='difference' , xlabel='x-coord [km]', ylabel='y-coord [km]')

for ax in (ax1,ax2,ax3):
    ax.get_xaxis().set_ticklabels([0,200,400,600,800,1000])
    ax.get_yaxis().set_ticklabels([0,250,500,750,1000,1250,1500,1750,2000])

plt.tight_layout()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(11,11))

KEG1[:,0].plot(ax=ax1, y='YC', linewidth=3, linestyle='-', color='r', xlim=[0, 0.2], ylim=[0, 2000000], label='offline')
KEG2[:,0].plot(ax=ax1, y='YC', linewidth=1, linestyle='-', color='b', xlim=[0, 0.2], ylim=[0, 2000000], label='online')
ax1.set(title='east boundary')
ax1.legend(loc=[0.7,0.1])

KEG1[:,-1].plot(ax=ax2, y='YC', linewidth=3, linestyle='-', color='r', xlim=[0, 0.2], ylim=[0, 2000000], label='offline')
KEG2[:,-1].plot(ax=ax2, y='YC', linewidth=1, linestyle='-', color='b', xlim=[0, 0.2], ylim=[0, 2000000], label='online')
ax2.set(title='west boundary')
ax2.legend(loc=[0.7,0.1])

KEG1[0,:].plot(ax=ax3, linewidth=3, linestyle='-', color='r', xlim=[0, 1000000], ylim=[0, 0.2], label='offline')
KEG2[0,:].plot(ax=ax3, linewidth=1, linestyle='-', color='b', xlim=[0, 1000000], ylim=[0, 0.2], label='online')
ax3.set(title='south boundary')
ax3.legend(loc=[0.7,0.1])

KEG1[-1,:].plot(ax=ax4, linewidth=3, linestyle='-', color='r', xlim=[0, 1000000], ylim=[0, 0.2], label='offline')
KEG2[-1,:].plot(ax=ax4, linewidth=1, linestyle='-', color='b', xlim=[0, 1000000], ylim=[0, 0.2], label='online')
ax4.set(title='north boundary')
ax4.legend(loc=[0.7,0.1])

plt.tight_layout()