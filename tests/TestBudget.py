# -*- coding: utf-8 -*-
"""
Created on 2020.03.02

@author: MiniUFO
"""
#%%  loading data
import matplotlib.pyplot as plt
import xmitgcm
import xarray as xr
from GeoApps.GridUtils import add_MITgcm_missing_metrics

# os.environ['OMP_NUM_THREADS']='2'
indir = 'I:/channel/output_budget/'

deltaTmom = 300

dset = xmitgcm.open_mdsdataset(indir, grid_dir='I:/channel/', read_grid=True,
                    delta_t=deltaTmom,
                    prefix=['Stat3D', 'Surf2D', 'DiagU', 'DiagV', 'DiagT'])
#dset = xmitgcm.open_mdsdataset('F:/xixi/', read_grid=True,
#                    delta_t=deltaTmom,
#                    prefix=['Stat', 'Surf', 'DiagU', 'ViscU'])

dset, grid = add_MITgcm_missing_metrics(dset, periodic='X')


#%%  test heat budget
from GeoApps.Budget import HeatBudget

budget = HeatBudget(dset, grid)

# obtains variables
TOTTTEND = dset['TOTTTEND']

ADVx_TH  = dset['ADVx_TH' ]
ADVy_TH  = dset['ADVy_TH' ]
ADVr_TH  = dset['ADVr_TH' ]

DFxE_TH  = dset['DFxE_TH' ]
DFyE_TH  = dset['DFyE_TH' ]
DFrE_TH  = dset['DFrE_TH' ]
DFrI_TH  = dset['DFrI_TH' ]

KPPg_TH  = dset['KPPg_TH' ]
TFLUX    = dset['TFLUX'   ]
oceQsw   = dset['oceQsw'  ]
WTHMASS  = dset['WTHMASS' ]
#TRELAX   = dset['TRELAX'  ]

budget.cal_true_tendency(TOTTTEND)
budget.cal_advection_tendency(ADVx_TH, ADVy_TH, ADVr_TH)
budget.cal_diffusion_tendency(DFxE_TH, DFyE_TH, DFrE_TH, DFrI_TH)
budget.cal_nlocalKPP_tendency(KPPg_TH)
budget.cal_heating_flux_tendency(TFLUX, oceQsw)
budget.cal_linFSCorr_tendency(WTHMASS)

re = budget.terms

sums = (re.advct_tdc + re.diffu_tdc + re.nlKPP_tdc +
        re.Qsflx_tdc + re.Qtflx_tdc + re.corrt_tdc)


#%%  plotting heat budget
print('start  loading')
# data1 = re.isel(dict(XC=50, YC= 50)).sel(Z=-505).load() # interier ocean
# data2 = re.isel(dict(XC=50, YC=390)).sel(Z=-505).load() # northern BC
# data3 = re.isel(dict(XC=50, YC= 50)).sel(Z=-45 ).load() # subsurface
# data4 = re.isel(dict(XC=50, YC= 50)).sel(Z=-5  ).load() # surface
tmp = re.isel(XC=xr.DataArray([50, 50, 50, 50], dims='point'),
              YC=xr.DataArray([50,395, 50, 50], dims='point'),
              Z =xr.DataArray([14, 14,  3,  0], dims='point')).load()
print('finish loading')


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(14, 12))

data = tmp.isel(point=0)
sumup = (data['advct_tdc'] + data['diffu_tdc'] +
         data['nlKPP_tdc'] + data['Qsflx_tdc'])
data['advct_tdc'].plot(ax=ax1, color='m', linewidth=2, label='Advct tend')
data['diffu_tdc'].plot(ax=ax1, color='b', linewidth=2, label='Diffu tend')
data['nlKPP_tdc'].plot(ax=ax1, color='y', linewidth=2, label='nlKPP tend')
data['Qsflx_tdc'].plot(ax=ax1, color='g', linewidth=2, label='Qsflx tend')
data['total_tdc'].plot(ax=ax1, color='r', linewidth=5, label='True  tend')
sumup.plot(ax=ax1, color='k', linewidth=2, label='Sumup tend')
ax1.set_xlabel('Time [step]')
ax1.set_ylabel('tendency (K/day)')
ax1.set_title('budget at interier ocean')


data = tmp.isel(point=1)
sumup = (data['advct_tdc'] + data['diffu_tdc'] +
         data['nlKPP_tdc'] + data['Qsflx_tdc'])
data['advct_tdc'].plot(ax=ax2, color='m', linewidth=2, label='Advct tend')
data['diffu_tdc'].plot(ax=ax2, color='b', linewidth=2, label='Diffu tend')
data['nlKPP_tdc'].plot(ax=ax2, color='y', linewidth=2, label='nlKPP tend')
data['Qsflx_tdc'].plot(ax=ax2, color='g', linewidth=2, label='Qsflx tend')
data['total_tdc'].plot(ax=ax2, color='r', linewidth=5, label='True  tend')
sumup.plot(ax=ax2, color='k', linewidth=2, label='Sumup tend')
ax2.set_xlabel('Time [step]')
ax2.set_ylabel('tendency (K/day)')
ax2.set_title('budget near northern boundary')
ax2.legend(loc=[1.05, 0])


data = tmp.isel(point=2)
sumup = (data['advct_tdc'] + data['diffu_tdc'] +
         data['nlKPP_tdc'] + data['Qsflx_tdc'])
data['advct_tdc'].plot(ax=ax3, color='m', linewidth=2, label='Advct tend')
data['diffu_tdc'].plot(ax=ax3, color='b', linewidth=2, label='Diffu tend')
data['nlKPP_tdc'].plot(ax=ax3, color='y', linewidth=2, label='nlKPP tend')
data['Qsflx_tdc'].plot(ax=ax3, color='g', linewidth=2, label='Qsflx tend')
data['total_tdc'].plot(ax=ax3, color='r', linewidth=5, label='True  tend')
sumup.plot(ax=ax3, color='k', linewidth=2, label='Sumup tend')
ax3.set_xlabel('Time [step]')
ax3.set_ylabel('tendency (K/day)')
ax3.set_title('budget at subsurface')


data = tmp.isel(point=3).isel(Zl=0)
sumup = (data['advct_tdc'] + data['diffu_tdc'] +
         data['nlKPP_tdc'] + data['Qsflx_tdc'] +
         data['Qtflx_tdc'] + data['corrt_tdc'])
data['advct_tdc'].plot(ax=ax4, color='m', linewidth=2, label='Advct tend')
data['diffu_tdc'].plot(ax=ax4, color='b', linewidth=2, label='Diffu tend')
data['nlKPP_tdc'].plot(ax=ax4, color='y', linewidth=2, label='nlKPP tend')
data['Qsflx_tdc'].plot(ax=ax4, color='g', linewidth=2, label='Qsflx tend')
data['Qtflx_tdc'].plot(ax=ax4, color='orange', linewidth=2, label='Qtflx tend')
data['corrt_tdc'].plot(ax=ax4, color='violet', linewidth=2, label='corrt tend')
data['total_tdc'].plot(ax=ax4, color='r', linewidth=5, label='True  tend')
sumup.plot(ax=ax4, color='k', linewidth=2, label='Sumup tend')
ax4.set_xlabel('Time [step]')
ax4.set_ylabel('tendency (K/day)')
ax4.set_title('budget at surface')
ax4.legend(loc=[1.05, 0])




#%%  test U momentum budget
from GeoApps.Budget import MomentumBudget

budget = MomentumBudget(dset, grid, var='U')

# obtains variables
TOTUTEND = dset['TOTUTEND']

#ADVx_Um  = dset['ADVx_Um' ]
#ADVy_Um  = dset['ADVy_Um' ]
#ADVrE_Um = dset['ADVrE_Um']
Um_Advec = dset['Um_Advec']
# Um_Cori  = dset['Um_Cori' ] # if CD scheme is used

Um_Diss  = dset['Um_Diss' ]
#VISCx_Um = dset['VISCx_Um'] # all zeros for channal, vector-invariant?
#VISCy_Um = dset['VISCy_Um'] # all zeros for channal, vector-invariant?
#VISrE_Um = dset['VISrE_Um'] # all zeros for channal, vector-invariant?
VISrI_Um = dset['VISrI_Um']

Um_dPHdx = dset['Um_dPHdx']
# Um_dNPHdx= dset['Um_dNPdx'] # nonhydrostatic term
PHI_SURF = dset['PHI_SURF']

AB_gU    = dset['AB_gU'   ]
Um_Ext   = dset['Um_Ext'  ]

budget.cal_true_tendency(TOTUTEND)
budget.cal_advection_tendency(Um_Advec)
budget.cal_viscous_tendency(Um_Diss, VISrI_Um)
budget.cal_pressure_gradX_tendency(Um_dPHdx, None, PHI_SURF)
budget.cal_Adams_Bashforth_tendency(AB_gU)
budget.cal_external_forcing_tendency(Um_Ext)

re = budget.terms

#sums = re.advct_tdc+re.visco_tdc+re.pGrdx_tdc+re.AdamB_tdc+re.exter_tdc
#sums2 = (re.advct_tdc[0]+
#         re.visco_tdc[0]+
#         re.hydrx_tdc[0]+
#         re.phisx_tdc[1]+
#         re.AdamB_tdc[0]+
#         re.exter_tdc[0])
#
#diff = re.total_tdc - sums
#diff2 = re.total_tdc[0]-sums2


#%%  plotting U momentum budget
tidx = 0
zidx = slice(0,10)
yidx = 250 #slice(dset.YC.size)
xidx = 41# slice(dset.XC.size)

term1 = re.total_tdc[tidx,zidx,yidx,xidx]
term2 = re.advct_tdc[tidx,zidx,yidx,xidx]
term3 = re.visco_tdc[tidx,zidx,yidx,xidx]
term4 = re.pGrdx_tdc[tidx,zidx,yidx,xidx]
term5 = re.AdamB_tdc[tidx,zidx,yidx,xidx]
term6 = re.exter_tdc[tidx,zidx,yidx,xidx]
sumup = term2 + term3 + term4 + term5 + term6

# plotting the vertical profile of a single point
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

term4.plot(ax=axes[0], linewidth=3, color='m', label='press grad', y='Z')
term2.plot(ax=axes[0], linewidth=2, color='y', label='advec tend', y='Z')
term3.plot(ax=axes[0], linewidth=2, color='g', label='visco tend', y='Z')
term6.plot(ax=axes[0], linewidth=2, color='b', label='exter tend', y='Z')
term5.plot(ax=axes[0], linewidth=2, color='r', label='AB    tend', y='Z')

axes[0].set_ylim(dset.Z.values[9],0)
axes[0].set_xlabel('tendency (m s^-1 day^-1)')
axes[0].set_ylabel('depth (m)')

axes[0].set_title('momentum terms in channel model')
axes[0].legend(loc=[0.4,0.05])


term1.plot(ax=axes[1], linewidth=6,color='orange',label='true  tend',y='Z')
sumup.plot(ax=axes[1], linewidth=3,color='k'     ,label='sumup tend',y='Z')

# plt.xticks(dsr.time.values,['1','10','20','30','40','50','60','70','80'])
axes[1].set_ylim(dset.Z.values[9],0)
axes[1].set_xlabel('tendency (m s^-1 day^-1)')
axes[1].set_ylabel('depth (m)')

axes[1].set_title('momentum budget in channel model')
axes[1].legend(loc=[0.05,0.05])

plt.tight_layout()


#%%  test V momentum budget
from GeoApps.Budget import MomentumBudget

budget = MomentumBudget(dset, grid, var='V')

# obtains variables
TOTVTEND = dset['TOTVTEND']

ADVx_Vm  = dset['ADVx_Vm' ]
ADVy_Vm  = dset['ADVy_Vm' ]
ADVrE_Vm = dset['ADVrE_Vm']
Vm_Advec = dset['Vm_Advec']
# Um_Cori  = dset['Um_Cori' ] # if CD scheme is used

Vm_Diss  = dset['Vm_Diss' ]
VISCx_Vm = dset['VISCx_Vm'] # all zeros for channal, vector-invariant?
VISCy_Vm = dset['VISCy_Vm'] # all zeros for channal, vector-invariant?
VISrE_Vm = dset['VISrE_Vm'] # all zeros for channal, vector-invariant?
VISrI_Vm = dset['VISrI_Vm']

Vm_dPHdy = dset['Vm_dPHdy']
# Um_dNPHdx= dset['Um_dNPdx'] # nonhydrostatic term
PHI_SURF = dset['PHI_SURF'];

AB_gV    = dset['AB_gV'   ]
Vm_Ext   = dset['Vm_Ext'  ]

budget.cal_true_tendency(TOTVTEND)
budget.cal_advection_tendency(Vm_Advec)
budget.cal_viscous_tendency(Vm_Diss, VISrI_Vm)
budget.cal_pressure_gradY_tendency(Vm_dPHdy, None, PHI_SURF)
budget.cal_Adams_Bashforth_tendency(AB_gV)
budget.cal_external_forcing_tendency(Vm_Ext)

re = budget.terms

sums = re.advct_tdc+re.visco_tdc+re.pGrdy_tdc+re.AdamB_tdc+re.exter_tdc


#%%  plotting V momentum budget
tidx = 0
zidx = slice(1,20)
yidx = 325 #slice(dset.YC.size)
xidx = 30# slice(dset.XC.size)

term1 = re.total_tdc[tidx,zidx,yidx,xidx]
term2 = re.advct_tdc[tidx,zidx,yidx,xidx]
term3 = re.visco_tdc[tidx,zidx,yidx,xidx]
term4 = re.pGrdy_tdc[tidx,zidx,yidx,xidx]
term5 = re.AdamB_tdc[tidx,zidx,yidx,xidx]
term6 = re.exter_tdc[tidx,zidx,yidx,xidx]
sumup = term2 + term3 + term4 + term5 + term6

# plotting the vertical profile of a single point
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

term5.plot(ax=axes[0], linewidth=3, color='m', label='press grad', y='Z')
term2.plot(ax=axes[0], linewidth=2, color='y', label='advec tend', y='Z')
term3.plot(ax=axes[0], linewidth=2, color='g', label='dissi tend', y='Z')
term4.plot(ax=axes[0], linewidth=2, color='b', label='exter tend', y='Z')
term6.plot(ax=axes[0], linewidth=2, color='r', label='AB    tend', y='Z')

axes[0].set_ylim(dset.Z.values[9],0)
axes[0].set_xlabel('tendency (m s^-1 day^-1)')
axes[0].set_ylabel('depth (m)')

axes[0].set_title('momentum terms in channel model')
axes[0].legend(loc=[0.4,0.05])


term1.plot(ax=axes[1], linewidth=6,color='orange',label='true  tend',y='Z')
sumup.plot(ax=axes[1], linewidth=3,color='k'     ,label='sumup tend',y='Z')

# plt.xticks(dsr.time.values,['1','10','20','30','40','50','60','70','80'])
axes[1].set_ylim(dset.Z.values[9],0)
axes[1].set_xlabel('tendency (m s^-1 day^-1)')
axes[1].set_ylabel('depth (m)')

axes[1].set_title('momentum budget in channel model')
axes[1].legend(loc=[0.05,0.05])

plt.tight_layout()




#%%  test energy budget
from GeoApps.Budget import EnergyBudget

budget = EnergyBudget(dset, grid, hydrostatic=True)

# obtains variables
UVEL     = dset['UVEL'    ]
VVEL     = dset['VVEL'    ]
WVEL     = dset['WVEL'    ]
PHI_SURF = dset['PHI_SURF']
Um_dPHdx = dset['Um_dPHdx']
Vm_dPHdy = dset['Vm_dPHdy']
Um_Ext   = dset['Um_Ext'  ]
Vm_Ext   = dset['Vm_Ext'  ]
PHIHYD   = dset['PHIHYD'  ]
AB_gU    = dset['AB_gU'   ]
AB_gV    = dset['AB_gV'   ]
VISrI_Um = dset['VISrI_Um']
VISrI_Vm = dset['VISrI_Vm']

KE       = budget.cal_kinetic_energy(UVEL, VVEL, WVEL)

budget.cal_true_tendency(KE, 10800)

budget.cal_advection_tendency(UVEL, VVEL, WVEL, KE)
budget.cal_diffusion_tendency(KE, 12, 0)
budget.cal_surf_tendency(UVEL, VVEL, PHI_SURF)
budget.cal_pressure_tendency(UVEL, VVEL, Um_dPHdx, Vm_dPHdy)
budget.cal_dissipation_tendency(UVEL, VVEL, WVEL, 12, 0)
budget.cal_external_force_tendency(UVEL, VVEL, Um_Ext, Vm_Ext)
budget.cal_Adams_Bashforth_tendency(UVEL, VVEL, AB_gU, AB_gV)
budget.cal_viscousE_tendency(UVEL, VVEL, None, 12, 0)
budget.cal_viscousI_tendency(UVEL, VVEL, VISrI_Um, VISrI_Vm)

re = budget.terms

re.coords['XC'] = re.coords['XC'] / 1000 # convert to km
re.coords['YC'] = re.coords['YC'] / 1000 # convert to km


#%%  plotting energy budget terms
zidx = 0

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

axes = axes.ravel()

terms = [re.advct_tdc,
         (re.viscE_tdc+re.viscI_tdc).rename('visco_tdc'),
         re.prsSF_tdc,
         re.prsGF_tdc,
         re.exfrc_tdc,
         re.AdamB_tdc]

sumup = (re.advct_tdc +
         re.viscE_tdc + re.viscI_tdc +
         re.prsSF_tdc +
         re.prsGF_tdc +
         re.exfrc_tdc +
         re.AdamB_tdc)

for i, var in enumerate(terms):
    var[2, zidx, 1:-2, :].plot(ax=axes[i], cmap='bwr', add_colorbar=True)
    axes[i].set_title("{0}, Z={1}".format(var.name, dset.Z[zidx].values))

plt.tight_layout()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

budget.terms.KE[1, zidx, 1:-2, :].plot(ax=axes[0], cmap='bwr', add_colorbar=True)
axes[0].set_title("{0}, Z={1}".format('KE', dset.Z[zidx].values))
budget.terms.total_tdc[1, zidx, 1:-2, :].plot(ax=axes[1], cmap='bwr',
                                              add_colorbar=True, vmin=-0.5, vmax=0.5)
axes[1].set_title("{0}, Z={1}".format('KE tendency', dset.Z[zidx].values))
sumup[1, zidx, 1:-2, :].plot(ax=axes[2], cmap='bwr',
                             add_colorbar=True, vmin=-0.5, vmax=0.5)
axes[2].set_title("{0}, Z={1}".format('sumup', dset.Z[zidx].values))

plt.tight_layout()



#%%  debug energy equations w.r.t momentum equation
from GeoApps.Budget import EnergyBudget
from GeoApps.Budget import MomentumBudget

budgetU = MomentumBudget(dset, grid, var='U')
budgetV = MomentumBudget(dset, grid, var='V')
budgetE = EnergyBudget(dset, grid, hydrostatic=True)

# obtains variables
TOTUTEND = dset['TOTUTEND']
TOTVTEND = dset['TOTVTEND']

Um_Advec = dset['Um_Advec']
Vm_Advec = dset['Vm_Advec']

Um_Diss  = dset['Um_Diss' ]
VISrI_Um = dset['VISrI_Um']

Vm_Diss  = dset['Vm_Diss' ]
VISrI_Vm = dset['VISrI_Vm']

Um_dPHdx = dset['Um_dPHdx']
Vm_dPHdy = dset['Vm_dPHdy']
PHI_SURF = dset['PHI_SURF']

AB_gU    = dset['AB_gU'   ]
AB_gV    = dset['AB_gV'   ]
Um_Ext   = dset['Um_Ext'  ]
Vm_Ext   = dset['Vm_Ext'  ]
UVEL     = dset['UVEL'    ]
VVEL     = dset['VVEL'    ]
WVEL     = dset['WVEL'    ]
Um_Ext   = dset['Um_Ext'  ]
Vm_Ext   = dset['Vm_Ext'  ]
RHOAnoma = dset['RHOAnoma']
PHIHYD   = dset['PHIHYD'  ]


KE       = budgetE.cal_kinetic_energy(UVEL, VVEL, WVEL)


budgetU.cal_true_tendency(TOTUTEND)
budgetU.cal_advection_tendency(Um_Advec)
budgetU.cal_viscous_tendency(Um_Diss, VISrI_Um)
budgetU.cal_pressure_gradX_tendency(Um_dPHdx, None, PHI_SURF)
budgetU.cal_Adams_Bashforth_tendency(AB_gU)
budgetU.cal_external_forcing_tendency(Um_Ext)
budgetU.calM_viscous_tendency(UVEL, VVEL, 12, 3e-4)
reU = budgetU.terms


budgetV.cal_true_tendency(TOTVTEND)
budgetV.cal_advection_tendency(Vm_Advec)
budgetV.cal_viscous_tendency(Vm_Diss, VISrI_Vm)
budgetV.cal_pressure_gradY_tendency(Vm_dPHdy, None, PHI_SURF)
budgetV.cal_Adams_Bashforth_tendency(AB_gV)
budgetV.cal_external_forcing_tendency(Vm_Ext)
reV = budgetV.terms


budgetE.cal_true_tendency(KE, 10800)
budgetE.cal_advection_tendency(KE, UVEL, VVEL, WVEL)
budgetE.cal_diffusion_tendency(KE, 12, 3e-4)
#budgetE.cal_conversion_tendency(WVEL, RHOAnoma)
budgetE.cal_surf_tendency(UVEL, VVEL, PHI_SURF)
budgetE.cal_pressure_tendency(UVEL, VVEL, Um_dPHdx, Vm_dPHdy)
budgetE.cal_dissipation_tendency(UVEL, VVEL, WVEL, 12, 3e-4)
budgetE.cal_external_force_tendency(UVEL, VVEL, Um_Ext, Vm_Ext)
budgetE.cal_Adams_Bashforth_tendency(UVEL, VVEL, AB_gU, AB_gV)
budgetE.cal_viscousE_tendency(UVEL, VVEL, None, 12, 0)
budgetE.cal_viscousI_tendency(UVEL, VVEL, VISrI_Um, VISrI_Vm)
reE = budgetE.terms


#%%
total_tdc = (grid.interp(UVEL * reU.total_tdc, 'X', boundary='fill') +
             grid.interp(VVEL * reV.total_tdc, 'Y', boundary='fill'))

advct_tdc = (grid.interp(UVEL * reU.advct_tdc, 'X', boundary='fill') +
             grid.interp(VVEL * reV.advct_tdc, 'Y', boundary='fill'))

exfrc_tdc = (grid.interp(UVEL * reU.exter_tdc, 'X', boundary='fill') +
             grid.interp(VVEL * reV.exter_tdc, 'Y', boundary='fill'))

AdamB_tdc = (grid.interp(UVEL * reU.AdamB_tdc, 'X', boundary='fill') +
             grid.interp(VVEL * reV.AdamB_tdc, 'Y', boundary='fill'))

prsGF_tdc = (grid.interp(UVEL * reU.hydrx_tdc, 'X', boundary='fill') +
             grid.interp(VVEL * reV.hydry_tdc, 'Y', boundary='fill'))

phisf_tdc = (grid.interp(UVEL * reU.phisx_tdc, 'X', boundary='fill') +
             grid.interp(VVEL * reV.phisy_tdc, 'Y', boundary='fill'))

viscE_tdc = (grid.interp(UVEL * reU.viscE_tdc, 'X', boundary='fill') +
             grid.interp(VVEL * reV.viscE_tdc, 'Y', boundary='fill'))

viscI_tdc = (grid.interp(UVEL * reU.viscI_tdc, 'X', boundary='fill') +
             grid.interp(VVEL * reV.viscI_tdc, 'Y', boundary='fill'))

visco_tdc = (grid.interp(UVEL * reU.visco_tdc, 'X', boundary='fill') +
             grid.interp(VVEL * reV.visco_tdc, 'Y', boundary='fill'))

sumup = advct_tdc + exfrc_tdc + AdamB_tdc + prsGF_tdc + phisf_tdc + visco_tdc

sumupU = reU.advct_tdc + reU.exter_tdc + reU.AdamB_tdc + \
         reU.hydrx_tdc + reU.phisx_tdc + reU.visco_tdc

sumupV = reV.advct_tdc + reV.exter_tdc + reV.AdamB_tdc + \
         reV.hydry_tdc + reV.phisy_tdc + reV.visco_tdc

# reE.coords['XC'] /= 1000 # convert to km
# reE.coords['YC'] /= 1000 # convert to km
# total_tdc.coords['XC'] /= 1000
# total_tdc.coords['YC'] /= 1000
# advct_tdc.coords['XC'] /= 1000
# advct_tdc.coords['YC'] /= 1000
# exfrc_tdc.coords['XC'] /= 1000
# exfrc_tdc.coords['YC'] /= 1000
# AdamB_tdc.coords['XC'] /= 1000
# AdamB_tdc.coords['YC'] /= 1000
# prsGF_tdc.coords['XC'] /= 1000
# prsGF_tdc.coords['YC'] /= 1000
# phisf_tdc.coords['XC'] /= 1000
# phisf_tdc.coords['YC'] /= 1000
# viscE_tdc.coords['XC'] /= 1000
# viscE_tdc.coords['YC'] /= 1000
# viscI_tdc.coords['XC'] /= 1000
# viscI_tdc.coords['YC'] /= 1000
# visco_tdc.coords['XC'] /= 1000
# visco_tdc.coords['YC'] /= 1000

#%%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

reE.total_tdc[1,0].plot(ax=axes[0], vmin=-0.5, vmax=0.5, cmap='bwr')
axes[0].set_title('true tendency (left)')
total_tdc[1,0].plot(ax=axes[1], vmin=-0.5, vmax=0.5, cmap='bwr')
axes[1].set_title('true tendency (right)')
(reE.total_tdc-total_tdc)[1,0].plot(ax=axes[2], vmin=-0.5, vmax=0.5, cmap='bwr')
axes[2].set_title('true tendency (diff)')

plt.tight_layout()

#%%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

reE.advct_tdc[1,0].plot(ax=axes[0], vmin=-0.8, vmax=0.8, cmap='bwr')
axes[0].set_title('advective tendency (left)')
advct_tdc[1,0].plot(ax=axes[1], vmin=-0.8, vmax=0.8, cmap='bwr')
axes[1].set_title('advective tendency (right)')
(reE.advct_tdc-advct_tdc)[1,0].plot(ax=axes[2], vmin=-0.8, vmax=0.8, cmap='bwr')
axes[2].set_title('advective tendency (diff)')

plt.tight_layout()

#%%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

reE.exfrc_tdc[1,0].plot(ax=axes[0], vmin=-1.2, vmax=1.2, cmap='bwr')
axes[0].set_title('external tendency (left)')
exfrc_tdc[1,0].plot(ax=axes[1], vmin=-1.2, vmax=1.2, cmap='bwr')
axes[1].set_title('external tendency (right)')
(reE.exfrc_tdc-exfrc_tdc)[1,0].plot(ax=axes[2], vmin=-1.2, vmax=1.2, cmap='bwr')
axes[2].set_title('external tendency (diff)')

plt.tight_layout()

#%%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

reE.AdamB_tdc[1,0].plot(ax=axes[0], vmin=-0.006, vmax=0.006, cmap='bwr')
axes[0].set_title('AdamB tendency (left)')
AdamB_tdc[1,0].plot(ax=axes[1], vmin=-0.006, vmax=0.006, cmap='bwr')
axes[1].set_title('AdamB tendency (right)')
(reE.AdamB_tdc-AdamB_tdc)[1,0].plot(ax=axes[2], vmin=-0.006, vmax=0.006, cmap='bwr')
axes[2].set_title('AdamB tendency (diff)')

plt.tight_layout()

#%%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

reE.prsGF_tdc[1,0].plot(ax=axes[0], vmin=-0.01, vmax=0.01, cmap='bwr')
axes[0].set_title('hydro PG tendency (left)')
prsGF_tdc[1,0].plot(ax=axes[1], vmin=-0.01, vmax=0.01, cmap='bwr')
axes[1].set_title('hydro PG tendency (right)')
(reE.prsGF_tdc-prsGF_tdc)[1,0].plot(ax=axes[2], vmin=-0.01, vmax=0.01, cmap='bwr')
axes[2].set_title('hydro PG tendency (diff)')

plt.tight_layout()

#%%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

reE.prsSF_tdc[1,0].plot(ax=axes[0], vmin=-1, vmax=1, cmap='bwr')
axes[0].set_title('surface PG tendency (left)')
phisf_tdc[1,0].plot(ax=axes[1], vmin=-1, vmax=1, cmap='bwr')
axes[1].set_title('surface PG tendency (right)')
(reE.prsSF_tdc-phisf_tdc)[1,0].plot(ax=axes[2], vmin=-1, vmax=1, cmap='bwr')
axes[2].set_title('surface PG tendency (diff)')

plt.tight_layout()

#%%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

reE.viscE_tdc[1,0].plot(ax=axes[0], vmin=-0.01, vmax=0.01, cmap='bwr')
axes[0].set_title('viscousE tendency (left)')
viscE_tdc[1,0].plot(ax=axes[1], vmin=-0.01, vmax=0.01, cmap='bwr')
axes[1].set_title('viscousE tendency (right)')
(reE.viscE_tdc-viscE_tdc)[1,0].plot(ax=axes[2], vmin=-0.01, vmax=0.01, cmap='bwr')
axes[2].set_title('viscousE tendency (diff)')

plt.tight_layout()

#%%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

reE.viscI_tdc[1,0].plot(ax=axes[0], vmin=-1, vmax=1, cmap='bwr')
axes[0].set_title('viscousI tendency (left)')
viscI_tdc[1,0].plot(ax=axes[1], vmin=-1, vmax=1, cmap='bwr')
axes[1].set_title('viscousI tendency (right)')
(reE.viscI_tdc-viscI_tdc)[1,0].plot(ax=axes[2], vmin=-1, vmax=1, cmap='bwr')
axes[2].set_title('viscousI tendency (diff)')

plt.tight_layout()

#%%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

(reE.diffx_tdc+reE.diffy_tdc+reE.dissi_tdc)[1,0].plot(ax=axes[0], vmin=-0.01, vmax=0.01, cmap='bwr')
axes[0].set_title('viscousE tendency (left)')
(reE.visUE_tdc+reE.visVE_tdc)[1,0].plot(ax=axes[1], vmin=-0.01, vmax=0.01, cmap='bwr')
axes[1].set_title('viscousE tendency (right)')
((reE.diffx_tdc+reE.diffy_tdc+reE.dissi_tdc)-(reE.visUE_tdc+reE.visVE_tdc))[1,0].plot(ax=axes[2], vmin=-0.01, vmax=0.01, cmap='bwr')
axes[2].set_title('viscousE tendency (diff)')

plt.tight_layout()

