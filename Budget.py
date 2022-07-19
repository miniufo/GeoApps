# -*- coding: utf-8 -*-
"""
Created on 2019.08.16

@author: MiniUFO, Emily
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from GeoApps.Application import Application


class Budget(Application):
    """
    This class is the base class.
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
        super().__init__(dset, grid=grid)
        


    """
    Calculate some common terms in every budget.
    """
    def cal_advection_tendency(self, ADVx, ADVy, ADVr):
        """
        Calculate tendency due to advection.
        
        Parameters
        ----------
        ADVx : xarray.DataArray
            Advective flux of some stuff ['TH', 'SLT', 'TRAC01'] for
            heat, salt, and passive tracer along x-dimension.
        ADVy : xarray.DataArray
            Advective flux of some stuff ['TH', 'SLT', 'TRAC01'] for
            heat, salt, and passive tracer along y-dimension.
        ADVr : xarray.DataArray
            Advective flux of some stuff ['TH', 'SLT', 'TRAC01'] for
            heat, salt, and passive tracer along z-dimension.
        """
        grid = self.grid
        BC   = self.BC
        
        # difference to get flux convergence, sign convention is opposite for z
        adv_x_tdc = -grid.diff(ADVx, 'X', boundary=BC['X']).rename('adv_x_tdc')
        adv_y_tdc = -grid.diff(ADVy, 'Y', boundary=BC['Y']).rename('adv_y_tdc')
        adv_r_tdc =  grid.diff(ADVr, 'Z', boundary=BC['Z']).rename('adv_r_tdc')

        # change unit to K/day
        adv_x_tdc *= 86400 / self.volume
        adv_y_tdc *= 86400 / self.volume
        adv_r_tdc *= 86400 / self.volume

        # sum up to get the total tendency due to advection
        advct_tdc = (adv_x_tdc + adv_y_tdc + adv_r_tdc).rename('advct_tdc')

        if self.terms is not None:
            self.terms['adv_x_tdc'] = adv_x_tdc
            self.terms['adv_y_tdc'] = adv_y_tdc
            self.terms['adv_r_tdc'] = adv_r_tdc
            self.terms['advct_tdc'] = advct_tdc
        else:
            self.terms = xr.merge([adv_x_tdc, adv_y_tdc, adv_r_tdc, advct_tdc])

    def cal_true_tendency(self, TOTTend):
        """
        Calculate true tendency output by the model.
        
        Parameters
        ----------
        TOTTend : xarray.DataArray
            Total tendency of ['TH', 'SLT', 'TRAC01'] for
            heat, salt and passive tracer.
        """
        # calculate the true tendency
        total_tdc = TOTTend.rename('total_tdc')

        if self.terms is not None:
            self.terms['total_tdc'] = total_tdc
        else:
            self.terms = xr.merge([total_tdc])



class TracerBudget(Budget):
    """
    This class is designed for the budget analysis in MITgcm.
    """
    def __init__(self, dset, grid=None):
        """
        Construct a TracerBudget instance using a Dataset
        
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
        super().__init__(dset, grid)
        
        self.volume = dset.drF * dset.hFacC * dset.rA


    """
    Calculate all the budget tendency terms.
    """
    def cal_diffusion_tendency(self, DFxE, DFyE, DFrE, DFrI):
        """
        Calculate tendency due to harmonic diffusion.
        
        Parameters
        ----------
        DFxE : xarray.DataArray
            Diffusive flux of some stuff ['TH', 'SLT', 'TRAC01'] for
            heat, salt, and passive tracer along x-dimension.
        DFyE : xarray.DataArray
            Diffusive flux of some stuff ['TH', 'SLT', 'TRAC01'] for
            heat, salt, and passive tracer along y-dimension.
        DFrE : xarray.DataArray
            Diffusive (explicit) flux of some stuff ['TH', 'SLT', 'TRAC01'] for
            heat, salt, and passive tracer along z-dimension.
        DFrI : xarray.DataArray
            Diffusive (implicit) flux of some stuff ['TH', 'SLT', 'TRAC01'] for
            heat, salt, and passive tracer along z-dimension.
        """
        grid = self.grid
        BC   = self.BC
        
        # difference to get flux convergence, sign convention is opposite for z
        dffxE_tdc = -grid.diff(DFxE, 'X', boundary=BC['X']).rename('dffxE_tdc')
        dffyE_tdc = -grid.diff(DFyE, 'Y', boundary=BC['Y']).rename('dffyE_tdc')
        dffrE_tdc =  grid.diff(DFrE, 'Z', boundary=BC['Z']).rename('dffrE_tdc')
        dffrI_tdc =  grid.diff(DFrI, 'Z', boundary=BC['Z']).rename('dffrI_tdc')

        # change unit to K/day
        dffxE_tdc *= 86400 / self.volume
        dffyE_tdc *= 86400 / self.volume
        dffrE_tdc *= 86400 / self.volume
        dffrI_tdc *= 86400 / self.volume

        # sum up to get the total tendency due to harmonic diffusion
        diffu_tdc = (dffxE_tdc + dffyE_tdc +
                     dffrE_tdc + dffrI_tdc).rename('diffu_tdc')

        if self.terms is not None:
            self.terms['dffxE_tdc'] = dffxE_tdc
            self.terms['dffyE_tdc'] = dffyE_tdc
            self.terms['dffrE_tdc'] = dffrE_tdc
            self.terms['dffrI_tdc'] = dffrI_tdc
            self.terms['diffu_tdc'] = diffu_tdc
        else:
            self.terms = xr.merge([dffxE_tdc, dffyE_tdc, dffrE_tdc,
                                   dffrI_tdc, diffu_tdc])

    def cal_nlocalKPP_tendency(self, KPPg):
        """
        Calculate tendency due to non-local KPP mixing parameterization.
        
        Parameters
        ----------
        KPPg : xarray.DataArray
            KPP non-local Flux of ['TH', 'SLT', 'TRAC01'] for
            heat, salt and passive tracer.
        """
        # difference to get flux convergence, sign convention is opposite for z
        nlKPP_tdc = self.grid.diff(KPPg, 'Z', boundary='fill'
                                   ).rename('nlKPP_tdc')

        # change unit to K/day
        nlKPP_tdc *= 86400 / self.volume

        if self.terms is not None:
            self.terms['nlKPP_tdc'] = nlKPP_tdc
        else:
            self.terms = xr.merge([nlKPP_tdc])

    def cal_linFSCorr_tendency(self, WTrMass):
        """
        Calculate tendency due to surface correction.
        
        Parameters
        ----------
        WTrMass : xarray.DataArray
            Vertical Mass-Weight Transp of ['TH', 'SLT', 'TRAC01'] for
            heat, salt and passive tracer.
        """
        coords = self.coords
        
        WTrMass = WTrMass.isel(Zl=0, drop=True)

        # Calculate area-weighted mean
        # This is equivalent to  grid.average(WTrMass, ['YC', 'XC'])
        WTrMean = (WTrMass * coords.rA).sum(['YC', 'XC']) / coords.rA.sum()

        # calculate correction-induced tendency
        corrt_tdc = - (WTrMass - WTrMean) / (coords.drF[0] * coords.hFacC[0])
        corrt_tdc = self.__surface_to_3d(corrt_tdc).rename('corrt_tdc')

        # change unit to K/day
        corrt_tdc *= 86400 / self.volume[0, ...]

        if self.terms is not None:
            self.terms['corrt_tdc'] = corrt_tdc
        else:
            self.terms = xr.merge([corrt_tdc])

    """
    Helper (private) methods are defined below
    """
    def __surface_to_3d(self, da):
        """
        Extending a single-level surface data to a 3D one, filling with zero.

        Parameters
        ----------
        da : DataArray
            A given single-level surface DataArray
        
        Returns
        -------
        result : xarray.DataArray
            Three-dimensional DataArray.
        """
        # add a 'Z' coordinate
        da.coords['Z'] = self.dset.Z[0]
        # add a 'Z' dimension
        return da.expand_dims(dim='Z', axis=1)



class HeatBudget(TracerBudget):
    """
    This class is designed for the heat budget analysis in MITgcm.
    """
    def __init__(self, dset, grid=None):
        """
        Construct a HeatBudget instance using a Dataset and a grid
        
        Parameters
        ----------
        dset : xarray.Dataset
            A given Dataset containing MITgcm output diagnostics
        grid : xgcm.Grid
            A given grid that accounted for grid metrics
        """
        super().__init__(dset, grid)


    """
    Calculate some HeatBudget tendency terms.
    """
    def cal_heating_flux_tendency(self, TFlux, oceQsw, rhoConst=999.8):
        """
        Calculate tendency due to surface heating.
        
        Parameters
        ----------
        TFlux : xarray.DataArray
            Total heat flux (match heat-content variations), >0 increases theta
        oceQsw : xarray.DataArray
            Net Short-Wave radiation (+=down), >0 increases theta.
        rhoConst : float
            For z coordinates, rUnit2mass is equal to rhoConst.
        """
        from GeoApps.ConstUtils import Cp_sw
        
        rA = self.coords.rA

        # difference to get flux convergence, sign convention is opposite for z
        Qtflx = ((TFlux - oceQsw) * rA / (Cp_sw * rhoConst))
        Qsflx = ( oceQsw          * rA / (Cp_sw * rhoConst))

        _, swdown = xr.align(self.dset.Zl, Qsflx * self.__sw_fraction(),
                             join='left')
        swdown    = swdown.fillna(0)

        Qsflx_tdc = -self.grid.diff(swdown, 'Z',
                                    boundary=self.BC['Z']).fillna(0.) \
                    .transpose('time','Z','YC','XC').rename('Qsflx_tdc')

        #Qtflx_tdc = self.__surface_to_3d(Qtflx).rename('Qtflx_Ttdc')
        Qtflx_tdc = Qtflx.rename('Qtflx_tdc')

        # change unit to K/day
        Qsflx_tdc *= 86400 / self.volume
        Qtflx_tdc *= 86400 / self.volume[0, ...]

        if self.terms is not None:
            self.terms['Qsflx_tdc'] = Qsflx_tdc
            self.terms['Qtflx_tdc'] = Qtflx_tdc
        else:
            self.terms = xr.merge([Qsflx_tdc, Qtflx_tdc])


    """
    Helper (private) methods are defined below
    """
    def __sw_fraction(self, fact=1., jwtype=2):
        """
        Clone of MITgcm routine for computing solar short-wave
        flux penetration to specified depth, due to exponential
        decay in Jerlov water type jwtype.

        Reference
            Paulson and Simpson (1977, JPO)
        
        Parameters
        ----------
        fact : number
            Copied from MITgcm source code.

        jwtype : number
            Copied from MITgcm source code.
        
        Returns
        -------
        result : xarray.DataArray
            Vertical profile of short-wave flux penetration.
        """
        rfac = [0.58, 0.62, 0.67, 0.77, 0.78]
        a1   = [0.35, 0.60, 1.00, 1.50, 1.40]
        a2   = [23.0, 20.0, 17.0, 14.0, 7.90]

        facz = fact * self.dset.Zl.sel(Zl=slice(0, -200))
        j    = jwtype - 1

        swdk = (      rfac[j]  * np.exp(facz / a1[j]) +
               (1.0 - rfac[j]) * np.exp(facz / a2[j]))

        return swdk.rename('swdk')



class SaltBudget(TracerBudget):
    """
    This class is designed for the heat budget analysis in MITgcm.
    """
    def __init__(self, dset, grid=None):
        """
        Construct a HeatBudget instance using a Dataset and a grid
        
        Parameters
        ----------
        dset : xarray.Dataset
            A given Dataset containing MITgcm output diagnostics
        grid : xgcm.Grid
            A given grid that accounted for grid metrics
        """
        super().__init__(dset, grid)


    """
    Calculate some HeatBudget tendency terms.
    """
    # TODO
    def cal_heating_flux_tendency(self, SFlux, oceQsw, rhoConst=999.8):
        """
        Calculate tendency due to surface heating.
        
        Parameters
        ----------
        SFlux : xarray.DataArray
            Total salt flux (match heat-content variations), >0 increases theta
        oceQsw : xarray.DataArray
            Net Short-Wave radiation (+=down), >0 increases theta.
        rhoConst : float
            For z coordinates, rUnit2mass is equal to rhoConst.
        """


    """
    Helper (private) methods are defined below
    """



class MomentumBudget(Budget):
    """
    This class is designed for momentum budget analysis in MITgcm.
    """
    def __init__(self, dset, grid=None, var='U'):
        """
        Construct a Budget instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        var : str
            Budget of x-momentum (U) or y-momentum (V).
        
        Return
        ----------
        terms : xarray.Dataset
            A Dataset containing all budget terms
        """
        super().__init__(dset, grid)
        
        if   var == 'U':
            self.volume = dset.drF * dset.hFacW * dset.rAw
        elif var == 'V':
            self.volume = dset.drF * dset.hFacS * dset.rAs
        else:
            raise Exception('unsupported MomentumBudget of ' + var)


    """
    Calculate all the budget tendency terms.
    """
    def cal_advection_tendency(self, Advec):
        """
        Calculate tendency due to 3D advection.  This is not similar to the
        parent class that take 3D divergence of advective fluxes.  It is
        strange that ADVx_Um = ADVy_Um = ADVr_Um = 0 in my channel similation
        (maybe caused by vector-invariant form???).
        
        # TODO
        Verify that this is caused by vector-invariant form.
        
        Just use the Um_Advec including all the advective effects here.
        
        Parameters
        ----------
        Advec : xarray.DataArray
            Advective tendency for either ['Um', 'Vm'].
        """
        # change unit to m/s/day
        advct_tdc = (Advec * 86400).rename('advct_tdc')

        if self.terms is not None:
            self.terms['advct_tdc'] = advct_tdc
        else:
            self.terms = xr.merge([advct_tdc])
    
    def cal_viscous_tendency(self, Diss, VISrI):
        """
        Calculate tendency due to viscosity.  Note that MITgcm contains
            Dissi = dset['Um_Diss'] or dset['Vm_Diss']
        related tendency already without implicit part.  So taking the 3D
        divergence of viscous fluxes is not necessary.
        
        Parameters
        ----------
        Diss : xarray.DataArray
            Viscous dissipation tendency.
        VISrI : xarray.DataArray
            r-component (implicit) of viscous Flux of momentum.
        """
        grid = self.grid
        
        # change unit to m/s/day
        viscE_tdc = (Diss * 86400).rename('viscE_tdc')

        # get a copy
        visco_tdc = viscE_tdc[:].rename('visco_tdc')
        
        # difference to get flux convergence, sign convention is opposite for z
        if VISrI is not None:
            viscI_tdc = grid.diff(VISrI, 'Z', boundary=self.BC['Z']
                                  ).rename('viscI_tdc')
            viscI_tdc *= 86400 / self.volume
            
            visco_tdc += viscI_tdc

        if self.terms is not None:
            self.terms['viscE_tdc'] = viscE_tdc
            self.terms['visco_tdc'] = visco_tdc
        else:
            self.terms = xr.merge([viscE_tdc, viscI_tdc, visco_tdc])
        
        if VISrI is not None:
            self.terms['viscI_tdc'] = viscI_tdc
    
    def calM_viscous_tendency(self, UVEL, VVEL, visH, vis4):
        """
        Calculate tendency due to explicit viscosity manually (using u, v, w).
        Taking the 3D divergence of viscous fluxes.
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        visH : float
            A constant horizontal viscosity.
        vis4 : float
            A constant horizontal biharmonic viscosity.
        """
        # if not self.hydro:
        #     raise Exception('nonhydro unsupported')
        
        # local variables
        diff   = self.grid.diff
        coords = self.coords
        
        from GeoApps.DiagnosticMethods import Dynamics
        
        dm = Dynamics(self.dset, self.grid)
        
        div = dm.cal_horizontal_divergence(UVEL, VVEL)
        vor = dm.cal_vertical_vorticity(UVEL, VVEL)
        lpU = dm.cal_Laplacian(UVEL)
        lpV = dm.cal_Laplacian(VVEL)
        
        Dst = self.__cal_Dstar(lpU, lpV)
        Zst = self.__cal_Zstar(lpU, lpV)
        
        bracket1 =  visH * div - vis4 * Dst
        bracket2 = (visH * vor - vis4 * Zst) * coords.hFacZ
        
        visD2_tdc = diff(bracket1, 'X',
                     boundary=self.BC['X']) / coords.dxC - \
                diff(bracket2, 'Y',
                     boundary=self.BC['Y']) / coords.dyG / coords.hFacW
        
        visD2_tdc *= 86400
        
        visD2_tdc = visD2_tdc.rename('visD2_tdc')
        
        if self.terms is not None:
            self.terms['visD2_tdc'] = visD2_tdc
        else:
            self.terms = xr.merge([visD2_tdc])

    def cal_pressure_gradX_tendency(self, dPHdx, dNPdx, PHI_SURF):
        """
        Calculate tendency due to x-component of pressure gradient.
        
        Parameters
        ----------
        dPHdx : xarray.DataArray
            Tendency from hydrostatic x-component of pressure gradient.
        dNPdx : xarray.DataArray
            Tendency from non-hydrostatic x-component of pressure gradient.
        PHI_SURF : xarray.DataArray
            Surface potential Anomaly.
        """
        # change unit to energy/day
        hydrx_tdc = (dPHdx * 86400).rename('hydrx_tdc')
        
        pGrdx_tdc = hydrx_tdc[:].rename('pGrdx_tdc')
        
        if PHI_SURF is not None:
            phisx_tdc = -((self.grid.diff(PHI_SURF, 'X',boundary=self.BC['X']))
                         / self.coords.dxC * 86400).rename('phisx_tdc')
            pGrdx_tdc += phisx_tdc
        
        if dNPdx is not None:
            nonhx_tdc = (dNPdx * 86400).rename('nonhx_tdc')
            pGrdx_tdc += nonhx_tdc
        
        # store the terms
        if self.terms is not None:
            self.terms['hydrx_tdc'] = hydrx_tdc
            self.terms['pGrdx_tdc'] = pGrdx_tdc
        else:
            self.terms = xr.merge([hydrx_tdc, pGrdx_tdc])

        if PHI_SURF is not None:
            self.terms['phisx_tdc'] = phisx_tdc

        if dNPdx is not None:
            self.terms['nonhx_tdc'] = nonhx_tdc

    def cal_pressure_gradY_tendency(self, dPHdy, dNPdy, PHI_SURF):
        """
        Calculate tendency due to y-component of pressure gradient.
        
        Parameters
        ----------
        dPHdy : xarray.DataArray
            Tendency from hydrostatic y-component of pressure gradient.
        dNPdy : xarray.DataArray
            Tendency from non-hydrostatic y-component of pressure gradient.
        PHI_SURF : xarray.DataArray
            Surface potential anomaly.
        """
        # change unit to energy/day
        hydry_tdc = (dPHdy * 86400).rename('hydry_tdc')
        
        pGrdy_tdc = hydry_tdc[:].rename('pGrdy_tdc')
        
        if PHI_SURF is not None:
            phisy_tdc = -((self.grid.diff(PHI_SURF, 'Y', boundary=self.BC['Y']))
                         / self.coords.dyC * 86400).rename('phisy_tdc')
            pGrdy_tdc += phisy_tdc
        
        if dNPdy is not None:
            nonhy_tdc = (dNPdy * 86400).rename('nonhy_tdc')
            pGrdy_tdc += nonhy_tdc
        
        # store the terms
        if self.terms is not None:
            self.terms['hydry_tdc'] = hydry_tdc
            self.terms['pGrdy_tdc'] = pGrdy_tdc
        else:
            self.terms = xr.merge([hydry_tdc, pGrdy_tdc])

        if PHI_SURF is not None:
            self.terms['phisy_tdc'] = phisy_tdc
        
        if dNPdy is not None:
            self.terms['nonhy_tdc'] = nonhy_tdc

    def cal_Adams_Bashforth_tendency(self, AB_g):
        """
        Calculate the tendency due to Adams-Bashforth integrator.
        
        Parameters
        ----------
        AB_g : xarray.DataArray
            Tendency from Adams-Bashforth.
        """
        # change unit to m/s/day
        AdamB_tdc = (AB_g * 86400).rename('AdamB_tdc')

        if self.terms is not None:
            self.terms['AdamB_tdc'] = AdamB_tdc
        else:
            self.terms = xr.merge([AdamB_tdc])

    def cal_external_forcing_tendency(self, Ext):
        """
        Calculate the tendency due to external forcing.
        
        Parameters
        ----------
        Ext : xarray.DataArray
            Tendency from external forcing.
        """
        # change unit to m/s/day
        Exter_tdc = (Ext * 86400).rename('exter_tdc')

        if self.terms is not None:
            self.terms['exter_tdc'] = Exter_tdc
        else:
            self.terms = xr.merge([Exter_tdc])


    """
    Helper (private) methods are defined below
    """
    def __cal_Dstar(self, lapU, lapV):
        """
        Calculate D_star defined at here:
        https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation
        
        Parameters
        ----------
        lapU : xarray.DataArray
            laplacian of u.
        lapV : xarray.DataArray
            laplacian of v.
        """
        # local variables
        diff   = self.grid.diff
        coords = self.coords
        
        Dstar = (diff(lapU * coords.hFacW * coords.dyG,
                      'X', boundary=self.BC['X']) +
                 diff(lapV * coords.hFacS * coords.dxG,
                      'Y', boundary=self.BC['Y'])) / coords.hFacC / coords.rA
     
        return Dstar
    
    def __cal_Zstar(self, lapU, lapV):
        """
        Calculate zeta_star defined at here:
        https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation
        
        Parameters
        ----------
        lapU : xarray.DataArray
            laplacian of u.
        lapV : xarray.DataArray
            laplacian of v.
        """
        # local variables
        diff   = self.grid.diff
        coords = self.coords
        
        Zstar = (diff(lapV * coords.dyC, 'X', boundary=self.BC['X']) +
                 diff(lapU * coords.dxC, 'Y', boundary=self.BC['Y'])) / coords.rAz
     
        return Zstar



class EnergyBudget(TracerBudget):
    """
    This class is designed for the energy budget analysis in MITgcm.
    """
    def __init__(self, dset, grid=None, hydrostatic=True):
        """
        Construct a EnergyBudget instance using a Dataset and Grid
        
        Parameters
        ----------
        dset : xarray.Dataset
            A given Dataset containing MITgcm output diagnostics
        grid : xgcm.Grid
            A given grid that accounted for grid metrics
        hydrostatic : boolean
            Hydrostatic or nonhydrostatic budget.  Nonhydrostatc includes
            WVEL-related terms
        """
        super().__init__(dset, grid)
        
        self.hydro = hydrostatic


    """
    Calculate all the budget tendency terms.
    """
    def cal_kinetic_energy(self, UVEL, VVEL, WVEL=None):
        """
        Calculate kinetic energy per unit volume (J m^-3).
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        WVEL : xarray.DataArray
            Vertical Velocity.
        """
        coords = self.coords
        interp = self.grid.interp
        
        # interpolate to get tracer-point energy
        KE = (interp(UVEL * coords.hFacW, 'X', boundary=self.BC['X']) **2 +
              interp(VVEL * coords.hFacS, 'Y', boundary=self.BC['Y']) **2)
        KE *= 0.5
        KE = KE.rename('KE')
        
        if not self.hydro:
            KE += interp(WVEL * coords.hFacC,
                         'Z', boundary=self.BC['Z']) **2 * 0.5
        
        if self.terms is not None:
            self.terms['KE'] = KE
        else:
            self.terms = xr.merge([KE])

        return KE

    def cal_true_tendency(self, KE, deltaT):
        """
        Calculate true tendency output by the model.
        
        Parameters
        ----------
        KE : xarray.DataArray
            Kinetic energy.
        """
        # get MITgcm diagnostics
        total_tdc = KE.diff('time')

        # calculate the true tendency (energy/day)
        total_tdc *= 86400 / deltaT
        total_tdc = total_tdc.rename('total_tdc')

        if self.terms is not None:
            self.terms['total_tdc'] = total_tdc
        else:
            self.terms = xr.merge([total_tdc])

    def cal_advection_tendency(self, KE, UVEL=None, VVEL=None, WVEL=None):
        """
        Calculate tendency due to advection.
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        WVEL : xarray.DataArray
            Vertical Velocity.
        """
        # get MITgcm diagnostics
        grid   = self.grid
        coords = self.coords
        BC     = self.BC

        # interpolate to velocity points
        KE_u = grid.interp(KE, 'X', boundary=BC['X'])
        KE_v = grid.interp(KE, 'Y', boundary=BC['Y'])
        KE_w = grid.interp(KE, 'Z', boundary=BC['Z'])
        
        # calculate the fluxes
        ADVx = UVEL * KE_u * coords.dyG * coords.hFacW * coords.drF
        ADVy = VVEL * KE_v * coords.dxG * coords.hFacS * coords.drF
        ADVr = WVEL * KE_w * coords.rA
        
        adv_x_tdc =-grid.diff(ADVx, 'X', boundary=BC['X']).rename('adv_x_tdc')
        adv_y_tdc =-grid.diff(ADVy, 'Y', boundary=BC['Y']).rename('adv_y_tdc')
        adv_r_tdc = grid.diff(ADVr, 'Z', boundary=BC['Z']).rename('adv_r_tdc')
        
        adv_x_tdc = adv_x_tdc * 86400 / self.volume
        adv_y_tdc = adv_y_tdc * 86400 / self.volume
        adv_r_tdc = adv_r_tdc * 86400 / self.volume

        # sum up to get the total tendency due to advection
        advct_tdc = (adv_x_tdc + adv_y_tdc + adv_r_tdc).rename('advct_tdc')

        if self.terms is not None:
            self.terms['adv_x_tdc'] = adv_x_tdc
            self.terms['adv_y_tdc'] = adv_y_tdc
            self.terms['adv_r_tdc'] = adv_r_tdc
            self.terms['advct_tdc'] = advct_tdc
        else:
            self.terms = xr.merge([adv_x_tdc, adv_y_tdc, adv_r_tdc, advct_tdc])

    def cal_surf_tendency(self, UVEL, VVEL, PHI_SURF):
        """
        Calculate pressure gradient tendency due to surface elevation.
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        ETAN : xarray.DataArray
            Surface Height Anomaly.
        g : float
            A constant gravity acceleration used in the model.
        """
        # get MITgcm diagnostics
        grid = self.grid
        BC     = self.BC

        # calculate gradient
        eu = UVEL * -grid.derivative(PHI_SURF, 'X', boundary=BC['X'])
        ev = VVEL * -grid.derivative(PHI_SURF, 'Y', boundary=BC['Y'])

        # interpolate to velocity points
        prsSx_tdc = grid.interp(eu, 'X', boundary=BC['X']).rename('prsEx_tdc')
        prsSy_tdc = grid.interp(ev, 'Y', boundary=BC['Y']).rename('prsEy_tdc')

        # change unit to energy/day
        prsSx_tdc *= 86400
        prsSy_tdc *= 86400

        # sum up to get the total tendency due to harmonic diffusion
        prsSF_tdc = (prsSx_tdc + prsSy_tdc).rename('prsSF_tdc')

        if self.terms is not None:
            self.terms['prsSx_tdc'] = prsSx_tdc
            self.terms['prsSy_tdc'] = prsSy_tdc
            self.terms['prsSF_tdc'] = prsSF_tdc
        else:
            self.terms = xr.merge([prsSx_tdc, prsSy_tdc, prsSF_tdc])

    def cal_pressure_tendency(self, UVEL, VVEL, Um_dPHdx, Vm_dPHdy):
        """
        Calculate pressure gradient tendency.
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        Um_dPHdx : xarray.DataArray
            Zonal pressure gradient tendency.
        Vm_dPHdx : xarray.DataArray
            Meridional pressure gradient tendency.
        """
        if not self.hydro:
            raise Exception('nonhydrostatic unsupported')

        # get MITgcm diagnostics
        interp = self.grid.interp
        
        # calculate gradient
        px = UVEL * Um_dPHdx
        py = VVEL * Vm_dPHdy
        
        # interpolate to velocity points
        prsGx_tdc = interp(px, 'X', boundary=self.BC['X']).rename('prsGx_tdc')
        prsGy_tdc = interp(py, 'Y', boundary=self.BC['Y']).rename('prsGy_tdc')
        
        # change unit to energy/day
        prsGx_tdc *= 86400
        prsGy_tdc *= 86400
        
        # sum up to get the total tendency due to harmonic diffusion
        prsGF_tdc = (prsGx_tdc + prsGy_tdc).rename('prsGF_tdc')
        
        if self.terms is not None:
            self.terms['prsGx_tdc'] = prsGx_tdc
            self.terms['prsGy_tdc'] = prsGy_tdc
            self.terms['prsGF_tdc'] = prsGF_tdc
        else:
            self.terms = xr.merge([prsGx_tdc, prsGy_tdc, prsGF_tdc])

    def cal_Adams_Bashforth_tendency(self, UVEL, VVEL, AB_gU, AB_gV):
        """
        Calculate the tendency due to Adams-Bashforth integrator.
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        AB_gU : xarray.DataArray
            U-tendency from Adams-Bashforth.
        AB_gV : xarray.DataArray
            V-endency from Adams-Bashforth.
        """
        interp = self.grid.interp
        
        AdamB_tdc = (interp(AB_gU * UVEL, 'X', boundary=self.BC['X']) +
                     interp(AB_gV * VVEL, 'Y', boundary=self.BC['Y'])
                    ).rename(f'AdamB_tdc')
        
        # change unit to energy/day
        AdamB_tdc *= 86400

        if self.terms is not None:
            self.terms['AdamB_tdc'] = AdamB_tdc
        else:
            self.terms = xr.merge([AdamB_tdc])
    
    def cal_diffusion_tendency(self, KE, visH, visR):
        """
        Calculate tendency due to harmonic diffusion \nabla^2 KE.
        
        Parameters
        ----------
        KE : xarray.DataArray
            Kinetic energy.
        visH : float
            A constant horizontal viscosity.
        visR : float
            A constant vertical viscosity.
        """
        # get MITgcm diagnostics
        grid   = self.grid
        coords = self.coords
        
        # calculate diffusive fluxes
        dffx = grid.diff(KE, 'X', boundary=self.BC['X']) * visH / coords.dxC \
               * (coords.hFacW * coords.drF * coords.dyG)
        dffy = grid.diff(KE, 'Y', boundary=self.BC['Y']) * visH / coords.dyC \
               * (coords.hFacS * coords.drF * coords.dxG)
        
        # difference to get flux convergence, sign convention is opposite for z
        diffx_tdc = grid.diff(dffx, 'X',
                              boundary=self.BC['X']).rename('diffx_tdc')
        diffy_tdc = grid.diff(dffy, 'Y',
                              boundary=self.BC['Y']).rename('diffy_tdc')
        
        # change unit to energy/day
        diffx_tdc *= 86400 / self.volume
        diffy_tdc *= 86400 / self.volume
        
        # sum up to get the total tendency due to harmonic diffusion
        diffu_tdc = (diffx_tdc + diffy_tdc).rename('diffu_KEtdc')
        
        if not self.hydro:
            dffr = grid.diff(KE, 'Z', boundary='fill') * visR * coords.rA
            diffr_tdc = (grid.diff(dffr, 'Z', boundary='fill')
                        * 86400 / self.volume).rename('diffr_tdc')
            
            diffu_tdc += diffr_tdc
        
        if self.terms is not None:
            self.terms['diffx_tdc'] = diffx_tdc
            self.terms['diffy_tdc'] = diffy_tdc
            self.terms['diffu_tdc'] = diffu_tdc
        else:
            self.terms = xr.merge([diffx_tdc, diffy_tdc, diffu_tdc])
        
        if not self.hydro:
            self.terms['diffr_tdc'] = diffr_tdc

    def cal_dissipation_tendency(self, UVEL, VVEL, WVEL, visH, visR):
        """
        Calculate tendency due to dissipation defined as:
            \nabla u_vec \dot \nabla u_vec
        Reference: dissipation.docx
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        WVEL : xarray.DataArray
            Vertical Velocity.
        visH : float
            A constant horizontal viscosity.
        visR : float
            A constant vertical viscosity.
        """
        # local variables
        interp = self.grid.interp
        diff   = self.grid.diff
        coords = self.coords
        BC     = self.BC
        
        # interpolat to tracer point
        u_u = UVEL
        u_v = interp(interp(UVEL, 'X', boundary=BC['X']), 'Y', boundary=BC['Y'])
        # u_w = interp(interp(UVEL, 'X', boundary=BC['X']), 'Z', boundary='fill')
        
        v_u = interp(interp(VVEL, 'Y', boundary=BC['Y']), 'X', boundary=BC['X'])
        v_v = VVEL
        # v_w = interp(interp(VVEL, 'Y', boundary=BCy), 'Z', boundary='fill')
        
        # calculate delta distances
        dX = interp(coords.dxC, 'X', boundary=BC['X'])
        dY = interp(coords.dyC, 'Y', boundary=BC['Y'])
        dR = coords.drF
        
        # calculate squared gradients
        uxS = (diff(u_u, 'X', boundary=BC['X']) / dX) ** 2
        uyS = (diff(u_v, 'Y', boundary=BC['Y']) / dY) ** 2
        # urS = (diff(u_w, 'Z', boundary='fill'  ) / dR) ** 2
        
        vxS = (diff(v_u, 'X', boundary=BC['X']) / dX) ** 2
        vyS = (diff(v_v, 'Y', boundary=BC['Y']) / dY) ** 2
        # vrS = (diff(v_w, 'Z', boundary='fill'  ) / dR) ** 2
        
        # sum up all the components
        dissi_tdc = - visH * (uxS + uyS + vxS + vyS)# - visR * (urS + vrS)
        
        if not self.hydro:
            w_u = interp(interp(WVEL, 'Z', boundary=BC['Z']),
                         'X', boundary=BC['X'])
            w_v = interp(interp(WVEL, 'Z', boundary=BC['Z']),
                         'Y', boundary=BC['Y'])
            w_w = WVEL
            
            wxS = (diff(w_u, 'X', boundary=BC['X']) / dX) ** 2
            wyS = (diff(w_v, 'Y', boundary=BC['Y']) / dY) ** 2
            wrS = (diff(w_w, 'Z', boundary=BC['Z']) / dR) ** 2
            
            dissi_tdc += - visH * (wxS + wyS) - visR * wrS
        
        # change unit to energy/day
        dissi_tdc *= 86400
        
        if self.terms is not None:
            self.terms['dissi_tdc'] = dissi_tdc
        else:
            self.terms = xr.merge([dissi_tdc])

    def cal_viscousE_tendency(self, UVEL, VVEL, WVEL, visH, vis4):
        """
        Calculate tendency due to explicit viscous dissipation defined as:
        https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        WVEL : xarray.DataArray
            Vertical Velocity.
        visH : float
            A constant horizontal viscosity.
        vis4 : float
            A constant horizontal biharmonic viscosity.
        """
        if not self.hydro:
            raise Exception('nonhydro unsupported')
        
        # local variables
        interp = self.grid.interp
        diff   = self.grid.diff
        coords = self.coords
        
        from GeoApps.DiagnosticMethods import Dynamics
        
        dm = Dynamics(self.dset, self.grid)
        
        div = dm.cal_horizontal_divergence(UVEL, VVEL)
        vor = dm.cal_vertical_vorticity(UVEL, VVEL)
        lpU = dm.cal_Laplacian(UVEL)
        lpV = dm.cal_Laplacian(VVEL)
        
        Dst = self.__cal_Dstar(lpU, lpV)
        Zst = self.__cal_Zstar(lpU, lpV)
        
        bracket1 =  visH * div - vis4 * Dst
        bracket2 = (visH * vor - vis4 * Zst) * coords.hFacZ
        
        visUE = diff(bracket1, 'X',
                     boundary=self.BC['X']) / coords.dxC - \
                diff(bracket2, 'Y',
                     boundary=self.BC['Y']) / coords.dyG / coords.hFacW
        visVE = diff(bracket2, 'X',
                     boundary=self.BC['X']) / coords.dxG / coords.hFacS - \
                diff(bracket1, 'Y',
                     boundary=self.BC['Y']) / coords.dyC
        
        visUE_tdc = interp(UVEL * visUE, 'X',
                           boundary=self.BC['X']).rename('visUE_tdc')
        visVE_tdc = interp(VVEL * visVE, 'Y',
                           boundary=self.BC['Y']).rename('visVE_tdc')
        
        visUE_tdc *= 86400
        visVE_tdc *= 86400
        
        viscE_tdc = (visUE_tdc + visVE_tdc).rename('viscE_tdc')
        
        if self.terms is not None:
            self.terms['visUE_tdc'] = visUE_tdc
            self.terms['visVE_tdc'] = visVE_tdc
            self.terms['viscE_tdc'] = viscE_tdc
        else:
            self.terms = xr.merge([visUE_tdc, visVE_tdc, viscE_tdc])
    
    def cal_viscousI_tendency(self, UVEL, VVEL, VISrI_Um, VISrI_Vm):
        """
        Calculate tendency due to viscous dissipation defined as:
        https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        VISrI_Um : xarray.DataArray
            r-component (implicit) of viscous Flux of U-momentum.
        VISrI_Vm : xarray.DataArray
            r-component (implicit) of viscous Flux of V-momentum.
        """
        # local variables
        interp = self.grid.interp
        diff   = self.grid.diff
        coords = self.coords
        
        visIU = diff(VISrI_Um, 'Z', boundary=self.BC['Z']) / \
                (coords.drF * coords.hFacW * coords.rAw)
    
        visIV = diff(VISrI_Vm, 'Z', boundary=self.BC['Z']) / \
                (coords.drF * coords.hFacS * coords.rAs)
        
        visUI_tdc = interp(UVEL * visIU, 'X',
                           boundary=self.BC['X']).rename('visUI_tdc')
        visVI_tdc = interp(VVEL * visIV, 'Y',
                           boundary=self.BC['Y']).rename('visVI_tdc')
        
        visUI_tdc *= 86400
        visVI_tdc *= 86400
        
        viscI_tdc = (visUI_tdc + visVI_tdc).rename('viscI_tdc')
        
        if self.terms is not None:
            self.terms['visUI_tdc'] = visUI_tdc
            self.terms['visVI_tdc'] = visVI_tdc
            self.terms['viscI_tdc'] = viscI_tdc
        else:
            self.terms = xr.merge([visUI_tdc, visVI_tdc, viscI_tdc])

    def cal_conversion_tendency(self, WVEL, RHOAnom, rhoRef=999.8, g=9.81):
        """
        Calculate tendency due to conversion to potential energy.
        
        Parameters
        ----------
        WVEL : xarray.DataArray
            Vertical velocity.
        RHOAnom : xarray.DataArray
            Density anomaly.
        g : float
            A constant gravity acceleration used in the model.
        """
        if self.hydro:
            raise Exception('no conversion term if hydrostatic=True')
        
        # interpolate to velocity points
        w_c = self.grid.interp(WVEL, 'Z', boundary=self.BC['Z'])
        
        # change unit to energy/day
        convs_tdc = (-RHOAnom * g * w_c * 86400 / rhoRef).rename('convs_tdc')

        if self.terms is not None:
            self.terms['convs_tdc'] = convs_tdc
        else:
            self.terms = xr.merge([convs_tdc])
    
    def cal_external_force_tendency(self, UVEL, VVEL, Um_Ext, Vm_Ext):
        """
        Calculate pressure gradient tendency.
        
        Parameters
        ----------
        UVEL : xarray.DataArray
            Zonal velocity.
        VVEL : xarray.DataArray
            Meridional velocity.
        Um_Ext : xarray.DataArray
            U momentum tendency from external forcing.
        Vm_Ext : xarray.DataArray
            V momentum tendency from external forcing.
        """
        # get MITgcm diagnostics
        grid = self.grid
        
        # calculate the fluxes
        ufx = UVEL * Um_Ext
        vfy = VVEL * Vm_Ext
        
        # interpolate to tracer point
        uExfx_tdc = grid.interp(ufx, 'X',
                                boundary=self.BC['X']).rename('uExfx_tdc')
        vExfy_tdc = grid.interp(vfy, 'Y',
                                boundary=self.BC['Y']).rename('vExfy_tdc')
        
        # change unit to energy/day
        uExfx_tdc *= 86400
        vExfy_tdc *= 86400
        
        # sum up to get the total tendency due to harmonic diffusion
        exfrc_tdc = (uExfx_tdc + vExfy_tdc).rename('exfrc_tdc')
        
        if self.terms is not None:
            self.terms['uExfx_tdc'] = uExfx_tdc
            self.terms['vExfy_tdc'] = vExfy_tdc
            self.terms['exfrc_tdc'] = exfrc_tdc
        else:
            self.terms = xr.merge([uExfx_tdc, vExfy_tdc, exfrc_tdc])


    """
    Helper (private) methods are defined below
    """
    def __cal_Dstar(self, lapU, lapV):
        """
        Calculate D_star defined at here:
        https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation
        
        Parameters
        ----------
        lapU : xarray.DataArray
            laplacian of u.
        lapV : xarray.DataArray
            laplacian of v.
        """
        # local variables
        diff   = self.grid.diff
        coords = self.coords
        
        Dstar = (diff(lapU * coords.hFacW * coords.dyG,
                      'X', boundary=self.BC['X']) +
                 diff(lapV * coords.hFacS * coords.dxG,
                      'Y', boundary=self.BC['Y'])) / coords.hFacC / coords.rA
     
        return Dstar
    
    def __cal_Zstar(self, lapU, lapV):
        """
        Calculate zeta_star defined at here:
        https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation
        
        Parameters
        ----------
        lapU : xarray.DataArray
            laplacian of u.
        lapV : xarray.DataArray
            laplacian of v.
        """
        # local variables
        diff   = self.grid.diff
        coords = self.coords
        
        Zstar = (diff(lapV * coords.dyC, 'X', boundary=self.BC['X']) +
                 diff(lapU * coords.dxC, 'Y', boundary=self.BC['Y'])) / coords.rAz
     
        return Zstar


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in Budget')

