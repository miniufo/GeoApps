# -*- coding: utf-8 -*-
"""
Created on 2020.02.05

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from utils.ArrayUtils import interp1d
from utils.ConstUtils import Re
from abc import ABCMeta, abstractmethod
from utils.Application import Application


class ContourAnalysis(Application):
    """
    This class is designed for performing the contour analysis.
    """
    __metaclass__ = ABCMeta
    
    
    def __init__(self, dset, trcr, grid=None):
        """
        Construct a Dynamics instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        trcr : str
            a given string indicating the tracer in Dataset
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        """
        super(ContourAnalysis, self).__init__(dset, grid=grid)
        
        self.tracer = dset[trcr]


    def cal_contours(self, levels=10, dims=None, increasing=True):
        """
        Calculate contours for a tracer from its min to max values.

        Parameters
        ----------
        trc : xarray.DataArray
            A given tracer in dset.
        levels : int or numpy.array
            The number of contour levels or specified levels.
        dims : dict
            Dimensions along which the min/max values are defined.
        increasing : boolean
            Whether the contours are defined from min to max values.

        Returns
        ----------
        contour : xarray.DataArray
            A array of contour levels.
        """
        if type(levels) is int:
            # specifying number of contours
            mmin = self.tracer.min(dim=dims)
            mmax = self.tracer.max(dim=dims)

            # if numpy.__version__ > 1.16, use numpy.linspace instead
            def mylinspace(start, stop, levels):
                divisor = levels - 1
                steps   = (1.0/divisor) * (stop - start)
    
                return steps[..., None] * np.arange(levels) + start[..., None]

            if increasing:
                start = mmin
                end   = mmax
            else:
                start = mmax
                end   = mmin
                
            ctr = xr.apply_ufunc(mylinspace, start, end, levels,
                                 dask='allowed',
                                 input_core_dims=[[], [], []],
                                 output_core_dims=[['contour']])

            ctr.coords['contour'] = np.linspace(0.0, levels-1.0, levels)
            
        else:
             # specifying levels of contours
            def mylinspace(tracer, levs):
                return tracer[..., None] + levs - tracer[..., None]

            ctr = xr.apply_ufunc(mylinspace, self.tracer.min(dim=dims), levels,
                                 dask='allowed',
                                 input_core_dims=[[], []],
                                 output_core_dims=[['contour']])

            ctr.coords['contour'] = levels

        return ctr


    def cal_integral_within_contours(self, contour, var=None, name=None, lt=True):
        """
        Calculate masked variable using pre-calculated tracer contours.

        Parameters
        ----------
        contour : xarray.DataArray
            A given contour levels.
        var  : xarray.DataArray
            A given variable in dset.  If None, area enclosed by contour
            will be calculated and returned
        name : str
            A given name for the returned variable.
        lt : boolean
            less than the given contour or greater than.

        Returns
        ----------
        intVar : xarray.DataArray
            The integral of var inside contour.  If None, area enclosed by
            contour will be calculated and returned
        """
        if var is None:
            var = self.tracer - self.tracer + 1
        
        if name is None:
            if var is None:
                name = 'area'
            else:
                name = 'int' + var.name
        if lt:
            mskVar = var.where(self.tracer < contour)
        else:
            mskVar = var.where(self.tracer > contour)

        intVar = self.grid.integrate(mskVar, ['X','Y']).rename(name)

        return intVar


    def cal_gradient_wrt_area(self, var, area):
        """
        Calculate gradient with respect to area.

        Parameters
        ----------
        var  : xarray.DataArray
            A variable that need to be differentiated.
        area : xarray.DataArray
            Area enclosed by contour levels.
        
        Returns
        ----------
        dVardA : xarray.DataArray
            The derivative of var w.r.t contour-encloesd area.
        """
        dfVar  =  var.diff('contour')
        dfArea = area.diff('contour')
        
        dVardA = (dfVar / dfArea).rename('d'+var.name+'dA')

        return dVardA


    def cal_sqared_equivalent_length(self, dgrdSdA, dqdA):
        """
        Calculate normalized effective diffusivity.

        Parameters
        ----------
        dgrdSdA : xarray.DataArray
            d [Integrated grd(q)^2] / dA.
        dqdA : xarray.DataArray
            d [q] / dA.
        
        Returns
        ----------
        Leq2 : xarray.DataArray
            The squared equivalent latitude.
        """
        Leq2  = (dgrdSdA / dqdA ** 2).rename('Leq2')

        return Leq2


    def cal_normalized_Keff(self, Leq2, Lmin):
        """
        Calculate normalized effective diffusivity.

        Parameters
        ----------
        Leq2 : xarray.DataArray
            Squared equivalent length.
        Lmin : xarray.DataArray
            Minimum possible length.

        Returns
        ----------
        nkeff : xarray.DataArray
            The normalized effective diffusivity (Nusselt number).
        """
        nkeff = Leq2 / Lmin / Lmin
        nkeff = nkeff.where(nkeff<1e5).rename('nkeff')

        return nkeff


    def check_monotonicity(self, var, dim):
        """
        Check monotonicity of a variable along a dimension.

        Parameters
        ----------
        var : xarray.DataArray
            A variable that need to be checked.
        dim : str
            A string indicate the dimension.

        Returns
        ----------
        True or False.
        """
        diffVar = var.diff(dim)

        if not diffVar.all():
            print('not monotonic var')
            return False
        
        return True


    def interp_to_dataset(self, lats, latEq, vs):
        """
        Interpolate given variables to prescribed equivalent latitudes
        and collect them into an xarray.Dataset.

        Parameters
        ----------
        lats  : numpy.array or xarray.DataArray
            Pre-defined latitudes where values are interpolated
        latEq : xarray.DataArray
            Equivalent latitudes.
        vs    : list of xrray.DataArray
            A list of variables to be interplated
        
        Returns
        ----------
        interp : xarray.Dataset
            The interpolated variables merged in a Dataset.
        """
        if type(self) is ContourAnalysisInCartesian:
            name = 'Y'
        elif type(self) is ContourAnalysisInLatLon:
            name = 'lat'
        else:
            raise Exception('unsupported type of ' + str(type(self)))
        
        re = []
        
        if type(vs) is xr.Dataset:
            for var in vs:
                re.append(self._interp_to_coords(lats, latEq, vs[var],
                                                 name=name).rename(var))
        else:
            for var in vs:
                re.append(self._interp_to_coords(lats, latEq, var,
                                                 name=name).rename(var.name))
        
        return xr.merge(re)


    @abstractmethod
    def cal_contours_at(self, preLat, dims=None): pass

    @abstractmethod
    def cal_equivalent_coords(self, area): pass

    @abstractmethod
    def cal_minimum_possible_length(self, Yeq): pass

    @abstractmethod
    def interp_to_coords(self, Ys, Yeq, var): pass

    """
    Helper (private) methods are defined below
    """
    def _interp_to_coords(self, Ys, Yeq, var, name='lat'):
        """
        Calculate equivalent coordinates.  Now only lat/lon or Cartesian
        coordinates are implemented for name = ['lat', 'Y'].

        Parameters
        ----------
        Ys  : numpy.array or xarray.DataArray
            Pre-defined Ys where values are interpolated
        Yeq : xarray.DataArray
            Equivalent Ys.
        var : xarray.DataArray
            A given variable to be interplated
        name : str
            The name for the interpolated coordinate.  Now it is only
            implemented in ['lat', 'Y'] for [lat/lon, Cartesian] coordinates
        
        Returns
        ----------
        interp : xarray.Dataset
            The interpolated variable.
        """
        dimName = 'new'

        if type(Ys) in [np.ndarray, np.array]:
            # add coordinate as a DataArray
            Ys  = xr.DataArray(Ys, dims=dimName, coords={dimName: Ys})
        else:
            dimName = Ys.dims[0]

        # get a single vector like Yeq[0, 0, ..., :]
        vals = Yeq
        
        while len(vals.shape) > 1:
            vals = vals[0]
        
        if vals[0] < vals[-1]:
            increasing = True
        else:
            increasing = False

        varIntp = xr.apply_ufunc(interp1d, Ys, Yeq, var,
                  kwargs={'inc': increasing},
                  dask='allowed',
                  input_core_dims=[[dimName],['contour'],['contour']],
                  output_core_dims=[[dimName]],
                  exclude_dims=set(('contour',)),
                  vectorize=True).rename({dimName: name}).rename(var.name)

        return varIntp



class ContourAnalysisInLatLon(ContourAnalysis):
    """
    This class is designed for performing the contour analysis
    in lat/lon coordinates.
    """
    __metaclass__ = ABCMeta
    
    
    def __init__(self, dset, trcr, grid=None):
        """
        Construct a Dynamics instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        trcr : str
            a given string indicating the tracer in Dataset
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        """
        super(self.__class__, self).__init__(dset, trcr, grid=grid)


    def cal_contours_at(self, preLat, dims=None):
        """
        Calculate contours for a tracer at prescribed latitudes,
        so that the returned contour and its enclosed area will give a
        monotonic increasing/decreasing results.

        This function will first rough estimate the contour-enclosed
        area and equivalent latitudes, and then interpolate the latEq(q)
        relation table to get the q(latEq) and return q.

        Parameters
        ----------
        preLat : xarray.DataArray or numpy.ndarray or numpy.array
            An 1D array of prescribed latitudes.
        dims : dict
            Dimensions along which the min/max values are defined.

        Returns
        ----------
        contour : xarray.DataArray
            A array of contour levels corresponding to preLat.
        """
        if len(preLat.shape) != 1:
            raise Exception('preLat should be a 1D array')

        if type(preLat) in [np.ndarray, np.array]:
            # add coordinate as a DataArray
            preLat = xr.DataArray(preLat, dims='new', coords={'new': preLat})

        N = preLat.size

        ctr   = self.cal_contours(N, dims=dims)
        area  = self.cal_integral_within_contours(ctr)
        latEq = self.cal_equivalent_latitude(area)
        qIntp = self.interp_to_coords(preLat, latEq, ctr) \
                    .rename({'lat': 'contour'}) \
                    .rename(ctr.name)

        qIntp['contour'] = np.linspace(0, N-1, N)

        return qIntp


    def cal_equivalent_coords(self, area):
        """
        Calculate equivalent latitude.

        Parameters
        ----------
        area : xarray.DataArray
            Contour-enclosed area.
        
        Returns
        ----------
        latEq : xarray.DataArray
            The equivalent latitudes.
        """
        ratio = area/2.0/np.pi/Re/Re - 1.0

        # clip ratio within [-1, 1]
        ratio = xr.where(ratio<-1, -1.0, ratio)
        ratio = xr.where(ratio> 1,  1.0, ratio)

        latEq = np.degrees(np.arcsin(ratio)).rename('latEq')

        return latEq


    def cal_minimum_possible_length(self, latEq):
        """
        Calculate minimum possible length.

        Parameters
        ----------
        latEq : xarray.DataArray
            Equivalent latitude.
        
        Returns
        ----------
        Lmin : xarray.DataArray
            The minimum possible length of the contour.
        """
        Lmin = (2.0 * np.pi * Re * np.cos(np.deg2rad(latEq))).rename('Lmin')

        return Lmin


    def interp_to_coords(self, lats, latEq, var):
        """
        Calculate equivalent latitude.

        Parameters
        ----------
        lats  : numpy.array or xarray.DataArray
            Pre-defined latitudes where values are interpolated
        latEq : xarray.DataArray
            Equivalent latitudes.
        var   : xarray.DataArray
            A given variable to be interplated
        
        Returns
        ----------
        interp : xarray.Dataset
            The interpolated variable.
        """
        return self._interp_to_coords(lats, latEq, var, name='lat')



class ContourAnalysisInCartesian(ContourAnalysis):
    """
    This class is designed for performing the contour analysis
    in Cartesian coordinates.
    """
    __metaclass__ = ABCMeta
    
    
    def __init__(self, dset, trcr, grid=None):
        """
        Construct a Dynamics instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        trcr : str
            a given string indicating the tracer in Dataset
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        """
        super(self.__class__, self).__init__(dset, trcr, grid=grid)


    def cal_contours_at(self, preY, dims=None):
        """
        Calculate contours for a tracer at prescribed Ys,
        so that the returned contour and its enclosed area will give a
        monotonic increasing/decreasing results.

        This function will first rough estimate the contour-enclosed
        area and equivalent Ys, and then interpolate the Y(q) relation
        table to get the q(Y) and return q.

        Parameters
        ----------
        preY : xarray.DataArray or numpy.ndarray or numpy.array
            An 1D array of prescribed Ys.
        dims : dict
            Dimensions along which the min/max values are defined.

        Returns
        ----------
        contour : xarray.DataArray
            A array of contour levels corresponding to preY.
        """
        if len(preY.shape) != 1:
            raise Exception('preLat should be a 1D array')

        if type(preY) in [np.ndarray, np.array]:
            # add coordinate as a DataArray
            preY = xr.DataArray(preY, dims='new', coords={'new': preY})

        N = preY.size

        ctr   = self.cal_contours(N, dims=dims)
        area  = self.cal_integral_within_contours(ctr)
        Yeq   = self.cal_equivalent_coords(area)
        qIntp = self.interp_to_coords(preY, Yeq, ctr) \
                    .rename({'Y': 'contour'}) \
                    .rename(ctr.name)

        qIntp['contour'] = np.linspace(0, N-1, N)

        return qIntp


    def cal_equivalent_coords(self, area):
        """
        Calculate equivalent Ys.

        Parameters
        ----------
        area : xarray.DataArray
            Contour-enclosed area.
        
        Returns
        ----------
        Yeq : xarray.DataArray
            The equivalent Ys.
        """
        Yeq = (area / np.sum(self.coords['dxG'][0].values)).rename('Yeq')

        # clip ratio within [-1, 1]
        # ratio = xr.where(ratio<-1, -1.0, ratio)
        # ratio = xr.where(ratio> 1,  1.0, ratio)

        return Yeq


    def cal_minimum_possible_length(self, Yeq):
        """
        Calculate minimum possible length.

        Parameters
        ----------
        Yeq : xarray.DataArray
            Equivalent Ys.
        
        Returns
        ----------
        Lmin : xarray.DataArray
            The minimum possible length of the contour.
        """
        width = np.sum(self.coords['dxG'][0].values)
        
        Lmin = (Yeq -Yeq + width).rename('Lmin')

        return Lmin


    def interp_to_coords(self, Ys, Yeq, var):
        """
        Calculate equivalent Ys.

        Parameters
        ----------
        Ys  : numpy.array or xarray.DataArray
            Pre-defined Ys where values are interpolated
        Yeq : xarray.DataArray
            Equivalent Ys.
        var : xarray.DataArray
            A given variable to be interplated
        
        Returns
        ----------
        interp : xarray.Dataset
            The interpolated variable.
        """
        return self._interp_to_coords(Ys, Yeq, var, name='Y')


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in ContourMethods')

