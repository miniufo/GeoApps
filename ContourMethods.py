# -*- coding: utf-8 -*-
"""
Created on 2020.02.05

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from GeoApps.ArrayUtils import interp1d
from GeoApps.ConstUtils import Rearth
from GeoApps.Application import Application
from xhistogram.xarray import histogram


class ContourAnalysis(Application):
    """
    This class is designed for performing the contour analysis.
    """
    def __init__(self, dset, trcr, dims, dimEq,
                 grid=None, increase=True, lt=False, check_mono=False):
        """
        Construct a Dynamics instance using a Dataset

        Parameters
        ----------
        dset : xarray.Dataset
            A given Dataset containing MITgcm output diagnostics
        trcr : str or xarray.DataArray
            A given string indicating the tracer in Dataset
        dims : dict
            Dimensions along which the min/max values are defined and then
            mapped to the contour space.  Example:
                dims = {'X': 'lon', 'Y': 'lat', 'Z': 'Z'}
        dimEq : dict
            Equivalent dimension that should be mapped from contour space.
            Example: dimEq = {'Y': 'lat'}
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        increase : bool
            Contour increase with the index of equivalent dimension or not.
        lt : bool
            If true, less than a contour is defined as inside the contour.
        check_mono: bool
            Check the monotonicity of the result or not (default: False).
        """
        super().__init__(dset, grid=grid)

        if isinstance(trcr, str):
            trcr = dset[trcr]

        if len(dimEq) != 1:
            raise Exception('dimEq should be one dimension e.g., {"Y","lat"}')

        self.tracer = trcr
        self.dimNs  = list(dims.keys())   # dim names,  ['X', 'Y', 'Z']
        self.dimVs  = list(dims.values()) # dim values, ['lon', 'lat', 'Z']
        self.dimEqN = list(dimEq.keys())[0]
        self.dimEqV = list(dimEq.values())[0]
        self.lt     = lt
        self.check_mono = check_mono
        self.increase   = increase


    def cal_area_eqCoord_table(self, mask):
        """
        Calculate the relation table between area and equivalent coordinate.

        Parameters
        ----------
        mask : xarray.DataArray
            A mask of 1 if valid data and Nan if topography.
        out_name : str
            A given name for the returned variable.

        Returns
        ----------
        tbl : xarray.DataArray
            The relation table between area and equivalent coordinate
        """
        ctr = mask[self.dimEqV].copy().rename({self.dimEqV:'contour'}) \
                                      .rename('contour')
        ctrVar, _ = xr.broadcast(mask[self.dimEqV], mask)
        
        eqDimIncre = ctr[-1] > ctr[0]
        
        if self.lt:
            if eqDimIncre == self.increase:
                if self.increase:
                    print('case 1: increase & lt')
                else:
                    print('case 1: decrease & lt')
                mskVar = mask.where(ctrVar < ctr)
            else:
                if self.increase:
                    print('case 2: increase & lt')
                else:
                    print('case 2: decrease & lt')
                mskVar = mask.where(ctrVar > ctr)
        else:
            if eqDimIncre == self.increase:
                if self.increase:
                    print('case 3: increase & gt')
                else:
                    print('case 3: decrease & gt')
                mskVar = mask.where(ctrVar > ctr)
            else:
                if self.increase:
                    print('case 4: increase & gt')
                else:
                    print('case 4: decrease & gt')
                mskVar = mask.where(ctrVar < ctr)
        
        tbl = abs(self.grid.integrate(mskVar, self.dimNs).rename('AeqCTbl')) \
                    .rename({'contour':self.dimEqV}).squeeze().load()
        
        maxArea = abs(self.grid.integrate(mask, self.dimNs)).load().squeeze()
        
        # assign the maxArea to the endpoint
        if tbl[-1] > tbl[0]:
            tbl[-1] = maxArea.values
        else:
            tbl[ 0] = maxArea.values
        
        if self.check_mono:
            check_monotonicity(tbl, 'contour')
        
        return Table(tbl, self.dimEqV)

    def cal_area_eqCoord_table_hist(self, mask):
        """
        Calculate the relation table between area and equivalent coordinate,
        using xhistogram.

        Parameters
        ----------
        mask : xarray.DataArray
            A mask of 1 if valid data and Nan if topography.
        out_name : str
            A given name for the returned variable.

        Returns
        ----------
        tbl : xarray.DataArray
            The relation table between area and equivalent coordinate
        """
        ctr = mask[self.dimEqV].copy().rename({self.dimEqV:'contour'}) \
                                      .rename('contour')
        ctrVar, _ = xr.broadcast(mask[self.dimEqV], mask)
        
        ctrVar = ctrVar.where(mask==1)
        
        increSame = True
        if (ctr.values[-1] > ctr.values[0]) != self.increase:
            increSame = False
        
        tbl = _histogram(ctrVar, ctr, self.dimVs,
                         self.grid.get_metric(ctrVar, self.dimNs), # weights
                         increSame == self.lt # less than or greater than
                         ).rename('AeqCTbl').rename({'contour':self.dimEqV})\
                          .squeeze().load()
        
        if increSame:
            tbl = tbl.assign_coords({self.dimEqV:ctr.values}).squeeze()
        else:
            tbl = tbl.assign_coords({self.dimEqV:ctr.values[::-1]}).squeeze()
        
        if self.check_mono:
            check_monotonicity(tbl, 'contour')
        
        return Table(tbl, self.dimEqV)

    def cal_contours(self, levels=10):
        """
        Calculate contours for a tracer from its min to max values.

        Parameters
        ----------
        trc : xarray.DataArray
            A given tracer in dset.
        levels : int or numpy.array
            The number of contour levels or specified levels.

        Returns
        ----------
        contour : xarray.DataArray
            A array of contour levels.
        """
        if type(levels) is int:
            # specifying number of contours
            mmin = self.tracer.min(dim=self.dimVs)
            mmax = self.tracer.max(dim=self.dimVs)

            # if numpy.__version__ > 1.16, use numpy.linspace instead
            def mylinspace(start, stop, levels):
                divisor = levels - 1
                steps   = (1.0/divisor) * (stop - start)
    
                return steps[..., None] * np.arange(levels) + start[..., None]

            if self.increase:
                start = mmin
                end   = mmax
            else:
                start = mmax
                end   = mmin
                
            ctr = xr.apply_ufunc(mylinspace, start, end, levels,
                                 dask='allowed',
                                 input_core_dims=[[], [], []],
                                 vectorize=True,
                                 output_core_dims=[['contour']])

            ctr.coords['contour'] = np.linspace(0.0, levels-1.0, levels,
                                                dtype='float32')
            
        else:
             # specifying levels of contours
            def mylinspace(tracer, levs):
                return tracer[..., None] - tracer[..., None] + levs

            ctr = xr.apply_ufunc(mylinspace,
                                 self.tracer.min(dim=self.dimVs), levels,
                                 dask='allowed',
                                 input_core_dims=[[], []],
                                 vectorize=True,
                                 output_core_dims=[['contour']])

            ctr.coords['contour'] = levels

        return ctr


    def cal_integral_within_contours(self, contour, tracer=None, integrand=None,
                                     out_name=None):
        """
        Calculate integral of a masked variable within
        pre-calculated tracer contours.

        Parameters
        ----------
        contour: xarray.DataArray
            A given contour levels.
        integrand: xarray.DataArray
            A given variable in dset.  If None, area enclosed by contour
            will be calculated and returned
        out_name: str
            A given name for the returned variable.

        Returns
        ----------
        intVar : xarray.DataArray
            The integral of var inside contour.  If None, area enclosed by
            contour will be calculated and returned
        """
        if type(contour) in [np.ndarray, np.array]:
            # add coordinate as a DataArray
            contour = xr.DataArray(contour, dims='contour',
                                   coords={'contour': contour})
        
        if tracer is None:
            tracer = self.tracer
        
        # this allocates large memory, xhistogram works better
        if integrand is None:
            integrand = tracer - tracer + 1
        
        if out_name is None:
            if integrand is None:
                out_name = 'area'
            else:
                out_name = 'int' + integrand.name

        if self.lt: # this allocates large memory, xhistogram works better
            mskVar = integrand.where(tracer < contour)
        else:
            mskVar = integrand.where(tracer > contour)

        intVar = self.grid.integrate(mskVar, self.dimNs).rename(out_name)
        
        if self.check_mono:
            check_monotonicity(intVar, 'contour')

        return intVar


    def cal_integral_within_contours_hist(self, contour, tracer=None,
                                          integrand=None, out_name=None):
        """
        Calculate integral of a masked variable within
        pre-calculated tracer contours, using histogram method.

        Parameters
        ----------
        contour : xarray.DataArray
            A given contour levels.
        tracedr : xarray.DataArray
            A given tracer.  Default is self.tracer.
        integrand: xarray.DataArray
            A given variable in dset.  If None, area enclosed by contour
            will be calculated and returned
        out_name : str
            A given name for the returned variable.

        Returns
        ----------
        intVar : xarray.DataArray
            The integral of var inside contour.  If None, area enclosed by
            contour will be calculated and returned
        """
        if tracer is None:
            tracer = self.tracer
        
        # weights are the metrics multiplied by integrand
        if integrand is not None: 
            wei = self.grid.get_metric(tracer, self.dimNs) * integrand
        else:
            wei = self.grid.get_metric(tracer, self.dimNs)
        
        # replacing nan with 0 in weights, as weights cannot have nan
        wei = wei.fillna(0.)
        
        CDF = _histogram(tracer, contour, self.dimVs,
                          wei, self.lt).rename(out_name)
        
        # ensure that the contour index is increasing
        if CDF['contour'][-1] < CDF['contour'][0]:
            CDF = CDF.isel({'contour':slice(None, None, -1)})
        
        if self.check_mono:
            check_monotonicity(CDF, 'contour')
        
        return CDF


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
        # centered difference rather than neighboring difference (diff)
        dfVar  =  var.differentiate('contour')
        dfArea = area.differentiate('contour')
        
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


    def cal_local_wave_activity(self, q, Q, table, reso=2):
        """
        Calculate local finite-amplitude wave activity.
        Reference: Huang and Nakamura 2016, JAS

        Parameters
        ----------
        q: xarray.DataArray
            A tracer field.
        Q: xarray.DataArray
            The sorted tracer field.
        table: Table
            The discretized table between equivalent dimension
            and q-contour e.g., Y(q) table.
        reso: int
            Increase resolution relative to the original one along
            the equivalent dimension.
        
        Returns
        ----------
        lwa : xarray.DataArray
            Local finite-amplitude wave activity.
        """
        if type(reso) is not int:
            raise Exception('reso should be int')
        
        wei = self.grid.get_metric(q, self.dimEqN).squeeze()
        
        q = q.squeeze()
        
        eqDim = q[self.dimEqV]
        eqDimLen = len(eqDim)
        tmp = []
        
        # equivalent dimension is increasing or not
        coord_incre = True
        if eqDim.values[-1] < eqDim.values[0]:
            coord_incre = False
        
        for j in range(eqDimLen):
            # deviation from the reference
            qe = q - Q.isel({self.dimEqV:j})
            
            # above or below the reference coordinate surface
            m = eqDim>eqDim.values[j] if coord_incre else eqDim<eqDim.values[j]
            
            if self.increase:
                mask1 = xr.where(qe>0, -1, 0)
                mask2 = xr.where(m, 0, mask1).transpose(*(mask1.dims))
                mask3 = xr.where(np.logical_and(qe<0, m), 1, mask2)
            else:
                mask1 = xr.where(qe<0, -1, 0)
                mask2 = xr.where(m, 0, mask1).transpose(*(mask1.dims))
                mask3 = xr.where(np.logical_and(qe>0, m), 1, mask2)
            
            lwa = (qe * mask3 * wei).sum(self.dimEqV)
            #lwa = self.grid.integrate(qe * mask3 * wei, self.dimEqV)
            
            tmp.append(lwa)
            
            # if j==180:
            #     mask3.plot()
        
        LWA = xr.concat(tmp, self.dimEqV).transpose(*(q.dims))
        LWA[self.dimEqV] = eqDim.values
        
        return LWA


    def cal_normalized_Keff(self, Leq2, Lmin, mask=1e5):
        """
        Calculate normalized effective diffusivity.

        Parameters
        ----------
        Leq2 : xarray.DataArray
            Squared equivalent length.
        Lmin : xarray.DataArray
            Minimum possible length.
        mask : float
            A threshold larger than which is set to nan.

        Returns
        ----------
        nkeff : xarray.DataArray
            The normalized effective diffusivity (Nusselt number).
        """
        nkeff = Leq2 / Lmin / Lmin
        nkeff = nkeff.where(nkeff<mask).rename('nkeff')

        return nkeff


    def interp_to_dataset(self, predef, dimEq, vs):
        """
        Interpolate given variables to prescribed equivalent latitudes
        and collect them into an xarray.Dataset.

        Parameters
        ----------
        predef : numpy.array or xarray.DataArray
            Pre-defined coordinate values are interpolated
        dimEq : xarray.DataArray
            Equivalent dimension defined in contour space.
        vs : list of xrray.DataArray or a xarray.Dataset
            A list of variables to be interplated
        
        Returns
        ----------
        interp : xarray.Dataset
            The interpolated variables merged in a Dataset.
        """
        re = []
        
        if type(vs) is xr.Dataset:
            for var in vs:
                re.append(self.interp_to_coords(predef, dimEq,
                                                vs[var]).rename(var))
        else:
            for var in vs:
                re.append(self.interp_to_coords(predef, dimEq,
                                                var).rename(var.name))
        
        return xr.merge(re)


    def cal_contours_at(self, predef, table):
        """
        Calculate contours for a tracer at prescribed Ys,
        so that the returned contour and its enclosed area will give a
        monotonic increasing/decreasing results.

        This function will first rough estimate the contour-enclosed
        area and equivalent Ys, and then interpolate the Y(q) relation
        table to get the q(Y) and return q.

        Parameters
        ----------
        predef : xarray.DataArray or numpy.ndarray or numpy.array
            An 1D array of prescribed coordinate values.
        table : Table
            A(dimEq) table.

        Returns
        ----------
        contour : xarray.DataArray
            A array of contour levels corresponding to preY.
        """
        if len(predef.shape) != 1:
            raise Exception('predef should be a 1D array')

        if type(predef) in [np.ndarray, np.array]:
            # add coordinate as a DataArray
            predef = xr.DataArray(predef, dims='new', coords={'new': predef})

        N = predef.size
        
        ctr   = self.cal_contours(N)
        area  = self.cal_integral_within_contours(ctr)
        dimEq = table.lookup_coordinates(area).rename('Z')
        # print(self.interp_to_coords(predef, dimEq, ctr))
        qIntp = self.interp_to_coords(predef.squeeze(), dimEq, ctr.squeeze()) \
                    .rename({'new': 'contour'}) \
                    .rename(ctr.name)

        qIntp['contour'] = np.linspace(0, N-1, N, dtype='float32')

        return qIntp


    def cal_contours_at_hist(self, predef, table):
        """
        Calculate contours for a tracer at prescribed Ys,
        so that the returned contour and its enclosed area will give a
        monotonic increasing/decreasing results.

        This function will first rough estimate the contour-enclosed
        area and equivalent Ys, and then interpolate the Y(q) relation
        table to get the q(Y) and return q.

        Parameters
        ----------
        predef : xarray.DataArray or numpy.ndarray or numpy.array
            An 1D array of prescribed coordinate values.
        table : Table
            A(dimEq) table.

        Returns
        ----------
        contour : xarray.DataArray
            A array of contour levels corresponding to preY.
        """
        if len(predef.shape) != 1:
            raise Exception('predef should be a 1D array')

        if type(predef) in [np.ndarray, np.array]:
            # add coordinate as a DataArray
            predef = xr.DataArray(predef, dims='new', coords={'new': predef})

        N = predef.size
        
        ctr   = self.cal_contours(N)
        area  = self.cal_integral_within_contours_hist(ctr)
        dimEq = table.lookup_coordinates(area).rename('Z')
        qIntp = self.interp_to_coords(predef.squeeze(), dimEq, ctr.squeeze()) \
                    .rename({'new': 'contour'}) \
                    .rename(ctr.name)

        qIntp['contour'] = np.linspace(0, N-1, N, dtype='float32')

        return qIntp


    def cal_equivalent_coords(self, mapFunc, area, out_name='Yeq'):
        return mapFunc(area).rename(out_name)

    def cal_minimum_possible_length(self, mapFunc, Yeq, out_name='Lmin'):
        return mapFunc(Yeq).rename(out_name)

    def interp_to_coords(self, predef, eqCoords, var, interpDim='contour'):
        """
        Interpolate a give variable from equally-spaced contour dimension
        to predefined coordinate values along equivalent dimension.

        Parameters
        ----------
        predef : numpy.array or xarray.DataArray
            Pre-defined Ys where values are interpolated
        eqCoords : xarray.DataArray
            Equivalent coordinates.
        var : xarray.DataArray
            A given variable to be interplated
        interpDim : str
            Dimension along which it is interpolated
        
        Returns
        ----------
        interp : xarray.Dataset
            The interpolated variable.
        """
        dimTmp = 'new'

        if isinstance(predef, (np.ndarray, list)):
            # add coordinate as a DataArray
            predef  = xr.DataArray(predef, dims=dimTmp, coords={dimTmp: predef})
        else:
            dimTmp = predef.dims[0]

        # get a single vector like Yeq[0, 0, ..., :]
        vals = eqCoords
        
        while len(vals.shape) > 1:
            vals = vals[0]
        
        if vals[0] < vals[-1]:
            increasing = True
        else:
            increasing = False
        
        # no dask support for np.linspace
        varIntp = xr.apply_ufunc(interp1d, predef, eqCoords, var.load(),
                  kwargs={'inc': increasing},
                  # dask='allowed',
                  input_core_dims =[[dimTmp],[interpDim],[interpDim]],
                  output_core_dims=[[dimTmp]],
                  exclude_dims=set((interpDim,)),
                  vectorize=True
                  ).rename(var.name)

        return varIntp


class Table(object):
    """
    This class is designed as a one-to-one mapping table between two
    mononitical increasing/decreasing quantities.
    
    The table is represented as y = F(x), with y as the values and
    x the coordinates.
    """
    def __init__(self, table, dimEq):
        """
        Construct a table.

        Parameters
        ----------
        table : xarray.Dataset
            A table quantity as a function of specific coordinate.
        dimEq : numpy.array or xarray.Dataset
            A set of equivalent coordinates along a dimension
        """
        if len(table.shape) != 1:
            raise Exception('require only 1D array')
        
        self._table = table
        self._coord = table[dimEq]
        self._dimEq = dimEq
        self._incVl = table[-1] > table[0]
        self._incCd = table[dimEq][-1] > table[dimEq][0]

    def lookup_coordinates(self, values):
        """
        For y = F(x), get coordinates (x) given values (y).

        Parameters
        ----------
        values: numpy.ndarray or xarray.DataArray
            Values as y.

        Returns
        -------
        coords : xarray.DataArray
            Coordinates as x.
        """
        # dimEq = self._dimEq
        # iDims = [[],[dimEq],[dimEq]]
        # oDims = [[]]
        
        # if 'contour' in values.dims:
        #     iDims = [['contour'],[dimEq],[dimEq]]
        #     oDims = [['contour']]
        
        # if len(values.shape) == 1:
        #     return interp1d(values, self._table, self._coord, self._incVl)
        # else:
        #     varIntp = xr.apply_ufunc(interp1d,
        #               values, self._table, self._coord,
        #               kwargs={'inc': self._incVl},
        #               dask='allowed',
        #               input_core_dims = iDims,
        #               output_core_dims= oDims,
        #               # exclude_dims=set(('contour',)),
        #               vectorize=True
        #               ).rename(dimEq + 'eq')
        
        #     return varIntp
        if len(values.dims) > 1:
            values = values.squeeze()
        
        re = interp1d(values, self._table, self._coord, self._incVl)
        
        if isinstance(re, np.ndarray):
            re = xr.DataArray(re, dims=values.dims, coords=values.coords)
        
        return re

    def lookup_values(self, coords):
        """
        For y = F(x), get values (y) given coordinates (x).

        Parameters
        ----------
        coords : list or numpy.array or xarray.DataArray
            Coordinates as x.

        Returns
        -------
        values : xarray.DataArray
            Values as y.
        """
        re = interp1d(coords, self._coord, self._vables, self._incCd)
        
        if isinstance(re, np.ndarray):
            re = xr.DataArray(re, dims=coords.dims, coords=coords.coords)
        
        return re
        


def equivalent_Latitudes(area):
    """
    Calculate equivalent latitude using the formular:
        2 * pi * a^2 * [sin(latEq) + sin(90)] = area.
    This is similar to a EqY(A) table.

    Parameters
    ----------
    area : xarray.DataArray
        Contour-enclosed area.
    
    Returns
    ----------
    latEq : xarray.DataArray
        The equivalent latitudes.
    """
    ratio = area/2.0/np.pi/Rearth/Rearth - 1.0

    # clip ratio within [-1, 1]
    ratio = xr.where(ratio<-1, -1.0, ratio)
    ratio = xr.where(ratio> 1,  1.0, ratio)

    latEq = np.degrees(np.arcsin(ratio))

    return latEq


def mininum_Length_at(lats):
    """
    Calculate minimum length on a sphere given latitudes.

    Parameters
    ----------
    latEq : xarray.DataArray
        Equivalent latitude.
    
    Returns
    ----------
    Lmin : xarray.DataArray
        The minimum possible length of the contour.
    """
    Lmin = (2.0 * np.pi * Rearth * np.cos(np.deg2rad(lats)))

    return Lmin


def check_monotonicity(var, dim):
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
        None.  Raise exception if not monotonic
    """
    dfvar = var.diff(dim)
    
    if not dfvar.all():
        pos = (dfvar == 0).argmax(dim=var.dims)
        
        for tmp in pos:
            print(tmp)
            print(pos[tmp].values)
            
            if tmp != dim:
                v = var.isel({tmp:pos[tmp].values}).load()
        
        raise Exception('not monotonic var at\n' + str(v))



"""
Below are the private helper methods
"""
def _histogram(var, bins, dim, weights, lt):
    """
    A wrapper for xhistogram, which allows decreasing bins and return
    a result that contains the same size as that of bins.
    
    Note that it is assumed the start and end bins correspond to the tracer
    extrema.
    
    Parameters
    ----------
    var: xarray.DataArray
        A variable that need to be histogrammed.
    bins: list or numpy.array or xarray.DataArray
        An array of bins.
    dim: str or list of str
        Dimensions along which histogram is performed.
    weights: xarray.DataArray
        Weights of each data in var.
    increase: bool
        Increasing bins with index or not.
    lt: bool
        Less than a given value or not.
    
    Returns
    ----------
    hist : xarray.DataArray
        Result of the histogram.
    """
    if type(bins) in [np.ndarray, np.array]:
        bvalues = bins
        
        if not np.diff(bvalues).all():
            raise Exception('non monotonic bins')
            
    elif type(bins) in [xr.DataArray]:
        bvalues = bins.squeeze() # squeeze the dimensions
        
        if not bvalues.diff('contour').all():
            raise Exception('non monotonic bins')
        
        if not 'time' in bvalues.dims:
            bvalues = bvalues.values
        
    elif type(bins) in [list]:
        bvalues = np.array(bins)
        
        if not np.diff(bvalues).all():
            raise Exception('non monotonic bins')
    else:
        raise Exception('bins should be numpy.array or xarray.DataArray')
            
    # unified index of the contour coordinate
    if type(bvalues) in [xr.DataArray]:
        binNum = np.array(range(len(bvalues['contour']))).astype(np.float32)
    else:
        binNum = np.array(range(len(bvalues))).astype(np.float32)
    
    if type(bvalues) in [xr.DataArray]:
        re = []
        
        for l in range(len(bvalues.time)):
            rng = {'time': l}
            
            trc = var.isel(rng)
            ctr = bvalues.isel(rng).values
            
            if 'time' in weights.dims:
                wt = weights.isel(rng)
            else:
                wt = weights
            
            bincrease = True if ctr[0] < ctr[-1] else False
            
            # add a bin so that the result has the same length of contour
            if bincrease:
                step = (ctr[-1] - ctr[0]) / (len(ctr) - 1)
                bins = np.concatenate([[ctr[0]-step], ctr])
            else:
                step = (ctr[0] - ctr[-1]) / (len(ctr) - 1)
                bins = np.concatenate([[ctr[-1]-step], ctr[::-1]])
                # bins[1:] -= step / 1e3
            
            tmp = histogram(trc, bins=[bins], dim=dim, weights=wt) \
                 .assign_coords({trc.name+'_bin':binNum})
            
            re.append(tmp)
        
        pdf = xr.concat(re, 'time').rename({var.name+'_bin':'contour'})
        
        if bincrease:
            pdf = pdf.assign_coords(contour=binNum)
        else:
            pdf = pdf.assign_coords(contour=binNum[::-1])
    else:
        bincrease = True if bvalues[0] < bvalues[-1] else False
        
        # add a bin so that the result has the same length of contour
        if bincrease:
            step = (bvalues[-1] - bvalues[0]) / (len(bvalues) - 1)
            bins = np.insert(bvalues, 0, bvalues[0]-step)
        else:
            step = (bvalues[0] - bvalues[-1]) / (len(bvalues) - 1)
            bins = np.insert(bvalues[::-1], 0, bvalues[-1]-step)
            # bins[1:] -= step / 1e3
        
        pdf = histogram(var, bins=[bins], dim=dim, weights=weights) \
              .rename({var.name+'_bin':'contour'})
        
        if bincrease:
            pdf = pdf.assign_coords(contour=binNum)
        else:
            pdf = pdf.assign_coords(contour=binNum[::-1])
    
    # assign time coord. to pdf
    if 'time' in var.dims:
        pdf = pdf.assign_coords(time=var['time'].values)
    
    # get CDF from PDF
    cdf = pdf.cumsum('contour')
    
    if not lt: # for the case of greater than
        cdf = cdf.isel({'contour':-1}) - cdf
    
    return cdf


def _get_extrema_extend(data, N):
    """
    Get the extrema by extending the endpoints
    
    Parameters
    ----------
    data: xarray.DataArray
        A variable that need to be histogrammed.
    N: int
        A given length to get step
    
    Returns
    ----------
    vmin, vmax : float, float
        Extended extrema.
    """
    vmin = data.min().values
    vmax = data.max().values
    
    step = (vmax - vmin) / N
    
    return vmin - step, vmax + step


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in ContourMethods')

