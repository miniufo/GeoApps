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
                 grid=None, increase=True, lt=False):
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
        """
        super().__init__(dset, grid=grid)

        if isinstance(trcr, str):
            trcr = dset[trcr]

        if len(dimEq) != 1:
            raise Exception('dimEq should be one dimension e.g., {"Y","lat"}')
        self.dimNs  = list(dims.keys())   # dim names,  ['X', 'Y', 'Z']
        self.dimVs  = list(dims.values()) # dim values, ['lon', 'lat', 'Z']
        self.dimEqN = list(dimEq.keys())[0]
        self.dimEqV = list(dimEq.values())[0]
        self.increase = increase
        self.lt     = lt
        
        # normalize between [min, max]
        minV = trcr.min(self.dimVs)
        maxV = trcr.max(self.dimVs)
        
        tr_norm = (trcr - minV) / (maxV - minV)

        self.tracer = tr_norm


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
                    .rename({'contour':self.dimEqV})
        
        print(tbl)

        return Table(tbl.squeeze(), self.dimEqV)

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
        
        eqDimIncre = ctr[-1] > ctr[0]
        
        if self.lt:
            if eqDimIncre == self.increase:
                if self.increase:
                    print('case 1: increase & lt')
                else:
                    print('case 1: decrease & lt')
            else:
                if self.increase:
                    print('case 2: increase & lt')
                else:
                    print('case 2: decrease & lt')
        else:
            if eqDimIncre == self.increase:
                if self.increase:
                    print('case 3: increase & gt')
                else:
                    print('case 3: decrease & gt')
            else:
                if self.increase:
                    print('case 4: increase & gt')
                else:
                    print('case 4: decrease & gt')
        
        tbl = histogram(mask, bins=[ctr.values], dim=self.dimVs) \
                .rename('AeqCTbl') \
                .rename({'contour':self.dimEqv})
        
        print(tbl)

        return Table(tbl.squeeze(), self.dimEqV)


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
                                 output_core_dims=[['contour']])

            ctr.coords['contour'] = np.linspace(0.0, levels-1.0, levels)
            
        else:
             # specifying levels of contours
            def mylinspace(tracer, levs):
                return tracer[..., None] - tracer[..., None] + levs

            ctr = xr.apply_ufunc(mylinspace,
                                 self.tracer.min(dim=self.dimVs), levels,
                                 dask='allowed',
                                 input_core_dims=[[], []],
                                 output_core_dims=[['contour']])

            ctr.coords['contour'] = levels

        return ctr


    def cal_integral_within_contours(self, contour, var=None, out_name=None):
        """
        Calculate integral of masked variable within
        pre-calculated tracer contours.

        Parameters
        ----------
        contour : xarray.DataArray
            A given contour levels.
        var  : xarray.DataArray
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
        if var is None:
            var = self.tracer - self.tracer + 1
        
        if out_name is None:
            if var is None:
                out_name = 'area'
            else:
                out_name = 'int' + var.name

        if self.lt:
            mskVar = var.where(self.tracer < contour)
        else:
            mskVar = var.where(self.tracer > contour)

        intVar = self.grid.integrate(mskVar, self.dimNs).rename(out_name)

        return intVar


    def cal_integral_within_contours_hist(self, contour, var=None, out_name=None):
        """
        Calculate integral of masked variable within
        pre-calculated tracer contours, using histogram method.

        Parameters
        ----------
        contour : xarray.DataArray
            A given contour levels.
        var  : xarray.DataArray
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
        if var is not None: 
            wei = self.grid.get_metric(self.tracer, self.dimNs) * var
        else:
            wei = self.grid.get_metric(self.tracer, self.dimNs)
        
        wei = wei.fillna(0.)
        
        # add a bin so that the result has the same length of contour
        if self.increase:
            inc = -0.1
        else:
            inc = 0.1
        
        re = []
        
        # unified the contour coordinate
        binNum = np.array(range(contour['contour'].shape[0])).astype(np.float32)
        
        if 'time' in contour.coords and False:
            for l in range(len(contour.time)):
                rng = {'time': l}
                
                tr  = self.tracer.isel(rng)
                ctr = contour.isel(rng)
                
                if 'time' in wei.coords:
                    wt = wei.isel(rng)
                else:
                    wt = wei
                
                # add a bin so that the result has the same length of contour
                # bins = np.insert(ctr.values, 0, ctr.values[0]+inc)
                print(ctr.values[0]+inc)
                bins = np.concatenate([[ctr.values[0]+inc], ctr.values])
                
                tmp = histogram(tr, bins=[bins], dim=self.dimVs, weights=wt)
                tmp[contour.name+'_bin'] = binNum
                
                re.append(tmp)
        else:
            # add a bin so that the result has the same length of contour
            print('ok1')
            bins = np.insert(contour.values, 0, contour.values[0]+inc)
            
            print('ok2')
            tmp = histogram(self.tracer, bins=[bins], dim=self.dimVs, weights=wei)
            print('ok3')
            tmp[contour.name+'_bin'] = binNum
            print('ok4')
            re.append(tmp)
        
        print('ok5')
        area = xr.concat(re, 'time').rename({contour.name+'_bin':'contour'}) \
                 .cumsum('contour').rename(out_name)
        print('ok6')
        area['time'] = self.tracer['time'].copy()
        print('ok7')
        
        return area


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
        qIntp = self.interp_to_coords(predef, dimEq, ctr) \
                    .rename({'new': 'contour'}) \
                    .rename(ctr.name)

        qIntp['contour'] = np.linspace(0, N-1, N)

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
        
        varIntp = xr.apply_ufunc(interp1d, predef, eqCoords, var,
                  kwargs={'inc': increasing},
                  dask='allowed',
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
        values : numpy.ndarray or xarray.DataArray
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
        


def eqLatitudes(area):
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


def mininumLats(lats):
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


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in ContourMethods')

