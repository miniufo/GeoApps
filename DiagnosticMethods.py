# -*- coding: utf-8 -*-
"""
Created on 2019.12.30

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from GeoApps.Application import Application


class Dynamics(Application):
    """
    This class is designed for calculating the dynamical methods.
    """
    def __init__(self, dset, grid=None, arakawa='A'):
        """
        Construct a Dynamics instance using a Dataset
        
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
        super().__init__(dset, grid=grid, arakawa=arakawa)

    
    """
    Three basic difference operators: divg, curl, grad
    """
    def divg(self, comps, dims, arakawa=None, iBCs='fixed'):
        """
        Calculate divergence as du/dx + dv/dy + dw/dz ...
        
        For example:
            du/dx            ->   divX   = divg(u, 'X')
            dv/dy            ->   divY   = divg(v, 'Y')
            dw/dz            ->   divZ   = divg(w, 'Z')
            du/dx+dv/dy      ->   divXY  = divg((u,v)  , ['X','Y'])
            dv/dy+dw/dz      ->   divVW  = divg((v,w)  , ['Y','Z'])
            du/dx+dw/dz      ->   divXZ  = divg((u,w)  , ['X','Z'])
            du/dx+dv/dy+dw/dz->   divXYZ = divg((u,v,w), ['X','Y','Z'])
        
        Parameters
        ----------
        comps: xarray.DataArray or list of xarray.DataArray
            Component(s) of a vector.
        dims: str or list of str
            Dimensions for differences calculation.  Length should be
            the same as that of comps.  Order of comps corresponds to
            that of the dims.
        arakawa: str
            The type of the grid, overwritting the default one. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        iBCs: str or dict of str
            Interior boundary conditions (isolated islands or topographies).
                e.g., {'X': 'extend', 'Y': 'fixed'}
        """
        if type(comps) is xr.DataArray:
            comps = [comps]
        
        if type(dims) is str:
            dims = [dims]
        
        if type(iBCs) is str:
            tmp = {}
            for dim in dims:
                tmp[dim] = iBCs
            iBCs = tmp
        
        if len(comps) != len(dims):
            raise Exception('lengths of comps ('+str(len(comps))+') and dims ('
                            +str(len(dims))+') should be the same')
        
        return sum([self._divg1D(c, d, arakawa=arakawa, iBC=iBCs[b])
                    for c, d, b in zip(comps, dims, iBCs)])


    def vort(self, u=None, v=None, w=None, components='k', arakawa=None, iBCs='fixed'):
        """
        Calculate vorticity component.  All the three components satisfy
        the right-hand rule so that we only need one function and input
        different components accordingly.
        
        For example:
            x-component (i) is: dw/dy - dv/dz  ->  vori = vort(v=v, w=w, 'i')
            y-component (j) is: du/dz - dw/dx  ->  vorj = vort(u=u, w=w, 'j')
            z-component (k) is: dv/dx - du/dy  ->  vork = vort(u=u, v=v, 'k')
            
            i,j components:  ->  vori,vorj      = vort(u=u,v=v,w=w, ['i','j'])
            all components:  ->  vori,vorj,vork = vort(u=u,v=v,w=w, ['i','j','k'])
        
        Parameters
        ----------
        u: xarray.DataArray
            X-component velocity.
        v: xarray.DataArray
            Y-component velocity.
        w: xarray.DataArray
            Z-component velocity.
        components: str or list of str
            Component(s) of the vorticity.  Order of component is the same as
            that of the outputs: vork, vorj, vori = vort(u,v,w, ['k','j','i'])
        arakawa : str
            The type of the grid, overwritting the default one. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        iBCs: str or dict of str
            Interior boundary conditions (isolated islands or topographies).
                e.g., {'X': 'extend', 'Y': 'fixed'}
        """
        scheme = self.arakawa if arakawa is None else arakawa
        
        if type(components) is str:
            components = [components]
        
        vors = []
        for comp in components:
            if   comp == 'i':
                dim = ['Y', 'Z']
                vor = self._vort2D(w, v, dim, arakawa=scheme, iBCs=iBCs)
            elif comp == 'j':
                dim = ['Z', 'X']
                vor = self._vort2D(u, w, dim, arakawa=scheme, iBCs=iBCs)
            elif comp == 'k':
                dim = ['X', 'Y']
                vor = self._vort2D(v, u, dim, arakawa=scheme, iBCs=iBCs)
            else:
                raise Exception('invalid component ' + str(comp) +
                                ', only in [k, j, i]')
            
            if scheme == 'A':
                vor = self.grid.interp(vor, dim[0])
                vor = self.grid.interp(vor, dim[1])
            
            vors.append(vor)
        
        return vors if len(vors) != 1 else vors[0]


    def grad(self, T, dims=['X', 'Y'], interp_back=True, arakawa=None, iBCs='fixed'):
        """
        Calculate spatial gradient components along each dimension given.
        For example:
            Tx, Ty     = grad(T, ['X', 'Y'])
            Tx, Ty, Tz = grad(T, ['X', 'Y', 'Z'])
        
        Parameters
        ----------
        T: xarray.DataArray
            A scalar variable.
        dims: list of str
            Dimensions for gradient.  Order of dims is the same as
            that of the outputs: `grdx, grdy = grad(T, ['X', 'Y'])`.
        interp_back: boolean
            Whether interpolate the gradient back to the grid point of T.
        arakawa : str
            The type of the grid, overwritting the default one. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        iBCs: str or dict of str
            Interior boundary conditions (isolated islands or topographies).
                e.g., {'X': 'extend', 'Y': 'fixed'}
        """
        if type(dims) is str:
            dims = [dims]
        
        if isinstance(iBCs, str):
            tmp = {}
            for dim in dims:
                tmp[dim] = iBCs
            iBCs = tmp
        
        if len(set(dims)) != len(dims):
            raise Exception('duplicated dimensions: ' + str(dims))
        
        grds = [self._grad1D(T, dim, interp_back, arakawa, iBC)
                for dim, iBC in zip(dims, iBCs)]
        
        return grds if len(grds) != 1 else grds[0]


    def Laplacian(self, var, dims=['X', 'Y'], iBCs='fixed'):
        """
        Calculate Laplacian of var as $\nabla^2 q$.  This operator can be
        either 2D or 3D, depending on the dims given.
        
        Parameters
        ----------
        var: xarray.DataArray
            A given variable.
        dims: list of str
            Dimensions for the operator.
        """
        # defined at tracer point
        grds = self.grad(var , dims, interp_back=False, iBCs=iBCs)
        lap  = self.divg(grds, dims, arakawa='C', iBCs='fixed')
        
        return lap
    
    def tension_strain(self, u, v):
        """
        Calculate tension strain as du/dx - dv/dy.
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        arakawa : str
            The type of the grid, overwritting the default one. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        """
        # defined at tracer point
        return self.divg([u, -v], ['X', 'Y'])
    
    def shear_strain(self, u, v, interp_back=False):
        """
        Calculate tension strain as dv/dx + du/dy.
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        arakawa : str
            The type of the grid, overwritting the default one. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        """
        # defined at vorticity point
        return self._vort2D(v, -u, ['X', 'Y'], interp_back)
    
    def deformation_rate(self, u, v):
        """
        Calculate sqrt(tension^2+shear^2).
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        """
        tension = self.tension_strain(u, v) # tracer point
        shear   = self.shear_strain  (u, v, True) # move to tracer point
        
        if self.arakawa == 'C':
            shear = self.grid.interp(shear, ['X', 'Y'])
        
        # defined at tracer point???
        # return np.sqrt(tension**2.0 + shear**2.0)
        return np.hypot(tension + shear)

    def Okubo_Weiss(self, u, v, dims=['Y', 'X']):
        """
        Calculate Okubo-Weiss parameter.
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        dims: list of str
            Dimensions for differences.  Order of dims should be the same as
            the storage of the variables.  E.g., if variable u is u[t, z, y, x]
            then dims should be ['y', 'x'], not ['x', 'y']!
        """
        deform = self.deformation_rate(u, v)  # tracer point
        
        interp = True if self.arakawa == 'C' else False
        
        curlZ  = self._vort2D(v, u, ['X', 'Y'], interp) # vorticity point
        
        # defined at tracer point???
        return deform**2.0 - curlZ**2.0


    def cal_kinetic_energy(self, comps, dims=['X', 'Y']):
        """
        Calculate kinetic energy.
        
        Parameters
        ----------
        comps: xarray.DataArray or list of xarray.DataArray
            Component(s) of a vector.
        dims: list of str
            Dimensions for the velocity components.
        """
        grid = self.grid
        
        if type(dims) is str:
            dims = [dims]
        
        if len(set(dims)) != len(dims):
            raise Exception('duplicated dimensions: ' + str(dims))
        
        Cs = []
        for c, d in zip(comps, dims):
            C = c**2
            
            if self.arakawa == 'C':
                C = grid.interp(C, d)
            
            Cs.append(C)
            

        # interpolate to get tracer-point energy
        KE = (sum(Cs)/2).rename('KE')

        return KE
    

    def cal_squared_gradient(self, T, dims, boundary=None):
        """
        Calculate squared gradient magnitude.
        
        Parameters
        ----------
        T: xarray.DataArray
            A given variable to be defined at tracer point.
        dims: list of str
            Dimensions for differences.  Order of dims should be the same as
            the storage of the variables.  E.g., if variable u is u[t, z, y, x]
            then dims should be ['y', 'x'], not ['x', 'y']!
        boundary: dict
        """
        grid = self.grid
        
        grd = []
        
        if boundary == None:
            boundary = []
        
        for dim in dims:
            if dim in boundary:
                # calculate gradient with half-grid offset
                tmp = grid.derivative(T, dim, boundary=boundary[dim])
                
                # interpolate to get tracer-point gradient magnitude
                grd.append(grid.interp(tmp**2, dim, boundary=boundary[dim]))
            else:
                # calculate gradient with half-grid offset
                tmp = grid.derivative(T, dim)
                
                # interpolate to get tracer-point gradient magnitude
                grd.append(grid.interp(tmp**2, dim))
        
        tmp = grd[0] - grd[0] # set to zero for sumup
        
        for g in grd:
            tmp += g
        
        return tmp.rename('grdS'+T.name)
    
    def divg2(self, u, v, dims=['Y', 'X'], arakawa=None):
        """
        Calculate divergence as du/dx + dv/dy.
        
        Parameters
        ----------
        u: xarray.DataArray
            X-component velocity.
        v: xarray.DataArray
            Y-component velocity.
        dims: list of str
            Dimensions for differences.  Order of dims should be the same as
            the storage of the variables.  E.g., if variable u is u[t, z, y, x]
            then dims should be ['y', 'x'], not ['x', 'y']!
        arakawa : str
            The type of the grid. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        """
        print('this is deprecated.  Using divg((u,v) dims=[\'Y\',\'X\']) instead')
        
        grid = self.grid
        scheme = self.arakawa if arakawa is None else arakawa

        if scheme == 'A':
            U = grid.interp(u, dims[1])
            V = grid.interp(v, dims[0])
        elif scheme == 'B': 
            U = grid.interp(u, dims[0])
            V = grid.interp(v, dims[1])
        elif scheme == 'C':
            U = u
            V = v
        else:
            raise Exception(f'unsupported type of grid: \'{scheme}\'')
        
        dUdx = grid.diff(U * grid.get_metric(U, dims[0]), dims[1])
        dVdy = grid.diff(V * grid.get_metric(V, dims[1]), dims[0])
        area = grid.get_metric(dVdy, [dims[1],dims[0]])
        
        return (dUdx + dVdy) / area


    def curl(self, u, v, dims=['Y', 'X'], arakawa=None):
        """
        Calculate curl as dv/dx - du/dy.
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        dims: list of str
            Dimensions for differences.  Order of dims should be the same as
            the storage of the variables.  E.g., if variable u is u[t, z, y, x]
            then dims should be ['y', 'x'], not ['x', 'y']!
        arakawa : str
            The type of the grid, overwritting the default one. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        """
        print('this is deprecated.  Using vort(u=u, v=v, components=\'k\') instead')
        grid = self.grid
        scheme = self.arakawa if arakawa is None else arakawa

        if scheme == 'A':
            U = grid.interp(u, dims[1])
            V = grid.interp(v, dims[0])
        elif scheme == 'B': 
            U = grid.interp(u, dims[0])
            V = grid.interp(v, dims[1])
        elif scheme == 'C':
            U = u
            V = v
        else:
            raise Exception(f'unsupported type of grid: \'{scheme}\'')
        
        dVdx = grid.diff(V * grid.get_metric(V, dims[0]), dims[1])
        dUdy = grid.diff(U * grid.get_metric(U, dims[1]), dims[0])
        area = grid.get_metric(dVdx, [dims[1], dims[0]])
        
        re = (dVdx - dUdy) / area
        
        if self.arakawa == 'A':
            re = grid.interp(re, dims[0])
            re = grid.interp(re, dims[1])
        
        return re
    

    """
    Helper (private) methods are defined below
    """
    def _divg1D(self, vel, dim, arakawa=None, iBC='fixed'):
        """
        Calculate differences of fluxes as:
        [du * dy*dz/dx,     x-component
         dv * dx*dz/dy,     y-component
         dw * dx*dy/dz].    z-component
        This is different from _grad1D as it requires proper
        weighting (using metrics)
        
        Parameters
        ----------
        vel: xarray.DataArray
            Velocity component.
        dim: list of str
            Dimension for the divergence.  Should be one of ['Z', 'Y', 'X']
        arakawa: str
            The type of the grid, overwritting the default one. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        iBC: str
            Interior boundary conditions (isolated islands or topographies).
        """
        grid = self.grid
        scheme = self.arakawa if arakawa is None else arakawa
        get_metric = self._get_metric

        if scheme == 'A':
            VEL = grid.interp(vel, dim)
        elif scheme == 'B':
            if   dim == 'X':
                VEL = grid.interp(vel, 'Y')
            elif dim == 'Y':
                VEL = grid.interp(vel, 'X')
            elif dim == 'Z':
                raise Exception('not implemented for d/dz on B grid')
                # VEL == vel
        elif scheme == 'C':
            VEL = vel
        else:
            raise Exception(f'unsupported type of grid: \'{scheme}\'')
        
        if   dim == 'X': # metrics may vary along Y? partial cell
            if iBC == 'extend':
                dname = self._get_dname(VEL, 'X')
                VEL = VEL.ffill(dname, 1).bfill(dname, 1)
            dy  = get_metric(VEL,'Y')
            div = grid.diff(VEL * dy, 'X') # weighted by hFacW/hFacS?
            dA  = get_metric(div, ['X', 'Y'])
            div = div / dA
            
        elif dim == 'Y': # metrics may vary along X? partial cell
            if iBC == 'extend':
                dname = self._get_dname(VEL, 'Y')
                VEL = VEL.ffill(dname, 1).bfill(dname, 1)
            dx  = get_metric(VEL,'X')
            div = grid.diff(VEL * dx, 'Y') # weighted by hFacW/hFacS?
            dA  = get_metric(div, ['X', 'Y'])
            div = div / dA
            
        elif dim == 'Z': # metrics may vary along z?
            if iBC == 'extend':
                dname = self._get_dname(VEL, 'Z')
                VEL = VEL.ffill(dname, 1).bfill(dname, 1)
            div = grid.diff(VEL, 'Z')
            dz  = get_metric(div, 'Z')
            div = div / dz
        
        return div

    def _vort2D(self, v, u, dims=['X', 'Y'], interp_back=True, arakawa=None,
                iBCs='fixed'):
        """
        Calculate vorticity component.  All the three components satisfy
        the right-hand rule so that we only need one function and input
        different components accordingly.
        
        x-component (i) is: dw/dy - dv/dz  ->  _vort2D(w, v, ['Y', 'Z'])
        y-component (j) is: du/dz - dw/dx  ->  _vort2D(u, w, ['Z', 'X'])
        z-component (k) is: dv/dx - du/dy  ->  _vort2D(v, u, ['V', 'U'])
        
        Parameters
        ----------
        u : xarray.DataArray
            X-component velocity.
        v : xarray.DataArray
            Y-component velocity.
        dims: list of str
            Dimensions for differences.  Order of dims should be the same as
            the storage of the variables.  E.g., if variable u is u[t, z, y, x]
            then dims should be ['y', 'x'], not ['x', 'y']!
        arakawa : str
            The type of the grid. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        iBCs: str or dict of str
            Interior boundary conditions (isolated islands or topographies).
                e.g., {'X': 'extend', 'Y': 'fixed'}
        """
        grid = self.grid
        scheme = self.arakawa if arakawa is None else arakawa
        get_metric = self._get_metric
        
        if type(iBCs) is str:
            tmp = {}
            for dim in dims:
                tmp[dim] = iBCs
            iBCs = tmp
        
        if scheme == 'A':
            U = grid.interp(u, dims[0])
            V = grid.interp(v, dims[1])
        elif scheme == 'B': 
            U = grid.interp(u, dims[1])
            V = grid.interp(v, dims[0])
        elif scheme == 'C':
            U = u
            V = v
        else:
            raise Exception(f'unsupported type of grid: \'{scheme}\'')
        
        dx  = get_metric(U, dims[0])
        dy  = get_metric(V, dims[1])
        
        if iBCs[dims[0]] == 'extend':
            V = V.ffill(dims[0], 1).bfill(dims[0], 1)
        if iBCs[dims[1]] == 'extend':
            U = U.ffill(dims[1], 1).bfill(dims[1], 1)
        
        vor = grid.diff(V * dy, dims[0]) - grid.diff(U * dx, dims[1])
        dA  = get_metric(vor, dims)
        vor = vor / dA
        
        if interp_back:
            if scheme == 'A':
                vor = grid.interp(vor, dims)
            elif scheme == 'C':
                vor = vor
            else:
                raise Exception(f'not implemented for \'{scheme}\' grid')
        
        return vor


    def _grad1D(self, var, dim, interp_back=True, arakawa=None, iBC='fixed'):
        """
        Calculate spatial gradient component along a given dimension.
        This is different from _divg1D as it does not requires proper
        weighting (metrics)???
        
        Parameters
        ----------
        var: xarray.DataArray
            A scalar variable.
        dim: str
            Dimensions for gradient, should be one of ['Z', 'Y', 'X'].
        interp_back: boolean
            Whether interpolate the gradient back to the grid point of var.
        arakawa : str
            The type of the grid, overwritting the default one. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
        iBC: str
            Interior boundary conditions (isolated islands or topographies).
        """
        grid = self.grid
        scheme = self.arakawa if arakawa is None else arakawa
        
        if iBC == 'extend':
            dname = grid.axes[dim].name
            var = var.ffill(dname, 1).bfill(dname, 1)
        
        # Flux should be weighted but not the gradient???
        grd = grid.derivative(var, dim)
        
        if interp_back:
            if scheme == 'A':
                # ensure that the gradients are still at A point
                grd = grid.interp(grd, dim)
            elif scheme == 'B': 
                # ensure that the gradients are defined at velocity point
                if   dim == 'X':
                    grd = grid.interp(var, 'Y')
                elif dim == 'Y':
                    grd = grid.interp(var, 'X')
                elif dim == 'Z':
                    raise Exception('not implemented for grad-Z on B grid')
                else:
                    raise Exception('only spatial gradients are supported')
            elif scheme == 'C':
                grd = grd # do nothing
            else:
                raise Exception(f'unsupported type of grid: {scheme}')
        
        return grd
    
    def _get_metric(self, var, dim):
        '''
        A simple wrapper for grid.get_metric.
        Use 1 to scale flux if not available in metrics
        '''
        grid = self.grid
        return grid.get_metric(var, dim)\
                    if frozenset(dim) in grid._metrics else 1
    
    def _get_dname(self, var, dim):
        coords = self.grid.axes[dim].coords
        
        for coord in coords:
            if coords[coord] in var.dims:
                return coords[coord]
        
        raise Exception('no dim name for '+str(dim))
        


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in DiagnosticMethods')
    
    from GeoApps.GridUtils import add_latlon_metrics
    
    ds = xr.tutorial.open_dataset('air_temperature')
    
    dset, grid = add_latlon_metrics(ds, {'lat':'lat', 'lon':'lon'})
    
    dyn = Dynamics(dset, grid, 'A')
    
    grdx, grdy = dyn.grad(dset.air, ['X','Y'])
    
    grdx[0].plot()


