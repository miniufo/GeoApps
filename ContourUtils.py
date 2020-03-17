# -*- coding: utf-8 -*-
'''
Created on 2020.02.04

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
'''
import numpy as np


def contour_enclosed_area_np(verts):
    """
    Compute the area enclosed by a contour.  Copied from
    https://github.com/rabernat/floater/blob/master/floater/rclv.py
    
    Parameters
    ----------
    verts : array_like
        2D shape (N,2) array of vertices. Uses scikit image convetions
        (j,i indexing)
        
    Returns
    ----------
    area : float
        Area of polygon enclosed by verts. Sign is determined by vertex
        order (cc vs ccw)
    """
    verts_roll = np.roll(verts, 1, axis=0)
    
    # use scikit image convetions (j,i indexing)
    area_elements = ((verts_roll[:,1] + verts[:,1]) *
                     (verts_roll[:,0] - verts[:,0]))
    
    # absolute value makes results independent of orientation
    return abs(area_elements.sum())/2.0


def contour_enclosed_area_py(verts):
    """
    Compute the area enclosed by a contour.  Copied from
    https://arachnoid.com/area_irregular_polygon/
    
    Parameters
    ----------
    verts : array_like
        2D shape (N,2) array of vertices. Uses scikit image convetions
        (j,i indexing)
        
    Returns
    ----------
    area : float
        Area of polygon enclosed by verts. Sign is determined by vertex
        order (cc vs ccw)
    """
    a = 0
    ox, oy = verts[0]

    for x, y in verts[1:]:
        a += (x * oy - y * ox)
        ox, oy = x, y

    return a / 2


def contour_length_np(verts):
    """
    Compute the length of a contour.  Copied from
    https://github.com/rabernat/floater/blob/master/floater/rclv.py
    
    Parameters
    ----------
    verts : array_like
        2D shape (N,2) array of vertices. Uses scikit image convetions
        (j,i indexing)
        
    Returns
    ----------
    Perimeter : float
        Perimeter of a contour by verts.
    """
    verts_roll = np.roll(verts, 1, axis=0)
    
    diff = verts_roll - verts
    
    p = np.sum(np.hypot(diff[:,0], diff[:,1]))
    
    return p


def contour_length_py(verts):
    """
    Compute the length of a contour.  Copied from
    https://arachnoid.com/area_irregular_polygon/
    
    Parameters
    ----------
    verts : array_like
        2D shape (N,2) array of vertices. Uses scikit image convetions
        (j,i indexing)
        
    Returns
    ----------
    Perimeter : float
        Perimeter of a contour by verts.
    """
    p = 0
    ox, oy = verts[0]
    
    for x, y in verts[1:]:
        p += abs((x - ox) + (y - oy) * 1j)
        ox, oy = x, y
    
    return p


def is_contour_closed(con):
    """
    Whether the contour is a closed one or intersect with boundaries.
    """
    return np.all(con[0] == con[-1])


'''
Helper (private) methods are defined below
'''



'''
Testing codes for each class
'''
if __name__ == '__main__':
    print('start testing in ContourUtils.py')
    
    import xarray as xr
    import matplotlib.pyplot as plt
    
    
    
