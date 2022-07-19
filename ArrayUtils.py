# -*- coding: utf-8 -*-
'''
Created on 2020.02.04

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
'''
import numpy as np


def interp1d(x, xf, yf, inc=True, outside=None):
    """
    Wrapper of np.interp, taking into account the decreasing case.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xf : 1-D sequence of floats
        The x-coordinates of the data points.
    yf : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as `xp`.
    inc : boolean
        xf is increasing or decresing.

    Returns
    ----------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as `x`.
    """
    # print(f"x : {x.shape} \nxf: {xf.shape} \nyf: {yf.shape}")
    if inc: # increasing case
        re = np.interp(x, xf, yf)
    else: # decreasing case
        # print(f"x: {x} \n xf: {xf} \n yf: {yf}")
        re = np.interp(x, xf[::-1], yf[::-1])

    # print(f"x: {x} \n xf: {xf} \n yf: {yf} \n y: {re}")

    return re

