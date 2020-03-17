# -*- coding: utf-8 -*-
'''
Created on 2020.02.04

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
'''
import numpy as np


'''
Here defines all the constants that are commonly used in earth sciences
'''
# Radius of the Earth (m)
Re = 6371200.0

# distance of unit degree at the equator
deg2m = 2.0 * np.pi * Re / 360.0

# Gravitational acceleration g (m s^-2)
g = 9.80665

# Rotating angular speed of the Earth (1)
omega = 7.292e-5

# Density of air (kg m^-3) at sea level at a temperature of 15 degree
rho = 1.225

# Thermal capacity of ideal gas under constant pressure (J kg^-1 K^-1)
Cp = 1004.88

# Thermal capacity of sea water (J kg^-1 K^-1)
Cp_sw = 3994.0

# Saturated vapor pressure above pure water surface (Pa)
E0 = 610.78

# Constant of dry air (J kg^-1 K^-1)
Rd = 287.04

# Constant of vapor (J kg^-1 K^-1)
Rv = 461.52

# Vertical temperature decrease rate
rd = g / Cp

# Thermal capacity of ideal gas under constant volume (J kg^-1 K^-1)
Cv =Cp-Rd

# kappa, just the ratio of the two constants
kappa = Rd/Cp
