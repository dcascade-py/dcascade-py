# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 08:42:09 2025

@author: FPitscheider
"""
'''
Dynamic estimaion of the active transport width (active channel width wac) with the  approach based on dimensionless streampower by Lugo et al. (2015)

    Garcia Lugo, G. A., W. Bertoldi,A. J. Henshaw, and A. M. Gurnell(2015), The effect of lateralconfinement on gravel bed rivermorphology,Water Resour. Res.,51,7145–7158, doi:10.1002/2015WR017081.
'''

import numpy as np
from constants import (
    GRAV,
    RHO_S,
    RHO_W,
    R_VAR
)

def static_width(reach_data, SedimSys, Q, t):
    
    SedimSys.width[t] = reach_data.wac
    
    return SedimSys.width


def dynamic_width_Lugo(reach_data, SedimSys, Q, t):
    
    #inputs
    d50 = reach_data.D50
    slope = reach_data.slope
    
    if hasattr(reach_data, 'wac_bf'):
        width = reach_data.wac_bf
    else:
        width = reach_data.wac
    
    # Dimensionless Stream Power
    w_star = (Q[t, :] * slope) / (width * np.sqrt(GRAV * R_VAR * (d50 ** 3)))

    r = np.maximum(0.2, np.minimum(2.36 * w_star + 0.09, 1))

    SedimSys.width[t] = width * r

    return SedimSys.width

def choose_width(reach_data, SedimSys, Q, t, width_calc):

    if  width_calc == 1:
        SedimSys.width = static_width(reach_data, SedimSys, Q, t)
       
    
    elif width_calc == 2:
        SedimSys.width = dynamic_width_Lugo(reach_data, SedimSys, Q, t)

    return SedimSys.width
