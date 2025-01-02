# -*- coding: utf-8 -*-
"""
Created on Wed Jan 2 15:52:00 2025

@author: DDoolaeghe
"""

'''
Functions for variying the width per time step and reach, as a function of, for example:
    - reach hypsometry (water level, discharge)
    - trend with time ..etc

The formula by Lugo (2015) is developped on flume experiments and link the insteantaneous 
    active width to the water width. Here we use it as it is used in Bizzi (2021),
    to link the water width to the bankfull width (yearly active width).
    Lugo et al. (2015), The effect of lateral confinement on gravel bed river morphology.
    Bizzi et al. (2021), Sediment transport at the network scale and its link to channel morphology in the braided Vjosa River system.    

'''

from constants import GRAV, R_VAR
import numpy as np
exponent_a = 1.5 # exponent a between 1-2, typically 1.5


def vary_width_Lugo(width, Q, reach_data, t):
    # reach input parameters                   
    d50s = reach_data.D50
    slopes = reach_data.slope
    max_widths = reach_data.wac 
    
    # Dimentionless stream power:
    w_star = (Q[t, :] * slopes) / (max_widths * np.sqrt(GRAV * R_VAR * d50s**3))
    
    # active width        
    width[t] = np.maximum(0.20, np.minimum(2.36 * w_star + 0.09, 1)) * max_widths
   
def choose_widthVariation(reach_data, width, Q, t, h, width_vary):
    if width_vary == 1:
        width = width    
    
    elif width_vary == 2:
        width = vary_width_Lugo(width, Q, reach_data, t)
    
    return width