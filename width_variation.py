"""
Created on Wed Jan 2 15:52:00 2025

@author: FPitscheider and DDoolaeghe
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

import numpy as np

from constants import GRAV, R_VAR


def dynamic_width_Lugo(width, D50, slopes, Q_t):

    # Dimensionless Stream Power
    w_star = (Q_t * slopes) / (width * np.sqrt(GRAV * R_VAR * (D50 ** 3)))

    r = np.maximum(0.2, np.minimum(2.36 * w_star + 0.09, 1))

    new_width = width * r

    return new_width

def choose_widthVar(reach_data, SedimSys, Q, t, indx_width_calc):

    if  indx_width_calc == 1:
        width_t = SedimSys.width[t] # Static width

    elif indx_width_calc == 2:
        width_t = dynamic_width_Lugo(SedimSys.width[t], reach_data.D50, reach_data.slope, Q[t,:])

    SedimSys.width[t] = width_t

    return SedimSys.width