# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:05:39 2024

@author: FPitscheider
"""

"""
For estimating the flow depth in the reaches two options are given:

    - The flow depth caculation proposed by Manning (---). This formula is based on Manning's roughness coefficient n.
    This formula is suitable for river networks, where the flow depth does not change a lot in comparison to the sediments
    on the river bed.
    Flow velocity is also calculated based on Manning's n, as proposed in to Mannig (---) 
    Source: __
    
    - The flow depth calculation proposed by Ferguson (2007). This formula estimated the flow depth according to the D84
    of the deposited material. This formula is suitable for river networks, where the flow depth has large changes with discharge
    in comparison to the sediments on the river bed. Such changes lead to higher relative bed roughness and would imply changes 
    in Manning's n. This is especcialy the case in mountain streams.
    Flow velocity is calculated according to Nitsche et al. (2011) based again on the D84.
    
    Sources:
        Ferguson, R. (2007), Flow resistance equations for gravel- and boulder-bed streams, Water Resour. Res., 43, W05427, 
    doi:10.1029/2006WR005422. 
        Nitsche, M., D. Rickenmann, J. M. Turowski, A. Badoux, and J. W. Kirchner (2011), Evaluation of bedload transport 
    predictions using flow resistance equations to account for macro-roughness in steep mountain streams, Water Resour. 
    Res., 47, W08513, doi:10.1029/2011WR010645. 

"""

import numpy as np
import numpy.matlib
from supporting_functions import D_finder    

def h_manning(reach_data, slope, Q, t):
    """
    The Manning equation.
    """
    h = np.power(Q[t,:] * reach_data.n / (reach_data.wac * np.sqrt(slope[t])), 3/5)
    v = 1 / reach_data.n * np.power(h, 2/3) * np.sqrt(slope[t])

    return h, v

def h_ferguson(reach_data, slope, Q, t):
    """
    """
    
    #calculate water depth and velocity with the Ferguson formula (2007)
    q_star = Q.iloc[t,:] / (reach_data.wac * np.sqrt(9.81 * slope[t] * reach_data.D84**3))
    
    #𝑝…𝑖𝑓 𝑞∗<100 → 𝑝=0.24, 𝑖𝑓 𝑞^∗>100 → 𝑝=0.31
    p = np.where(q_star < 100, 0.24, 0.31)
    
    h = 0.015 * reach_data.D84 * (q_star**(2*p)) / (p**2.5)    
    v = (np.sqrt(9.81 * h * slope[t])* 6.5 * 2.5 * (h / reach_data.D84)) / np.sqrt((6.2 ** 2) * (2.5 ** 2) * ((h / reach_data.D84) ** (5/3)))
   
    return h, v

def choose_flow_depth(reach_data, slope, Q, t, flow_depth):
    if flow_depth == 1:
        [h, v] = h_manning(reach_data, slope, Q, t)
    
    elif flow_depth == 2:
        [h, v] = h_ferguson(reach_data, slope, Q, t)
    
    return h, v