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

from constants import GRAV
import numpy as np
import numpy.matlib
from supporting_functions import D_finder
from line_profiler import profile

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
    q_star = Q[t,:] / (reach_data.wac * np.sqrt(GRAV * slope[t] * reach_data.D84**3))
    
    #𝑝…𝑖𝑓 𝑞∗<100 → 𝑝=0.24, 𝑖𝑓 𝑞^∗>100 → 𝑝=0.31
    p = np.where(q_star < 100, 0.24, 0.31)
    
    h = 0.015 * reach_data.D84 * (q_star**(2*p)) / (p**2.5)    
    v = (np.sqrt(GRAV * h * slope[t])* 6.5 * 2.5 * (h / reach_data.D84)) / np.sqrt((6.2 ** 2) * (2.5 ** 2) * ((h / reach_data.D84) ** (5/3)))
   
    return h, v

def choose_flow_depth(reach_data, slope, Q, t, flow_depth):
    if flow_depth == 1:
        [h, v] = h_manning(reach_data, slope, Q, t)
    
    elif flow_depth == 2:
        [h, v] = h_ferguson(reach_data, slope, Q, t)
    
    elif flow_depth == 3:
        [h, v] = h_ferguson(reach_data, slope, Q, t)
        
    return h, v

@profile
def hypso_manning_Q(H, Hsec, dy, n, slope):
    #return a Q based on given H (water level) and height section ()
    #Adapted from Marco Redolfi solver by JMR
    g = 9.81
    Npoints = len(Hsec)
    Qsec = 0
    bsec = 0
    
    Hsave = np.zeros(Npoints - 1)
    Vsave = np.zeros(Npoints - 1)
    Csave = np.zeros(Npoints - 1)
    
    # Elevation adjustment
    Hsec = Hsec - np.min(Hsec)
    
    for j_point in range(Npoints - 1):
        hl = H - Hsec[j_point]
        hr = H - Hsec[j_point + 1]
        
        if hl > 0 and hr > 0:
            # Trapezoidal slice
            hm = (hl + hr) / 2
            Bi = np.sqrt(dy**2 + (hl - hr)**2)
            Ai = hm * dy
            Rhi = Ai / Bi
            C = hm**(1/6) / n
            Qi = C * Ai * np.sqrt(Rhi) * np.sqrt(slope)
            
            bsec += dy
            Qsec += Qi
            Hsave[j_point] = hm
            Vsave[j_point] = Qi / Ai
            Csave[j_point] = C
            
        elif hl > 0 or hr > 0:
            # Triangular slice
            hmax = max(hl, hr)
            bi = dy * hmax / abs(hr - hl)
            Bi = np.sqrt(bi**2 + hmax**2)
            Ai = hmax * bi / 2
            Rhi = Ai / Bi
            C = hmax**(1/6) / n
            Qi = C * Ai * np.sqrt(Rhi) * np.sqrt(slope)
            
            bsec += bi
            Qsec += Qi
            #save H, V and Chezy C
            Hsave[j_point] = hmax
            Vsave[j_point] = Qi / Ai
            Csave[j_point] = C
       
    JS = {'Hsave': Hsave, 'Vsave': Vsave, 'Csave': Csave}
    return Qsec, bsec, JS

def hypso_ferguson_Q(H, Hsec, dy, D84, slope):
    #return a Q based on given H (water level) and height section with D84
    #Adapted from Marco Redolfi solver by JMR
    g = 9.81
    Npoints = len(Hsec)
    Qsec = 0
    bsec = 0
    
    Hsave = np.zeros(Npoints - 1)
    Vsave = np.zeros(Npoints - 1)

    sqrt_g_slope = np.sqrt(g * slope)
    
    D84_ratio_factor = 6.5 * 2.5 / np.sqrt(6.2**2 * 2.5**2)
    Hsec = Hsec - np.min(Hsec)
    
    for j_point in range(Npoints - 1):
        hl = H - Hsec[j_point]
        hr = H - Hsec[j_point + 1]
        
        if hl > 0 and hr > 0:
            # Trapezoidal slice
            hm = (hl + hr) / 2
            Ai = hm * dy
            s8f = (6.5 * 2.5 * (hm / D84)) / np.sqrt(6.5**2 + (2.5**2) * (hm / D84)**(5/3))
            v_flow = s8f * sqrt_g_slope  # Ferguson's C * sqrt(S)
#            v_flow = sqrt_g_slope * hm * D84_ratio_factor / np.sqrt((hm / D84) ** (5/3))
            Qi = v_flow * Ai
            
            bsec += dy
            Qsec += Qi
            Hsave[j_point] = hm
            Vsave[j_point] = v_flow
            
        elif hl > 0 or hr > 0:
            # Triangular slice
            hmax = max(hl, hr)
            bi = dy * hmax / abs(hr - hl)
            Ai = hmax * bi / 2
            s8f = (6.5 * 2.5 * (hmax / D84)) / np.sqrt(6.5**2 + (2.5**2) * (hmax / D84)**(5/3))
            v_flow = s8f * sqrt_g_slope  # Ferguson's C * sqrt(S)
            Qi = v_flow * Ai
            
            bsec += bi
            Qsec += Qi
            Hsave[j_point] = hmax
            Vsave[j_point] = v_flow
        
    JS = {'Hsave': Hsave, 'Vsave': Vsave}
    return Qsec, bsec, JS

def hypso_ferguson_Q_vec(H, Hsec, dy, D84, slope):
    """Compute flow rate Q using Ferguson's 2007 roughness model, vectorized with clarity."""
    g = 9.81
    sqrt_g_slope = np.sqrt(g * slope)
    Hsec -= np.min(Hsec)  # Normalize elevation

    # Compute depths relative to H
    hl = H - Hsec[:-1]  # Left depth
    hr = H - Hsec[1:]   # Right depth

    # Identify trapezoidal and triangular sections
    is_trapezoid = (hl > 0) & (hr > 0)  # Both points submerged
    is_triangle = (hl > 0) ^ (hr > 0)   # Only one point submerged

    # Compute hydraulic depth
    h_wet = np.zeros_like(hl)
    h_wet[is_trapezoid] = (hl[is_trapezoid] + hr[is_trapezoid]) / 2  # Mean depth for trapezoidal sections
    h_wet[is_triangle] = np.maximum(hl[is_triangle], hr[is_triangle])  # Max depth for triangular sections

    # Ferguson roughness factor
    s8f = (6.5 * 2.5 * (h_wet / D84)) / np.sqrt(6.5**2 + (2.5**2) * (h_wet / D84)**(5/3))

    # Compute wetted area
    A_wet = np.zeros_like(hl)
    A_wet[is_trapezoid] = h_wet[is_trapezoid] * dy  # Trapezoidal area
    A_wet[is_triangle] = (h_wet[is_triangle]**2 * dy) / (2 * np.abs(hr[is_triangle] - hl[is_triangle]))  # Triangular area

    # Wetted width
    b_wet = np.zeros_like(hl)
    b_wet[is_trapezoid] = dy  # Full section width for trapezoidal slices
    b_wet[is_triangle] = 2 * A_wet[is_triangle] / h_wet[is_triangle]  # Base width for triangular sections

    # Compute discharge Todo: move s8f * sqrt_g_slope from twice-calced below.
    Q_wet = s8f * sqrt_g_slope * A_wet  # Velocity * Area

    # Sum total discharge and wetted width
    return np.sum(Q_wet), np.sum(b_wet), {'Hsave': h_wet, 'Vsave': s8f * sqrt_g_slope}


def hypso_ferguson_Q_vec_srhfac(H, Hsec, dy, D84, slope, srhfac):
    """Compute flow rate Q using Ferguson's 2007 roughness model, vectorized with clarity."""
    g = 9.81
    Hsec -= np.min(Hsec)  # Normalize elevation

    # Compute depths relative to H
    hl = H - Hsec[:-1]  # Left depth
    hr = H - Hsec[1:]   # Right depth

    # Identify trapezoidal and triangular sections
    is_trapezoid = (hl > 0) & (hr > 0)  # Both points submerged
    is_triangle = (hl > 0) ^ (hr > 0)   # Only one point submerged

    # Compute hydraulic depth
    h_wet = np.zeros_like(hl)
    h_wet[is_trapezoid] = (hl[is_trapezoid] + hr[is_trapezoid]) / 2  # Mean depth for trapezoidal sections
    h_wet[is_triangle] = np.maximum(hl[is_triangle], hr[is_triangle])  # Max depth for triangular sections

    # Ferguson roughness factor
    s8f = (6.5 * 2.5 * (h_wet / D84)) / np.sqrt(6.5**2 + (2.5**2) * (h_wet / D84)**(5/3))
    
    #implement slope reduction based on depth
    sqrt_g_slope = np.sqrt(g * slope * (1-h_wet*srhfac))
    
    # Compute wetted area
    A_wet = np.zeros_like(hl)
    A_wet[is_trapezoid] = h_wet[is_trapezoid] * dy  # Trapezoidal area
    A_wet[is_triangle] = (h_wet[is_triangle]**2 * dy) / (2 * np.abs(hr[is_triangle] - hl[is_triangle]))  # Triangular area

    # Wetted width
    b_wet = np.zeros_like(hl)
    b_wet[is_trapezoid] = dy  # Full section width for trapezoidal slices
    b_wet[is_triangle] = 2 * A_wet[is_triangle] / h_wet[is_triangle]  # Base width for triangular sections

    # Compute discharge  
    V_wet = s8f * sqrt_g_slope
    Q_wet = V_wet * A_wet  # Velocity * Area

    # Sum total discharge and wetted width
    return np.sum(Q_wet), np.sum(b_wet), {'Hsave': h_wet, 'Vsave': V_wet}
