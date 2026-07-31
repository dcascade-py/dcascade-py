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
from scipy.optimize import brentq


from constants import GRAV
from d_finder import D_finder


def h_manning(reach_data, SedimSys, Q, t):
    """
    The Manning equation.
    """

    h = np.power(Q[t,:] * reach_data.n / (SedimSys.width[t] * np.sqrt(SedimSys.slope[t])), 3/5)
    v = 1 / reach_data.n * np.power(h, 2/3) * np.sqrt(SedimSys.slope[t])

    return h, v

def h_ferguson(reach_data, SedimSys, Q, t):
    """
    """

    #calculate water depth and velocity with the Ferguson formula (2007)
    q_star = Q[t,:] / (SedimSys.width[t] * np.sqrt(GRAV * SedimSys.slope[t] * reach_data.D84**3))

    #𝑝…𝑖𝑓 𝑞∗<100 → 𝑝=0.24, 𝑖𝑓 𝑞^∗>100 → 𝑝=0.31
    p = np.where(q_star < 100, 0.24, 0.31)

    h = 0.015 * reach_data.D84 * (q_star**(2*p)) / (p**2.5)
    v = (np.sqrt(GRAV * h * SedimSys.slope[t])* 6.5 * 2.5 * (h / reach_data.D84)) / np.sqrt((6.5 ** 2) + (2.5 ** 2) * ((h / reach_data.D84) ** (5/3)))

    return h, v

def h_chezy(reach_data, SedimSys, Q, t, min_slope = 0.0001):
    """
    The Chezy equation. Using C = 2.5.ln(11h/e) (after Walter Bertoldi). 
    Therefore needs a solver, to invert for h.
    """
    
    widths = SedimSys.width[t]
    slopes = SedimSys.slope[t]
    d50s = reach_data.D50          #choice: we keep using the initial one for roughness
    Q_t = Q[t, :]

    h = np.zeros_like(Q_t)
    v = np.zeros_like(Q_t)
    
    Q_TOL = 1e-12

    for i in range(len(Q_t)):
        
        B = widths[i]
        S = slopes[i]
        D50 = d50s[i]
        Qi_target = Q_t[i]
        
        # if Q is too small:
        if Qi_target <= Q_TOL:
            h[i] = 0.0
            v[i] = 0.0            
            # print(
            #     f"Warning: target Q below TOL "
            #     f"at t={t}, reach={i}. "
            #     f"Qtarget={Qi_target}, "
            # )
            
            continue
        
        # if S is too small:
        if S <= min_slope:
            S = min_slope
            
        roughness = 5.3 * D50

        def discharge_from_depth(H):
    
            A = B * H
            P = B + 2.0 * H
            Rh = A / P
    
            C = 2.5 * np.log(11.0 * H / roughness)
    
            return C * A * np.sqrt(Rh) * np.sqrt(GRAV * S)

        def residual(H):
            return discharge_from_depth(H) - Qi_target
    
        # Lower bound must ensure C > 0
        h_min = roughness / 11.0 * np.exp(1e-6)
        # Check whether the target discharge is reachable
        if residual(h_min) > 0:        
            # print(
            #     f"Warning: target Q below minimum valid Chézy depth "
            #     f"at t={t}, reach={i}. "
            #     f"Qtarget={Qi_target}, "
            #     f"Q(h_min)={discharge_from_depth(h_min)}, "
            #     f"h_min={h_min}"
            # )
            h[i] = 0.0
            v[i] = 0.0
            continue
               
        # Upper braket
        h_max = 1.0    
        # Increase upper bound until the target Q is bracketed
        while residual(h_max) < 0:
            h_max *= 2.0
        try:
            h[i] = brentq(residual, h_min, h_max)
        except Exception as e:
            print(f"brentq failed at t={t}, reach={i}: {e}")
            
        # Compute velocity
        A = B * h[i]
        v[i] = Qi_target / A

    return h, v
    

def choose_flow_depth(reach_data, SedimSys, Q, t, flow_depth):
    if flow_depth == 1:
        [h, v] = h_manning(reach_data, SedimSys, Q, t)

    elif flow_depth == 2:
        [h, v] = h_ferguson(reach_data, SedimSys, Q, t)
        
    elif flow_depth == 3:
        [h, v] = h_chezy(reach_data, SedimSys, Q, t)

    return h, v







