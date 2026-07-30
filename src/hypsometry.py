"""
Created on Thu Jun 13 10:05:39 2024

@author: Justin Rogers, Diane Doolaeghe
"""

"""
Hypsometric calculations related functions.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import copy

from constants import GRAV
from d_finder import D_finder
from transport_capacity_calculator import TransportCapacityCalculator




def initialise_hypso_data(reach_data, CS_curves, dx = 5, dz = 0.1):
    '''
    Function to compute hypsometric curve and function and attach it to reach_data
    '''
    
    if CS_curves == None:
        raise ValueError('You did not provide cross section data for this hyspo_code option')    
    
    import matplotlib.pyplot as plt
            
    
    h_max = max(np.max(section[1, :] - np.min(section[1, :]))
                for sections in CS_curves.values() for section in sections.values())       
    z_vec = np.arange(0, h_max + dz, dz)        # elevation vector common to all reaches

    # Loop through each reach for which we have a CS
    for fn in CS_curves.keys():
        widths_all_CS = []
        
        # Temp: for plot         
        # fig, ax = plt.subplots(figsize=(7, 4))
        
        # Loop for each CS available for this reach
        for CS in CS_curves[fn].values():
            
            # fig, ax = plt.subplots(figsize=(7, 4))
            
            x_sec = CS[0, :]
            z_sec = CS[1, :]   
        
            Q_steps = np.array([0, 4000]) # validity range for hypsometric profile(s) (DD: why?)
    
            h_xs = z_sec - np.min(z_sec) # elevation from CS lowest point
        
            x_dense = np.linspace(x_sec.min(), x_sec.max(), 2000)
            h_dense = np.interp(x_dense, x_sec, h_xs)
            
            # Width vector associated with each water height
            widths = []
            for z in z_vec:
                mask = h_dense <= z        
                if np.any(mask):
                    width = np.ptp(x_dense[mask])
                else:
                    width = 0.0        
                widths.append(width)        
            
            widths_all_CS.append(np.array(widths))
                                    
            # # Temp for plot
            # ax.plot(x_dense - x_dense.min(), h_dense, "-", color = 'gray', linewidth = 1)
            # ax.plot(widths, z_vec, "-")
            
            # plt.show()
            
        # Compute avg width for each water height (DD: is it a correct method?)
        widths_all_CS = np.array(widths_all_CS)
        widths_mean = np.mean(widths_all_CS, axis = 0)
                
        # Temp for plot
        # ax.plot(widths_mean, z_vec, ".-", color = 'black', label = "hypsometric width curve - avg")
        
        # ax.set_xlabel("x or wetted width [m]")
        # ax.set_ylabel("height above thalweg [m]")
        # ax.grid(True, alpha=0.3)
        # ax.legend()
        # plt.tight_layout()
        # plt.show()

        # Force monotonic, unique width values for inverse interpolation width -> height.
        widths_mean = np.maximum.accumulate(widths_mean)
        widths_mean = widths_mean + np.arange(widths_mean.size) * 1e-9
        widths_mean[0] = 0.0
        w_vec_q = np.tile(widths_mean[:, None], (1, len(Q_steps)))

        # Store info in reach data class
        n = fn - 1
        reach_data.hypsometric_data[n] = {
            'Zvec': z_vec,
            'Wvec_q': w_vec_q,  # DD: why we duplicate the info for two discharge ?
            'Hvec': z_vec-z_vec.min(),
            'Qsteps': Q_steps,
            'hypsoDX': dx,
        }
        
    # Second part of hypso initialization: create a function
                  
    # Initialize width vector common to all reaches
    max_w = max(np.max(reach_data.hypsometric_data[n]['Wvec_q'])
        for n in reach_data.hypsometric_data)        
    max_w = np.ceil(max_w / dx) * dx # Round to upper     
    w_vec = np.arange(0, max_w + dx, dx)   # see how to store this common Xgrid  
           
    for n in reach_data.hypsometric_data.keys(): 
        
        w_vec_q = reach_data.hypsometric_data[n]['Wvec_q']
        nqsteps = w_vec_q.shape[1]        
        h_vec = reach_data.hypsometric_data[n]['Hvec']
    
        # Base case: static hypsometry. 
        # (JR developped a more advanced case, for having hypsometry depending on Q, that DD removed for now)
        if nqsteps <= 2:
            width_to_height = interp1d(w_vec_q[:, 0], h_vec, bounds_error=False, fill_value=10)  
            reach_data.hypsometric_data[n]['Xgrid'] = w_vec
            reach_data.hypsometric_data[n]['heights_at_Xgrid'] = width_to_height(w_vec)
    
            def q_to_H_static(pts, func=width_to_height, Xgrid = w_vec):
                # Called downstream as q_to_H_static((Q_value, Xgrid)).
                # Q is ignored because geometry is not discharge-dependent here.
                # DD: it will do width_to_height(w_wec) in our case
                if isinstance(pts, tuple):
                    x = pts[1]
                else:
                    x = Xgrid
                return func(x)
           
        reach_data.hypsometric_data[n]['q_to_H_interp_func'] = q_to_H_static
    


def update_hypsometric_hydraulics(reach_data, SedimSys, Q, t,  indx_flo_depth):
    """
    Update hydraulics for all hypsometric reaches at timestep t.
    Will update h, v, and width for these reaches. etasave ?
    Returns also hypsometric water height for tr_cap calculation (if hypso_code == 2)
    """
    # Dictionnary to store hypsometric water heights per reach
    hypso_hw = {}
    
    for n in reach_data.hypsometric_data.keys():
        hypso_hw[n] = {}
        
        # Retrive hypso data from reach
        hypso_n = reach_data.hypsometric_data[n]    
        dx = hypso_n['hypsoDX']
        x_vec = hypso_n['Xgrid']
        
        # Compute water heigths corresponding to x_vec
        z_vec = hypso_n['q_to_H_interp_func']((Q[t, n], x_vec))
        z_vec = np.nan_to_num(z_vec, posinf=10, neginf=10)
        
        # Define function of Q = f(Hw), and solve it to get Hw (eta)
        def func(Hw):
            if indx_flo_depth == 1:
                Q_, b_, JS = hypso_manning_Q(
                    Hw, z_vec, dx, reach_data.n[n], SedimSys.slope[t, n]
                )
            elif indx_flo_depth == 2:
                Q_, b_, JS = hypso_ferguson_Q(
                    Hw, z_vec, dx,
                    reach_data.C84_fac[n] * reach_data.D84[n],
                    SedimSys.slope[t, n]
                ) # check C84_fac                
            elif indx_flo_depth == 3:
                Q_, b_, JS = hypso_chezy_Q(
                    Hw, z_vec, dx,
                    SedimSys.slope[t, n], reach_data.D50[n]
                )
            else:
                raise ValueError(f"Unsupported indx_flo_depth for hypsometry: {indx_flo_depth}")
    
            return (Q_ / Q[t, n]) - 1
                
        try:
            result = root_scalar(func, bracket=[1e-4, 50.0], method='brentq')
            if result.converged:
                eta = result.root #(solved water height)
                hypso_n['eta_save'] = eta
            else:
                print(f"Root solver did not converge at t={t}, reach={n}")
                eta = max(hypso_n['eta_save'], 1e-4)
        except Exception as e:
            print(f"root_scalar failed at t={t}, reach={n}: {e}")
            eta = max(hypso_n['eta_save'], 1e-4)
        
        # Re-compute reach hydraulics using eta (= solved Hw)
        if indx_flo_depth == 1:
            Q_, b_, JS = hypso_manning_Q(
                eta, z_vec, dx, reach_data.n[n], SedimSys.slope[t, n],
            )
        elif indx_flo_depth == 2:
            Q_, b_, JS = hypso_ferguson_Q(
                eta, z_vec, dx, reach_data.C84_fac[n] * reach_data.D84[n],
                SedimSys.slope[t, n]
            )
            
        elif indx_flo_depth == 3:
            Q_, b_, JS = hypso_chezy_Q(
                eta, z_vec, dx,
                SedimSys.slope[t, n], reach_data.D50[n]
            )
            
        # Save hypsometric results of this time step:
        hypso_hw[n]['v_save'] = JS['Vsave'] # v(x)
        hypso_hw[n]['h_save'] = JS['Hsave'] # h(x)
        
        # Save the means:
        Vsave_trimmed = np.trim_zeros(JS['Vsave'], 'b')
        Hsave_trimmed = np.trim_zeros(JS['Hsave'], 'b')
        hypso_hw[n]['h_mean'] = Hsave_trimmed.mean()
        hypso_hw[n]['v_mean'] = Vsave_trimmed.mean()
        
        # Save the wetted width
        hypso_hw[n]['width'] = b_
        
        # Save the width discretisation for latter
        hypso_hw[n]['Xgrid'] = hypso_n['Xgrid']
        
        # Here we directly update width and water height for the chosen reaches
        SedimSys.width[t, n] = b_
        SedimSys.flow_depth[t, n] = hypso_hw[n]['h_mean'] #DD: is hmean appropriate ?
        SedimSys.water_velocity[t, n] = hypso_hw[n]['v_mean']
        
        # To keep eta_save (DD: why?)
        reach_data.hypsometric_data[n] = hypso_n 

    return hypso_hw




def hypso_manning_Q(H, Hsec, dy, n, slope):
    # Return a Q based on given water level H, and elevation section Hsec
    # Adapted from Marco Redolfi solver by JR
    # Return: Q total, wetted width, and hypso informations (JR)
    
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
        
        if hl > 0 and hr > 0: # if both points are under water
            # Trapezoidal slice
            hm = (hl + hr) / 2
            Bi = np.sqrt(dy**2 + (hl - hr)**2)
            Ai = hm * dy
            Rhi = Ai / Bi
            C = hm**(1/6) / n
            Qi = C * Ai * np.sqrt(Rhi) * np.sqrt(slope)
            
            bsec += dy
            Qsec += Qi
            
            # Save heights, velocities, and chezy coef
            Hsave[j_point] = hm
            Vsave[j_point] = Qi / Ai
            Csave[j_point] = C
            
        elif hl > 0 or hr > 0: # if only one of the two points is under water
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
            
            # Save heights, velocities, and chezy coef
            Hsave[j_point] = hmax
            Vsave[j_point] = Qi / Ai
            Csave[j_point] = C
       
    JS = {'Hsave': Hsave, 'Vsave': Vsave, 'Csave': Csave}
    
    return Qsec, bsec, JS


def hypso_chezy_Q(H, Hsec, dy, slope, d50):
    # Return a Q based on given water level H, and elevation section Hsec
    # Adapted from Marco Redolfi solver by JR
    # Return: Q total, wetted width, and hypso informations (JR)
    # DD: I added this to be compared with Manning results. 
    
    Npoints = len(Hsec)
    e = 5.3*d50 # roughness
    
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
            C = 2.5*np.log(11*hm/e)
            Qi = C * Ai * np.sqrt(Rhi) * np.sqrt(GRAV*slope) #DD: we need a g
            
            bsec += dy
            Qsec += Qi
            
            # Save heights, velocities, and chezy coef
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
            C = 2.5*np.log(11*hmax/e)
            Qi = C * Ai * np.sqrt(Rhi) * np.sqrt(GRAV*slope) #why g ? 
            
            
            bsec += bi
            Qsec += Qi
            
            # Save heights, velocities, and chezy coef
            Hsave[j_point] = hmax
            Vsave[j_point] = Qi / Ai
            Csave[j_point] = C
       
    JS = {'Hsave': Hsave, 'Vsave': Vsave, 'Csave': Csave}#
    return Qsec, bsec, JS


def hypso_ferguson_Q(H, Hsec, dy, D84, slope):
    """vectorized, roughness as per Ferguson.
    DD: to be checked
    """
  
    #sqrt_g_slope = np.sqrt(GRAV * slope)
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
    V_wet = s8f * np.sqrt(GRAV * h_wet * slope)
    Q_wet = V_wet * A_wet  # Velocity * Area

    # Sum total discharge and wetted width
    return np.sum(Q_wet), np.sum(b_wet), {'Hsave': h_wet, 'Vsave': V_wet}



def hypso_transport_capacity(Vdep, roundpar, t, n, Q, 
                             hypso_hw, #Xwac, vsave, hsave,
                             indx_tr_cap, indx_tr_partition,
                             SedimSys,
                             passing_cascades = None):

    """
    Compute hypsometric transport capacity by lateral wet slice.
    First lines are same as in 1D transport capacity calculation.
    (i.e. preparing passing through volume and active layer GSD)    
    Hypsometry appear only when calculating the transport capacity.
    """   
    
    #--- Concatenate passing cascades into one volume (if they are)
    if passing_cascades == None or passing_cascades == []:
        passing_volume = None

    else:
        # Particular case where external cascades are passed to the next reach and excluded of the calculation
        if SedimSys.force_pass_external_inputs == True:
            passing_cascades = [cascade for cascade in passing_cascades
                                  if not (cascade.is_external == True and cascade.provenance == n)]
        if passing_cascades == []: 
            passing_volume = None
        else:
            # Makes a single volume out of the passing cascade list:
            passing_volume = np.concatenate([cascade.volume for cascade in passing_cascades], axis=0)
            passing_volume = SedimSys.matrix_compact(passing_volume) #compact by original provenance    
    
    #--- Compute fraction and D50 in the active layer
    # TODO: warning when the AL is very small, we can have Fi_r is 0 due to roundpar
    
    # Because passing through cascade volume are physically 
    # transported on a different width than the one over which AL volume and Vdep where defined, 
    # we adjust temporarilly these two volumes to the new width (to get coherence in the layer depths):
    W_new = SedimSys.width[t, n]
    W_init = SedimSys.reach_data.wac[n]
    al_vol_ = SedimSys.al_vol[t, n] * (W_new / W_init)
    Vdep_ = copy.deepcopy(Vdep) # I dont want to modify the reach Vdep
    SedimSys.sediments(Vdep_)[:] = SedimSys.sediments(Vdep_) * (W_new / W_init)

    if passing_volume is None:
        AL_volume = al_vol_
    else:
        if SedimSys.al_depth_method == 1:
            # Method 1: (default) if there are passing cascades, their total volume is added to the user-defined active volume
            sum_pass = np.sum(SedimSys.sediments(passing_volume))
            AL_volume = al_vol_ + sum_pass
        elif SedimSys.al_depth_method == 2:
            # Method 2: the active depth is measured from the top of the passing cascades
            AL_volume = al_vol_

    _,_,_, Fi_al_ = SedimSys.layer_search(Vdep_, AL_volume, Qpass_volume = passing_volume, roundpar = roundpar)


    # In case the active layer is empty, I use the GSD of the previous timestep
    if np.sum(Fi_al_) == 0:
       Fi_al_ = SedimSys.Fi_al[t-1, n, :]
    D50_al_ = float(D_finder(Fi_al_, 50, SedimSys.psi))
    
    #--- Compute hypsometric transport capacity
    
    # Retrieve width discretisation vector, hyspometric height, and velocities
    w_vec = hypso_hw[n]['Xgrid'] 
    hsave = hypso_hw[n]['h_save']
    vsave = hypso_hw[n]['v_save']
    
    hypso_tr_cap_per_s = np.zeros((len(hsave), len(SedimSys.psi)))
    
    # Total discharge
    Qtot = 0
    
    # Loop over lateral slices i:
    for i in range(len(hsave)):
        
        h_i = hsave[i]
        if h_i == 0:
            continue # DD: not ideal, see later
        v_i = vsave[i]
        w_i = w_vec[i + 1] - w_vec[i]
        Q_i = w_i * h_i * v_i        
        Qtot += Q_i
            
        # Transport capacity in m3/s
        calculator = TransportCapacityCalculator(Fi_al_ , D50_al_, SedimSys.slope[t,n],
                                               Q_i, w_i, v_i, h_i,
                                               SedimSys.psi, SedimSys.reach_data.roughness[n])
        
        tr_cap_per_s_i, Qc = calculator.tr_cap_function(indx_tr_cap, indx_tr_partition)
        # NB: Qc does not change lateraly here (homogeneous bed)
        
        hypso_tr_cap_per_s[i, :] = tr_cap_per_s_i
        
    # Check that sum Q is not to different from input Q
    rel_err = abs(Q[t, n] - Qtot) / abs(Qtot) * 100
    if rel_err > 5.0:
        raise ValueError(f"Flow mismatch at t={t}, n={n}: "
                         f"Q={Q[t, n]:.6g}, Qtot={Qtot:.6g}, "
                         f"relative error={rel_err:.2f}%")
    
    # Compute sum of hypso tr_cap
    tr_cap_per_s = np.sum(hypso_tr_cap_per_s, axis = 0)  
    
    return tr_cap_per_s, Fi_al_, D50_al_, Qc, hypso_tr_cap_per_s

    
    


################### To keep from JR


# def hypso_transport_capacity(Vdep, roundpar, t, n, Q, 
#                              hypso_hw, #Xwac, vsave, hsave,
#                              indx_tr_cap, indx_tr_partition,
#                              SedimSys,
#                              passing_cascades = None):

#     """
#     Compute hypsometric transport capacity by lateral wet slice.

#     Current simplification:
#     - passing_cascades are ignored in hypso reaches.
#     - bed material is sliced from thalweg outward using layer_search().
#     - if a slice consumes essentially all remaining deposit, we take it directly
#       to avoid tiny negative dry remainders from layer_search roundoff.
#     - true negative sediment volumes are treated as an error.
#     - tiny numerical negatives are snapped to zero, with mass balance checked.
#     """
    
#     # Retrieve width discretisation vector
#     w_vec = hypso_hw[n]['Xgrid']
    
#     # Retrieve hyspometric height and velocities
#     hsave = hypso_hw[n]['h_save']
#     vsave = hypso_hw[n]['v_save']
    

#     mbdebug = False

#     # Explicitly ignore passing cascades in hypso reaches for now.
#     passing_volume = None

#     subQ = np.zeros(len(vsave))
#     h_tr_cap_per_s = np.zeros((len(vsave), len(SedimSys.psi)))
#     Vdep_slices = []

#     Vdep_from_thalweg = copy.deepcopy(Vdep)
#     Vdep_wet = np.zeros((1, 1 + len(SedimSys.psi)), dtype=np.float64)

#     Qc = np.full(len(SedimSys.psi), np.nan)

#     for w in range(len(vsave)):

#         dX = w_vec[w + 1] - w_vec[w]
#         subQ[w] = dX * vsave[w] * hsave[w]

#         slicevol = SedimSys.reach_data.deposit[n] * SedimSys.reach_data.length[n] * dX
#         vol_before = np.sum(SedimSys.sediments(Vdep_from_thalweg))

#         # If this slice consumes all remaining bed material, avoid layer_search()
#         # roundoff creating a tiny negative dry remainder.
#         if slicevol >= vol_before * (1 - 1e-7):
#             Vdep_slice = copy.deepcopy(Vdep_from_thalweg)
#             Vdep_remaining = SedimSys.create_volume(provenance=n)

#             sed_sum = np.sum(SedimSys.sediments(Vdep_slice), axis=0)
#             if np.sum(sed_sum) > 0:
#                 Fi_al_ = sed_sum / np.sum(sed_sum)
#             else:
#                 Fi_al_ = np.zeros(len(SedimSys.psi))

#         else:
#             _, Vdep_slice, Vdep_remaining, Fi_al_ = SedimSys.layer_search(
#                 Vdep_from_thalweg,
#                 np.float32(slicevol),
#                 Qpass_volume=None,
#                 roundpar=roundpar
#             )

#         # ------------------------------------------------------------
#         # Safety check: split should conserve deposit and not create
#         # true negative sediment. Tolerance is relative to remaining bed.
#         # ------------------------------------------------------------
#         neg_tol = 1e-6 #max(1e-6, 1e-7 * max(vol_before, 1.0))

#         sed_slice = SedimSys.sediments(Vdep_slice)
#         sed_remain = SedimSys.sediments(Vdep_remaining)

#         min_slice = np.min(sed_slice) if sed_slice.size else 0.0
#         min_remain = np.min(sed_remain) if sed_remain.size else 0.0

#         if min_slice < -neg_tol or min_remain < -neg_tol:
#             raise ValueError(
#                 f"Negative sediment after hypso split at t={t}, n={n}, w={w}. "
#                 f"min_slice={min_slice:.6g}, min_remaining={min_remain:.6g}, "
#                 f"slicevol={slicevol:.6g}, vol_before={vol_before:.6g}, "
#                 f"neg_tol={neg_tol:.6g}"
#             )

#         # Clean signed zero and tiny floating-point sediment noise.
#         # True negatives already raised above.
#         #zero_tol = min(1e-6, max(1e-12, 1e-12 * max(vol_before, 1.0)))
#         zero_tol = 1e-9
        
#         sed_slice[np.abs(sed_slice) <= zero_tol] = 0.0
#         sed_remain[np.abs(sed_remain) <= zero_tol] = 0.0

#         vol_after = (
#             np.sum(SedimSys.sediments(Vdep_slice)) +
#             np.sum(SedimSys.sediments(Vdep_remaining))
#         )

#         mb_tol = max(1e-4, 1e-7 * max(vol_before, 1.0))
#         if abs(vol_after - vol_before) > mb_tol:
#             raise ValueError(
#                 f"Mass balance error in hypso split at t={t}, n={n}, w={w}. "
#                 f"before={vol_before:.6g}, after={vol_after:.6g}, "
#                 f"diff={vol_after - vol_before:.6g}, mb_tol={mb_tol:.6g}"
#             )

#         Vdep_slices.append(copy.deepcopy(Vdep_slice))
#         Vdep_wet = Vdep_wet + Vdep_slice[:, :].sum(0)

#         # Skip tiny/dry/narrow slices, but keep a smooth fallback from prior slice.
#         if slicevol < 10**roundpar or hsave[w] < 0.1 or dX < 1:
#             if w == 0 or subQ[w - 1] <= 0 or not np.isfinite(subQ[w - 1]):
#                 h_tr_cap_per_s[w, :] = 0.0
#             else:
#                 dQ_ratio = subQ[w] / subQ[w - 1]
#                 h_tr_cap_per_s[w, :] = dQ_ratio * h_tr_cap_per_s[w - 1, :]

#             Vdep_from_thalweg = Vdep_remaining
#             continue

#         if np.sum(Fi_al_) <= 0 or not np.all(np.isfinite(Fi_al_)):
#             h_tr_cap_per_s[w, :] = 0.0
#             Vdep_from_thalweg = Vdep_remaining
#             continue

#         D50_al_ = float(D_finder(Fi_al_, 50, SedimSys.psi))

#         calculator = TransportCapacityCalculator(
#             Fi_al_, D50_al_, SedimSys.slope[t, n], subQ[w],
#             dX, vsave[w], hsave[w], SedimSys.psi, SedimSys.reach_data.roughness[n]) #self.SUSP_MULT


#         h_tr_cap_per_s[w, :], Qc = calculator.tr_cap_function(
#             indx_tr_cap, indx_tr_partition
#         )

#         # True negative capacity is an error. Negative zero / tiny roundoff is fine.
#         cap_neg_tol = 1e-12
#         if np.any(h_tr_cap_per_s[w, :] < -cap_neg_tol):
#             raise ValueError(
#                 f"Negative hypso transport capacity at t={t}, n={n}, w={w}. "
#                 f"min={np.min(h_tr_cap_per_s[w, :]):.6g}"
#             )

#         h_tr_cap_per_s[w, np.abs(h_tr_cap_per_s[w, :]) < cap_neg_tol] = 0.0

#         if np.any(np.isnan(h_tr_cap_per_s[w, :])):
#             print("NaN here", Fi_al_)
#             print(w, SedimSys.slope[t, n], subQ[w], dX, vsave[w], hsave[w])

#             if w == 0 or subQ[w - 1] <= 0 or not np.isfinite(subQ[w - 1]):
#                 h_tr_cap_per_s[w, :] = 0.0
#             else:
#                 dQ_ratio = subQ[w] / subQ[w - 1]
#                 h_tr_cap_per_s[w, :] = dQ_ratio * h_tr_cap_per_s[w - 1, :]

#         Vdep_from_thalweg = Vdep_remaining

#     # Final dry remainder
#     Vdep_slices.append(copy.deepcopy(Vdep_remaining))

#     if mbdebug:
#         vol_slices = sum(np.sum(SedimSys.sediments(vs)) for vs in Vdep_slices)
#         vol_total = np.sum(SedimSys.sediments(Vdep))
#         print(
#             f"[MB hypso split] t={t} n={n} "
#             f"split/total={vol_slices / max(vol_total, 1e-12):.6f}"
#         )

#     total_h_tr_cap_per_s = np.sum(h_tr_cap_per_s, axis=0)

#     Vdep_wet[0, 0] = n
#     total_sum = np.sum(Vdep_wet[:, 1:])
#     sum_per_class = np.sum(Vdep_wet[:, 1:], axis=0)

#     if total_sum > 0:
#         Fi_wet = sum_per_class / total_sum
#         D50_wet_ = float(D_finder(Fi_wet, 50, SedimSys.psi))
#     else:
#         Fi_wet = np.zeros(len(SedimSys.psi))
#         D50_wet_ = np.nan

#     return total_h_tr_cap_per_s, Fi_wet, D50_wet_, Qc, h_tr_cap_per_s, Vdep_slices


