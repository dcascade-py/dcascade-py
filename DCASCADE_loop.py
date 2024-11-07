# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:00:36 2022

This script contains the time-space loop which assess the sediment routing through the network 

This script was adapted from the Matlab version by Marco Tangi 

@author: Elisa Bozzolan
"""

""" Libraries to import """
import numpy as np
import numpy.matlib
import pandas as pd
from tqdm import tqdm 
import copy
import sys
import os

from supporting_classes import ReachData, Cascade

from supporting_functions import D_finder, sortdistance, matrix_compact, change_slope
from supporting_functions import layer_search, tr_cap_deposit
from supporting_functions import cascades_end_time_or_not
from supporting_functions import deposit_from_passing_sediments
from supporting_functions import compute_time_lag

from transport_capacity_computation import tr_cap_function, compute_cascades_velocities

from flow_depth_calc import choose_flow_depth
from slope_reduction import choose_slopeRed
import itertools



np.seterr(divide='ignore', invalid='ignore')
             
""" MAIN FUNCTION SECTION """



def DCASCADE_main(indx_tr_cap, indx_tr_partition, indx_velocity, indx_vel_partition,                   
                  reach_data, network, Q, Qbi_input, Qbi_dep_in, timescale, psi, roundpar, 
                  update_slope, eros_max, save_dep_layer, ts_length,                  
                  indx_flo_depth = 1, indx_slope_red = 1,                  
                  consider_overtaking_sed_in_outputs = True,
                  compare_with_tr_cap = True, time_lag_for_mobilised = True):
    
    """
    Main function of the D-CASCADE software.
    
    INPUT :
    indx_tr_cap         = the index indicating the transport capacity formula
    indx_tr_partition   = the index indicating the type of sediment flux partitioning
    indx_velocity       = the index indicating the method for calculating velocity (see compute_cascades_velocities)
    indx_vel_partition  = the index indicating the type of partitioning in the section used to compute velocity
    reach_data          = nx1 Struct defining the features of the network reaches
    network             = 1x1 struct containing for each node info on upstream and downstream nodes
    Q                   = txn matrix reporting the discharge for each timestep
    Qbi_input           = per each reach and per each timestep is defined an external sediment input of a certain sediment class
    Qbi_dep_in          = sediment material volume available in reaches at the beginning of the simulation
                        (it could be that for the same reach id, there are two strata defined so two rows of the dataframe with the top row is the deepest strata)
    timescale           = length for the time horizion considered
    psi                 = sediment classes considered (from coarse to fine)
    roundpar            = mimimum volume to be considered for mobilization of subcascade
                         (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)
    update_slope        = bool to chose if we change slope trought time or not. If Flase, constant slope. If True, slope changes according to sediment deposit.
    eros_max            = maximum erosion depth per time step [m]
    save_dep_layer      = saving option of the deposit layer for each time step
    ts_length           = the length in seconds of the timestep (60*60*24 for daily timesteps)
    
    OPTIONAL:
    indx_flo_depth      = the index indicating the flow depth formula, default 1 is Manning
    indx_slope_red      = the index indicating the slope reduction formula, default 1 is no reduction
    
    Options for the dcascade algorithme (if all False, we reproduce the version 1)
    consider_overtaking_sed_in_outputs = Bool to activate or not this option (default True)
    compare_with_tr_cap                = Bool to activate or not this option (default True)
    time_lag_for_mobilised             = Bool to activate or not this option (default True)
    
    OUTPUT: 
    data_output      = struct collecting the main aggregated output matrices 
    extended_output  = struct collecting the raw D-CASCADE output datasets
    """
    
    # Constrain on the option of the algorithm:
    if compare_with_tr_cap == True and consider_overtaking_sed_in_outputs == False:
        raise ValueError("You can not use this combination of algorithm options")
    if time_lag_for_mobilised == True and (consider_overtaking_sed_in_outputs == False or compare_with_tr_cap == False):
        raise ValueError("You can not use this combination of algorithm options")
    
    ################### Fixed parameters
    phi = 0.4 # sediment porosity in the maximum active layer
    minvel = 0.0000001
    outlet = network['n_hier'][-1] #outlet reach ID identification
    n_reaches = reach_data.n_reaches
    n_classes = len(psi)
    
    # Initialise slopes 
    min_slope = min(reach_data.slope) # put a minimum value to guarantee movement 
    slope = np.zeros((timescale, n_reaches))
    slope[0,:] = np.maximum(reach_data.slope, min_slope)
    slope[1,:] = np.maximum(reach_data.slope, min_slope) 
    
    # Initialize node elevation (for each reach the matrix reports the fromN elevation)
    # The last column reports the outlet ToNode elevation (last node of the network), 
    # which can never change elevation.
    node_el = np.zeros((timescale, n_reaches + 1))
    node_el[0,:] = np.append(reach_data.el_fn, reach_data.el_tn[outlet])
    node_el[1,:] = np.append(reach_data.el_fn, reach_data.el_tn[outlet])
    node_el[:,-1] = node_el[1,-1]
    
    # Initialise all sediment variables 
    
    # Qbi dep matrix with size size depending on how often we want to save it:
    if save_dep_layer=='never':
        dep_save_number = 1
    if save_dep_layer=='yearly':
        dep_save_number = int(timescale/365)+1 #+1 because we also keep t0.
    if save_dep_layer=='always':
        dep_save_number=timescale
    Qbi_dep = [[np.expand_dims(np.zeros(n_classes + 1, dtype=numpy.float32), axis = 0) for _ in range(n_reaches)] for _ in range(dep_save_number)]
    
    Qbi_tr = [np.zeros((n_reaches, n_reaches, n_classes), dtype=numpy.float32) for _ in range(timescale)] # sediment within the reach AFTER transfer, which also gives the provenance 
    Qbi_mob = [np.zeros((n_reaches, n_reaches, n_classes), dtype=numpy.float32) for _ in range(timescale)] # sediment within the reach BEFORE transfer, which also gives the provenance 
    # Note Qbi_tr and Qbi_mob are 3D matrices, if we add the time as a 4th dimension, we can not look at the matrix in spyder. 
    Fi_r_act = np.empty((timescale, n_reaches, n_classes)) # contains grain size distribution of the active layer
    Fi_r_act[:,0] = np.nan
    Q_out = np.zeros((timescale, n_reaches, n_classes)) # amount of material delivered outside the network in each timestep
    D50_AL = np.zeros((timescale, n_reaches)) # D50 of the active layer in each reach in each timestep
    D50_AL2 = np.zeros((timescale, n_reaches))
    V_sed = np.zeros((timescale, n_reaches, n_classes)) #velocities

    tr_cap_all = np.zeros((timescale, n_reaches, n_classes)) #transport capacity per each sediment class
    tr_cap_sum = np.zeros((timescale, n_reaches)) #total transport capacity 

    Qc_class_all = np.zeros((timescale, n_reaches, n_classes))
    flow_depth = np.zeros((timescale, n_reaches)) 
    
    Delta_V_all = np.zeros((timescale, n_reaches)) # reach mass balance (volumes eroded or deposited)
    Delta_V_class_all = np.zeros((timescale, n_reaches, n_classes))
    
    # In case of constant slope
    if update_slope == False:
        slope[:,:] = slope[0,:]
        node_el[:,: ] = node_el[0,:]
    
    # Initialise sediment deposit in the reaches 

    Qbi_dep_0 = [np.expand_dims(np.zeros(n_classes+1, dtype=numpy.float32), axis = 0) for _ in range(n_reaches)]
    for n in network['n_hier']:  
        # if no inputs are defined, initialize deposit layer with a single cascade with no volume and GSD equal to 0
        q_bin = np.array(Qbi_dep_in[n])
        if not q_bin.any(): #if all zeros 
            Qbi_dep_0[n] = np.hstack((n, np.zeros(n_classes))).reshape(1,-1)
        else:           
            Qbi_dep_0[n] = np.float32(np.hstack((np.ones(q_bin.shape[0]) * n, Qbi_dep_in[n, 0]))).reshape(1,-1)
            Fi_r_act[0,n,:] = np.sum(q_bin, axis=0) / np.sum(q_bin)
            D50_AL[0,n] = D_finder(Fi_r_act[0,n,:], 50, psi)

    Qbi_dep[0] = copy.deepcopy(Qbi_dep_0) #store init condition of dep layer

    # Set maximum volume in meters that can be eroded for each reach, for each time step.
    eros_max_all = np.ones((1, n_reaches)) * eros_max
    eros_max_vol = np.round(eros_max_all * reach_data.wac * reach_data.length, roundpar)

    # Set active layer volume, the one used for calculating the tr_cap in [m3/s]
    # corresponds to the depth that the river can see every second (more like a continuum carpet ...)  
    # defined here as 2.D90 [Parker 2008]
    al_vol_all = np.zeros((timescale, n_reaches)) #store the volumes
    al_depth_all = np.zeros((timescale, n_reaches)) #store also the depths 
    # We take the input D90, or if not provided, the D84:
    if ~np.isnan(reach_data.D90):
        reference_d = reach_data.D90
    else:
        reference_d = reach_data.D84
    for n in network['n_hier']:
        al_depth = np.maximum(2*reference_d[n], 0.01)
        al_vol = al_depth * reach_data.wac[n] * reach_data.length[n]
        al_vol_all[:,n] = np.repeat(al_vol, timescale, axis=0)
        al_depth_all[:,n] = np.repeat(al_depth, timescale, axis=0)

    # start waiting bar    
    for t in tqdm(range(timescale-1)):
        
        # Define flow depth and flow velocity from flow_depth_calc
        h, v = choose_flow_depth(reach_data, slope, Q, t, indx_flo_depth)
        flow_depth[t] = h
        
        # Slope reduction functions
        slope = choose_slopeRed(reach_data, slope, Q, t, h, indx_slope_red)
       
        # deposit layer from previous timestep
        Qbi_dep_old = copy.deepcopy(Qbi_dep_0)
        
        # volumes of sediment passing through a reach in this timestep,
        # ready to go to the next reach in the same time step.
        Qbi_pass = [[] for n in range(n_reaches)]
                
        # loop for all reaches:
        for n in network['n_hier']:  
            
            # TODO : How to include the lateral input ?
                               
            # Extracts the deposit layer left in previous time step          
            V_dep_init = Qbi_dep_old[n] # extract the deposit layer of the reach 
                                       
            ###------Step 1 : Cascades generated from the reaches upstream during 
            # the present time step, are passing the inlet of the reach
            # (stored in Qbi_pass[n]).
            # This step make them pass to the outlet of the reach or stop there,
            # depending if they reach the end of the time step or not.
            
            if consider_overtaking_sed_in_outputs == True:            
                # Store the arriving cascades in the transported matrix (Qbi_tr)
                # Note: we store the volume by original provenance
                for cascade in Qbi_pass[n]:
                    Qbi_tr[t][[cascade.volume[:,0].astype(int)], n, :] += cascade.volume[:, 1:]
                    # DD: If we want to store instead the direct provenance
                    # Qbi_tr[t][cascade.provenance, n, :] += np.sum(cascade.volume[:, 1:], axis = 0)  
            
                
            # Compute the velocity of the cascades in this reach [m/s] 
            if Qbi_pass[n] != []:
                # Define the section height:
                # coef_AL_vel = 0.1
                # hVel = coef_AL_vel * h[n]                
                hVel = al_depth_all[t,n] 
                
                velocities = compute_cascades_velocities(Qbi_pass[n], 
                                           indx_velocity, indx_vel_partition, hVel,
                                           indx_tr_cap, indx_tr_partition,
                                           reach_data.wac[n], slope[t,n], Q[t,n], v[n], h[n],
                                           phi, minvel, psi,
                                           V_dep_init, al_vol_all[0,n],
                                           roundpar)
                # Store velocities
                V_sed[t, n, :] = velocities * ts_length
            
            # Decides weather cascades, or parts of cascades, 
            # finish the time step here or not.
            # After this step, Qbi_pass[n] contains volume that do not finish
            # the time step in this reach
            if Qbi_pass[n] != []:
                Qbi_pass[n], to_be_deposited = cascades_end_time_or_not(Qbi_pass[n], reach_data.length[n], ts_length)
                
            else:
                to_be_deposited = None
            
            # In the case we don't consider overpassing sediments
            # store cascade as transported there for the next time step,
            # only if it has stopped there
            if consider_overtaking_sed_in_outputs == False:    
                if to_be_deposited is not None:
                    Qbi_tr[t+1][[to_be_deposited[:,0].astype(int)], n, :] += to_be_deposited[:, 1:]
                                                    
            ###------Step 2 : Mobilise volumes from the reach material before 
            # considering the passing cascades (if they are passing cascades). 
            # The parameter "time_lag" is the proportion of the time step where this
            # mobilisation occurs, i.e. before the first possible cascade arrives 
            # at the outlet.
            time_lag = compute_time_lag(Qbi_pass[n], n_classes, compare_with_tr_cap, time_lag_for_mobilised)
                                
                           
            # In the case time_lag is not all zeros, we mobilise from the reach
            # before the cascades arrive, proportionally to this time_lag
            mobilized_cascades = []
            if np.any(time_lag != 0):
                # Get the sediment fraction and D50 in the active layer. 
                # DD: to be discussed if we keep the active layer (2D90) or a bigger active layer,
                # since we don't consider any incoming sediment now                          
                _,_,_, Fi_r_act[t,n,:] = layer_search(V_dep_init, al_vol_all[0,n], roundpar)
                D50_AL[t,n] = D_finder(Fi_r_act[t,n,:], 50, psi)           
                # In case the active layer is empty, I use the GSD of the previous timestep
                if np.sum(Fi_r_act[t,n,:]) == 0:
                   Fi_r_act[t,n,:] = Fi_r_act[t-1,n,:] 
                # Transport capacity in m3/s 
                tr_cap_per_s, Qc = tr_cap_function(Fi_r_act[t,n,:] , D50_AL[t,n], slope[t,n], 
                                                   Q[t,n], reach_data.wac[n], v[n], 
                                                   h[n], psi, indx_tr_cap, indx_tr_partition)                                   
                # Store tr_cap (and Qc)
                tr_cap_all[t,n,:] = tr_cap_per_s
                tr_cap_sum[t,n] = np.sum(tr_cap_per_s)
                if indx_tr_cap == 7:
                    Qc_class_all[t,n,:]=Qc   
                    
                # Total volume possibly mobilised during the time lag (per class)
                time_for_mobilising = time_lag * ts_length
                volume_mobilisable = tr_cap_per_s * time_for_mobilising
                
                # Erosion maximum during the time lag 
                # (we take the mean time lag among the classes)
                eros_max_vol_t = eros_max_vol[0,n] * np.mean(time_lag)
                
                # Mobilise from the reach layers
                V_inc_EL, V_dep_EL, V_dep_not_EL, _ = layer_search(V_dep_init, eros_max_vol_t, roundpar)
                [V_mob, V_dep_init] = tr_cap_deposit(V_inc_EL, V_dep_EL, V_dep_not_EL, volume_mobilisable, roundpar)
                                
                if np.any(V_mob[:,1:] != 0): 
                    # The mobilised cascade is added to a temporary container
                    elapsed_time = np.zeros(n_classes) # its elapsed time is 0
                    provenance = n
                    mobilized_cascades.append(Cascade(provenance, elapsed_time, V_mob))
                
                    # Store this volume if we consider only this one
                    if consider_overtaking_sed_in_outputs == False:
                        Qbi_mob[t][[V_mob[:,0].astype(int)], n, :] += V_mob[:, 1:]

            
            # New deposit layer after this step 2.
            V_dep_after_tlag = V_dep_init
                                             
            ###-----Step 3 : Now mobilise material considering that cascades are passing
            # at the outlet. We recalculate the transport capacity by including 
            # now the passing cascades.  
            if np.any(time_lag != 1):
                # Compute the transport capacity considering continuous arriving flux [m3/s]
                Q_pass_volume = np.concatenate([cascade.volume for cascade in Qbi_pass[n]], axis=0)
                Q_pass_volume = matrix_compact(Q_pass_volume)
                Q_pass_volume_per_s = copy.deepcopy(Q_pass_volume)
                # DD: should we put it in per second ? 
                Q_pass_volume_per_s[:,1:] = Q_pass_volume_per_s[:,1:] / ts_length                                            
                _,_,_, Fi_r = layer_search(V_dep_after_tlag, al_vol_all[0,n], roundpar, Qbi_incoming = Q_pass_volume_per_s) 
                D50_AL2[t,n] = float(D_finder(Fi_r, 50, psi))   
                tr_cap_per_s, Qc = tr_cap_function(Fi_r , D50_AL2[t,n], slope[t,n], 
                                                   Q[t,n], reach_data.wac[n], v[n], 
                                                   h[n], psi, indx_tr_cap, indx_tr_partition)                                   
                # Total volume possibly mobilised during (1 - time_lag)
                time_for_mobilising = (1-time_lag) * ts_length
                volume_mobilisable = tr_cap_per_s * time_for_mobilising
                
                # Total volume arriving
                sum_pass = np.sum(Q_pass_volume[:,1:], axis=0)
                              
                # Compare sum of passing cascade to the mobilisable volume (for each sediment class)               
                diff_with_capacity = volume_mobilisable - sum_pass
                                    
                # Sediment classes with positive values in diff_with_capacity are mobilised from the reach n
                diff_pos = np.where(diff_with_capacity < 0, 0, diff_with_capacity)     
                if np.any(diff_pos): 
                    # Erosion maximum during (1 - time_lag)
                    eros_max_vol_t = eros_max_vol[0,n] * (1 - np.mean(time_lag)) #erosion maximum
                    
                    # Mobilise from the reach layers
                    V_inc_EL, V_dep_EL, V_dep_not_EL, _ = layer_search(V_dep_after_tlag, eros_max_vol_t, roundpar)
                    [V_mob, V_dep_after_tlag] = tr_cap_deposit(V_inc_EL, V_dep_EL, V_dep_not_EL, diff_pos, roundpar)
                    
                    if np.any(V_mob[:,1:] != 0): 
                        # The Vmob is added to the temporary container 
                        elapsed_time = time_lag # it start its journey at the end of the time lag
                        provenance = n
                        mobilized_cascades.append(Cascade(provenance, elapsed_time, V_mob))
                                        
                # Sediment classes with negative values in diff_with_capacity
                # are over capacity
                # They are deposited, i.e. directly added to Vdep
                diff_neg = -np.where(diff_with_capacity > 0, 0, diff_with_capacity)     
                if np.any(diff_neg):  
                    Vm_removed, Qbi_pass[n], residual = deposit_from_passing_sediments(np.copy(diff_neg), Qbi_pass[n], roundpar)
                    V_dep_after_tlag = np.concatenate([V_dep_after_tlag, Vm_removed], axis=0) 
                    
                                
            # After this step, Qbi_pass[n] contains the traveling volumes that effectively 
            # reach and pass the outlet of the reach.
            
            # Update the deposit layer
            V_dep_final = V_dep_after_tlag      
            
            
            ###-----Step 4: Finalisation.
            
            # Add the cascades that were mobilised in this reach to Qbi_pass[n]
            if mobilized_cascades != []:
                Qbi_pass[n].extend(mobilized_cascades) 
            
            if consider_overtaking_sed_in_outputs == True:
                # ..and store the total mobilised volumes (passing + mobilised from the reach)
                for cascade in Qbi_pass[n]:
                    Qbi_mob[t][[cascade.volume[:,0].astype(int)], n, :] += cascade.volume[:, 1:]
                    # DD: If we want to store instead the direct provenance
                    # Qbi_mob[t][cascade.provenance, n, :] += np.sum(cascade.volume[:, 1:], axis = 0)
                                                
            # Deposit the stopping cascades in Vdep 
            if to_be_deposited is not None:                   
                to_be_deposited = sortdistance(to_be_deposited, network['upstream_distance_list'][n])
                V_dep_final = np.concatenate([V_dep_final, to_be_deposited], axis=0)
                
            # ..and store Vdep for next time step
            Qbi_dep_0[n] = np.float32(V_dep_final)

            # Finally, pass these cascades to the next reach (if we are not at the outlet)
            # Find reach downstream:   
            if n != int(network['outlet']):
                n_down = np.squeeze(network['downstream_node'][n], axis = 1) 
                n_down = int(n_down) # Note: This is wrong if there is more than 1 reach downstream (to consider later)
            else:
                n_down = None
            
            if n != int(network['outlet']):       
                Qbi_pass[n_down].extend(copy.deepcopy(Qbi_pass[n]))
            # If it is the outlet, we add the cascades to Qout    
            else:
                for cascade in Qbi_pass[n]:
                    Q_out[t, [cascade.volume[:,0].astype(int)], :] += cascade.volume[:,1:]
                
            # Compute the changes in bed elevation
            # Modify bed elevation according to increased deposit
            Delta_V = np.sum(Qbi_dep_0[n][:,1:]) -  np.sum(Qbi_dep_old[n][:,1:])
            # Record Delta_V
            Delta_V_all[t,n] = Delta_V 
            # And Delta_V per class
            Delta_V_class = np.sum(Qbi_dep_0[n][:,1:], axis=0) - np.sum(Qbi_dep_old[n][:,1:], axis=0)
            Delta_V_class_all[t,n,:] = Delta_V_class            
            
            # Update slope, if required.
            if update_slope == True:
                node_el[t+1][n]= node_el[t,n] + Delta_V/( np.sum(reach_data.Wac[np.append(n, network['upstream_node'][n])] * reach_data.length[np.append(n, network['Upstream_Node'][n])]) * (1-phi) )
            
        """End of the reach loop"""
        
        # Save Qbi_dep according to saving frequency
        if save_dep_layer == 'always':
            Qbi_dep[t+1] = copy.deepcopy(Qbi_dep_0)            
        if save_dep_layer == 'yearly':
            if int(t+2) % 365 == 0 and t != 0:
                t_y = int((t+2)/365)
                Qbi_dep[t_y] = copy.deepcopy(Qbi_dep_0)
                            
            
        # in case of changing slope..
        if update_slope == True:
            #..change the slope accordingly to the bed elevation
            slope[t+1,:], node_el[t+1,:] = change_slope(node_el[t+1,:] ,reach_data.length, network, s = min_slope)
                
    """end of the time loop"""                



    # output processing
    # aggregated matrixes
    
    QB_mob_t = [np.sum(x, axis = 2) for x in Qbi_mob[0:timescale-1]] #sum along sediment classes
    Qbi_mob_class = [np.sum(x, axis = 0) for x in Qbi_mob[0:timescale-1]] #sum along sediment classes
    QB_mob = np.rollaxis(np.dstack(QB_mob_t),-1) 
    QB_mob_sum = np.sum(QB_mob, 1) #total sediment mobilized in that reach for that time step (all sediment classes, from all reaches)
    
    #total sediment delivered in each reach (column), divided by reach provenance (row) 
    QB_tr_t = [np.sum(x, axis = 2) for x in Qbi_tr[0:timescale-1]] 
    QB_tr = np.rollaxis(np.dstack(QB_tr_t),-1)
    
    
    V_dep_sum = np.zeros((len(Qbi_dep)-1, n_reaches ))  # EB : last time step would be equal to 0 - delete to avoid confusion 
    V_class_dep = [[np.expand_dims(np.zeros(n_classes+1), axis = 0) for _ in range(n_reaches)] for _ in range(len(Qbi_dep))]
   
    for t in (np.arange(len(Qbi_dep)-1)):
        for n in range(len(Qbi_dep[t])): 
            q_t = Qbi_dep[t][n] 
            #total material in the deposit layer 
            V_dep_sum[t,n] = np.sum(q_t[:,1:])
            # total volume in the deposit layer for each timestep, divided by sed.class 
            V_class_dep[t][n] = np.sum(q_t[:,1:], axis = 0) 
            
    #--Total material in a reach in each timestep (both in the deposit layer and mobilized layer)                       
    if save_dep_layer=='always':           
        tot_sed = V_dep_sum + np.sum(QB_tr, axis = 1) 
    else:
        tot_sed= []
        
    #--Total material transported 
    tot_tranported = np.sum(QB_tr, axis = 1) 
    
    
    #total material in a reach in each timestep, divided by class 
    tot_sed_temp = []
    Qbi_dep_class = []
    # D50_tot = np.zeros((timescale-1, n_reaches))
 
    for t in np.arange(len(Qbi_dep)-1):
        v_dep_t = np.vstack(V_class_dep[t])
        # tot_sed_temp.append(Qbi_mob_class[t] + v_dep_t)
        Qbi_dep_class.append(v_dep_t)
        # Fi_tot_t = tot_sed_temp[t]/ (np.sum(tot_sed_temp[t],axis = 1).reshape(-1,1))
        # Fi_tot_t[np.isnan(Fi_tot_t)] = 0
        # for i in np.arange(n_reaches):
        #     D50_tot[t,i] = D_finder(Fi_tot_t[i,:], 50, psi)
    
    #--D50 of mobilised volume 
    D50_mob = np.zeros((timescale-1, n_reaches))
 
    for t in np.arange(len(Qbi_mob_class)):
        Fi_mob_t = Qbi_mob_class[t]/ (np.sum(Qbi_mob_class[t],axis = 1).reshape(-1,1))
        Fi_mob_t[np.isnan(Fi_mob_t)] = 0
        for i in np.arange(n_reaches):
            D50_mob[t,i] = D_finder(Fi_mob_t[i,:], 50, psi)
            
    #--D50 of deposited volume 
    dep_sed_temp = []
    D50_dep = np.zeros((timescale-1, n_reaches))
    
    # stack the deposited volume 
    dep_sed_temp = []
    D50_dep = np.zeros((timescale-1, n_reaches))
    
    for t in np.arange(len(Qbi_dep_class)):
        Fi_dep_t = Qbi_dep_class[t]/ (np.sum(Qbi_dep_class[t],axis = 1).reshape(-1,1))
        Fi_dep_t[np.isnan(Fi_dep_t)] = 0
        for i in np.arange(n_reaches):
            D50_dep[t,i] = D_finder(Fi_dep_t[i,:], 50, psi)
            
            
    #--Total material in a reach in each timestep, divided by class (transported + dep)
    tot_sed_class =  [np.empty((len(Qbi_dep), n_reaches)) for _ in range(n_classes)]
    q_d = np.zeros((1, n_reaches))
    
    for c in range(n_classes): 
        for t in range(len(Qbi_dep)): 
            q_t = Qbi_dep[t] # get the time step
            for i, reaches in enumerate(q_t): # get the elements of that class per reach 
                q_d[0,i] = np.sum(reaches[:,c+1])
            q_tt = Qbi_tr[t][:,:,c]
            tot_sed_class[c][t,:] = q_d + np.sum(q_tt, axis = 0)
            
    #--Deposited per class         
    deposited_class =  [np.empty((len(Qbi_dep), n_reaches)) for _ in range(n_classes)]

    for c in range(n_classes): 
        for t in range(len(Qbi_dep)): 
            q_t = Qbi_dep[t]
            deposited_class[c][t,:] = np.array([np.sum(item[:,c+1], axis = 0) for item in q_t]) 
   
    
    #--Mobilised per class
    mobilised_class =  [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    
    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_m = Qbi_mob[t][:,:,c]
            mobilised_class[c][t,:] = np.sum(q_m, axis = 0)

    #--Transported per class        
    transported_class =  [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    
    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_m = Qbi_tr[t][:,:,c]
            transported_class[c][t,:] = np.sum(q_m, axis = 0)
                        
    #--Tranport capacity per class (put in same format as mob and trans per class)
    tr_cap_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_m = tr_cap_all[t,:,c]
            tr_cap_class[c][t,:] = q_m     
    
    #--Critical discharge per class (put in same format as mob and trans per class)
    if indx_tr_cap == 7:   
        Qc_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
        for c in range(n_classes): 
            for t in range(timescale-1): 
                q_m = Qc_class_all[t,:,c]
                Qc_class[c][t,:] = q_m  
            
    Q_out_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_m = Q_out[t,:,c]
            Q_out_class[c][t,:] = q_m 
    
    
    V_sed_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    for t in range(timescale-1):
        for c in range(n_classes):
            q_m = V_sed[t,:,c]
            V_sed_class[c][t, :] = q_m
        
    #--Total sediment volume leaving the network
    outcum_tot = np.array([np.sum(x) for x in Q_out])
    df = pd.DataFrame(outcum_tot)
    
    #set all NaN transport capacity to 0
    tr_cap_sum[np.isnan(tr_cap_sum)] = 0 
    
    #set all NaN active layer D50 to 0; 
    D50_AL[np.isnan(D50_AL)] = 0
    D50_AL2[np.isnan(D50_AL2)] = 0
    
    Q = np.array(Q)
    
    #--Output struct definition 
    #data_plot contains the most important D_CASCADE outputs 
    data_output = {'Channel Width [m]': np.repeat(np.array(reach_data.wac).reshape(1,-1),len(Qbi_dep), axis = 0), 
                   'Reach slope': slope,
                   'Discharge [m^3/s]': Q[0:timescale,:],
                   'Mobilized [m^3]': QB_mob_sum,
                   'Transported [m^3]': tot_tranported,
                   'Transported + deposited [m^3]': tot_sed,
                   'D50 deposit layer [m]': D50_dep,
                   'D50 mobilised layer [m]': D50_mob,
                   'D50 active layer before time lag[m]': D50_AL,
                   'D50 active layer [m]': D50_AL2,
                   'Transport capacity [m^3]': tr_cap_sum,
                   'Deposit layer [m^3]': V_dep_sum,
                   'Delta deposit layer [m^3]': Delta_V_all,
                   'Transported + deposited - per class [m^3]': tot_sed_class,
                   'Deposited - per class [m^3]': deposited_class,
                   'Mobilised - per class [m^3]': mobilised_class,
                   'Transported- per class [m^3]': transported_class,
                   'Delta deposit layer - per class [m^3]': Delta_V_class,
                   'Transport capacity - per class [m^3]': tr_cap_class,
                   'Sed_velocity [m/day]': V_sed,
                   'Sed_velocity - per class [m/day]': V_sed_class,
                   'Flow depth': flow_depth,
                   'Active layer [m]': al_depth_all,
                   'Maximum erosion layer [m]': eros_max_all,
                   'Q_out [m^3]': Q_out,
                   'Q_out_class [m^3]': Q_out_class,
                   'Q_out_tot [m^3]': outcum_tot
                   }

    if indx_tr_cap == 7:
        data_output["Qc - per class"] = Qc_class
         
    #all other outputs are included in the extended_output cell variable 
    extended_output = {'Qbi_tr': Qbi_tr,
                       'Qbi_mob': Qbi_mob,
                       'Q_out': Q_out,
                       'Qbi_dep': Qbi_dep,
                       'Fi_r_ac': Fi_r_act,
                       'node_el': node_el
                       }
    
    return data_output, extended_output


















