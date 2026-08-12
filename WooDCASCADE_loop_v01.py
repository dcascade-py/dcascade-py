# -*- coding: utf-8 -*-
"""
Created on 13 Aug 2026

This script contains the time-space loop which assess the sediment routing through the network 

This script was adapted from the Matlab version by Marco Tangi 

@author: Elisa Bozzolan
@author for wood updates: Bryce Finch
"""

""" Libraries to import """
import numpy as np
import numpy.matlib
import pandas as pd
from tqdm import tqdm 
import copy
import sys
import os

from supporting_functions import D_finder
from supporting_functions import sortdistance
from supporting_functions import layer_search
from supporting_functions import tr_cap_deposit
from supporting_functions import matrix_compact 
from supporting_functions import sed_transfer_simple
from supporting_functions import change_slope

from supporting_wood_functions_v01 import Global_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb
from supporting_wood_functions_v01 import HiLo_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb
from supporting_wood_functions_v01 import WoodClass_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb
from supporting_wood_functions_v01 import HiLo_WoodClass_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb

from transport_capacity_computation import tr_cap_function
from transport_capacity_computation import sed_velocity
from transport_capacity_computation import sed_velocity_OLD
from flow_depth_calc import choose_flow_depth
from slope_reduction import choose_slopeRed

np.seterr(divide='ignore', invalid='ignore')
             
""" MAIN FUNCTION SECTION """



def DCASCADE_main(ReachData, Network, Qbi_dep_in, timescale, roundpar, 
                  update_slope, save_dep_layer, n_classes):
    """INPUT :
    indx_tr_cap    = the index indicating the transport capacity formula
    indx_partition = the index indicating the type of sediment flux partitioning
    indx_flo_depth = the index indicating the flow depth formula
    indx_slope_red = the index indicating the slope reduction formula
    ReachData      = nx1 Struct defining the features of the network reaches
    Network        = 1x1 struct containing for each node info on upstream and downstream nodes
    Q              = txn matrix reporting the discharge for each timestep
    Qbi_input      = per each reach and per each timestep is defined an external sediment input of a certain sediment class
    Qbi_dep_in     = deposit of a sediment material known to be at a certain reach
                     (it could be that for the same reach id, there are two strata defined so two rows of the dataframe with the top row is the deepest strata)
    timescale      = length for the time horizion considered
    psi            = sediment classes considered (from coarse to fine)
    roundpar       = mimimum volume to be considered for mobilization of subcascade
                     (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)
    update_slope   = bool to chose if we change slope trought time or not. If Flase, constant slope. If True, slope changes according to sediment deposit.
    eros_max       = maximum erosion depth per time step [m]
    save_dep_layer = saves the deposit layer for each time step
    ts_length      = the length in seconds of the timestep (60*60*24 for daily timesteps)
    
    OUTPUT: 
    data_output      = struct collecting the main aggregated output matrices 
    extended_output  = struct collecting the raw D-CASCADE output datasets"""
         

    ################### Fixed parameters
    # phi = 0.4 # sediment porosity in the maximum active layer
    # minvel = 0.0000001
    outlet = Network['NH'][-1] #outlet reach ID identification
    n_reaches = len(ReachData)
    # n_classes = len(psi)
    
    # Correct the value in the Downstream_Node
    # Network['Downstream_Node'][12] = np.int64(13)
    
    # Initialise slopes 
    min_slope = min(ReachData['Slope']) # put a minimum value to guarantee movement 
    Slope = np.zeros((timescale, n_reaches))
    Slope[0,:] = np.maximum(ReachData['Slope'], min_slope)
    Slope[1,:] = np.maximum(ReachData['Slope'], min_slope) 
    
    # Initialize node elevation (for each reach the matrix reports the fromN elevation)
    # The last column reports the outlet ToNode elevation (last node of the network), which can never change elevation.
    Node_el = np.zeros((timescale, n_reaches+1))
    Node_el[0,:] = pd.concat([ReachData['el_FN'], ReachData['el_TN'][[outlet]]])
    Node_el[1,:] = pd.concat([ReachData['el_FN'], ReachData['el_TN'][[outlet]]])
    Node_el[:,-1] =  Node_el[1,-1]
    
    
    # Initialise all sediment variables 
    
    # Qbi dep matrix with size size depending on how often we want to save it:
    if save_dep_layer=='never':
        dep_save_number = 1
    if save_dep_layer=='yearly':
        dep_save_number = int(timescale/365)+1 #+1 because we also keep t0.
    if save_dep_layer=='always':
        dep_save_number= 1
    Qbi_dep = [[np.expand_dims(np.zeros(n_classes+1, dtype=numpy.float32), axis = 0) for _ in range(n_reaches)] for _ in range(dep_save_number)]
    Qbi_tr = [np.zeros((n_reaches,n_reaches,n_classes), dtype=numpy.float32) for _ in range(timescale)] # sediment within the reach AFTER transfer, which also gives the provenance 
    Qbi_mob = [np.zeros((n_reaches,n_reaches,n_classes), dtype=numpy.float32) for _ in range(timescale)] # sediment within the reach BEFORE transfer, which also gives the provenance 
    Q_out = np.zeros((timescale, n_reaches, n_classes)) # amount of material delivered outside the network in each timestep
   
    # tr_cap_all = np.zeros((timescale, n_reaches, n_classes)) #transport capacity per each sediment class
    # tr_cap_sum = np.zeros((timescale, n_reaches)) #total transport capacity 
   


    # Note Qbi_tr and Qbi_mob are 3D matrices, if we add the time as a 4th dimension, we can not look at the matrix in spyder. 
    # Fi_r_act = np.empty((timescale, n_classes, n_reaches)) # contains grain size distribution of the active layer
    # Fi_r_act[0,:] = np.nan
    # D50_AL = np.zeros((timescale, n_reaches)) # D50 of the active layer in each reach in each timestep
    # V_sed = np.zeros((timescale, n_classes, n_reaches)) #velocities
    # Qc_class_all = np.zeros((timescale, n_reaches, n_classes))
    # flow_depth = np.zeros((timescale, n_reaches)) 
    
    Delta_V_all = np.zeros((timescale, n_reaches)) # reach mass balance (volumes eroded or deposited)
    Delta_V_class_all = np.zeros((timescale, n_reaches, n_classes))
    
    # In case of constant slope
    if update_slope == False:
        Slope[:,:] = Slope[0,:]
        Node_el[:,: ] =  Node_el[0,:]

    
    # Initialise sediment deposit in the reaches 
    
    Qbi_dep_0 = [np.expand_dims(np.zeros(n_classes+1, dtype=numpy.float32), axis = 0) for _ in range(n_reaches)]
    for n in Network['NH']:  
    # If no inputs are defined, initialize deposit layer with a single cascade with no volume and GSD equal to 0
        q_bin = np.array(Qbi_dep_in[n])
        if not q_bin.any():  # If all zeros
            Qbi_dep_0[n] = np.hstack((n, np.zeros(n_classes))).reshape(1, -1)
        else:
            Qbi_dep_0[n] = np.float32(np.hstack((np.ones(q_bin.shape[0])*n, Qbi_dep_in[n,0]))).reshape(1,-1) # If there are multiple classes
        
    Qbi_dep[0] = copy.deepcopy(Qbi_dep_0)  # Only initialize the first time step

               
    # Set maximum volume in meters that can be eroded for each reach, for each time step.
    # eros_max_all = np.ones((1, n_reaches)) * eros_max
    # eros_max_vol = np.round(eros_max_all * ReachData['Wac'].values * ReachData['Length'].values, roundpar)
        
    # Set active layer volume, the one used for calculating the tr_cap in [m3/s]
    # corresponds to the depth that the river can see every second (more like a continuum carpet ...)  
    # defined here as 2.D90 [Parker 2008]
    # AL_vol_all=np.zeros((timescale, n_reaches)) #store the volumes
    # AL_depth_all=np.zeros((timescale, n_reaches)) #store also the depths 
    # # We take the input D90, or if not provided, the D84:
    # if 'D90' in ReachData:
    #     reference_D = 'D90'
    # elif 'D84' in ReachData:
    #     reference_D = 'D84'
    # for n in Network['NH']:
    #     AL_depth = np.maximum(2*ReachData[reference_D].values[n], 0.01)
    #     AL_vol = AL_depth * ReachData['Wac'].values[n] * ReachData['Length'].values[n]
    #     AL_vol_all[:,n] = np.repeat(AL_vol, timescale, axis=0)
    #     AL_depth_all[:,n] = np.repeat(AL_depth, timescale, axis=0)
          

# ================================================================
# EXAMPLE: RECRUITMENT AND LOSS SCHEDULE
# ================================================================
# Format:
# timestep: [(reach_id, volume_change), ...]
#
# volume_change may be:
#   scalar             -> one total volume
#   [class_0, class_1] -> explicit volume for each wood class
#
# For a single-class model:
#   (5, 5.0) adds 5.0 m³ to reach 5.
#
# For a two-class model with class order [L, S]:
#   (5, [5.0, 1.0]) adds 5.0 m³ L and 1.0 m³ S.
#
# Positive values add wood.
# Negative values remove wood.
# Reach IDs and timesteps are 0-based.
# Use [] when no recruitment or loss occurs.
#
# In a multiclass model, scalar values are distributed according
# to recruitment_mode. Use a vector to specify exact class volumes.

    recruitment_schedule = {
    # ----- Single-class examples -----
    # 0: [(5, 5.0)],                  # Reach 5: add 5.0 m³
    # 1: [],                          # No recruitment or loss
    # 2: [
    #     (0, 2.0),                   # Reach 0: add 2.0 m³
    #     (5, -1.0),                  # Reach 5: remove 1.0 m³
    # ],
    # 3: [(9, 3.5)],                  # Reach 9: add 3.5 m³

    # ----- Two-class examples: [L, S] -----
    # 0: [(5, [5.0, 1.0])],          # Add 5.0 m³ L and 1.0 m³ S
    # 1: [],                          # No recruitment or loss
    # 2: [
    #     (0, [2.0, 0.5]),            # Add 2.0 m³ L and 0.5 m³ S
    #     (5, [-1.0, -0.2]),          # Remove 1.0 m³ L and 0.2 m³ S
    # ],
    # 3: [(9, [0.0, 3.5])],          # Add only 3.5 m³ S
}


# ================================================================
# EXAMPLE: BARRIER SCHEDULE
# ================================================================
# Format:
# timestep: [(reach_id, retention), ...]
#
# retention may be:
#   scalar             -> same retention for every wood class
#   [class_0, class_1] -> class-specific retention
#
# Retention values:
#   0.0 = no retention; all mobilized wood transfers
#   1.0 = complete retention; no mobilized wood transfers
#   0–1 = partial retention
#
# For a two-class model, class order is [L, S].
# Barriers must be listed at every timestep they remain active.
# Reach IDs and timesteps are 0-based.

    barrier_schedule = {
    # ----- Same retention for all classes -----
    # 0: [(2, 1.0), (8, 0.5)],       # Complete at reach 2; 50% at reach 8
    # 1: [(2, 1.0), (8, 0.5)],       # Both barriers remain active
    # 2: [(2, 1.0)],                  # Barrier at reach 8 is removed
    # 3: [],                          # No active barriers

    # ----- Two-class examples: [L, S] -----
    # 0: [
    #     (2, [1.0, 0.25]),           # Retain 100% L and 25% S
    #     (8, [0.5, 0.0]),            # Retain 50% L and 0% S
    # ],
    # 1: [
    #     (2, [1.0, 0.25]),           # Both barriers remain active
    #     (8, [0.5, 0.0]),
    # ],
    # 2: [(2, [1.0, 0.25])],         # Barrier at reach 8 is removed
    # 3: [],                          # No active barriers
}


# ================================================================
# EXAMPLE: EVENT SCHEDULE
# ================================================================
# The event schedule selects the "lo" or "hi" b-range at each
# timestep.
#
# A global event label applies to all reaches.
#
# In the single-class HiLo function, the label selects one b-range.
# In the multiclass HiLo function, it selects the corresponding
# b-range for each wood class.
#
# Timesteps are 0-based.

    event_schedule = {
    # 0: "lo",                        # Low-magnitude event
    # 1: "lo",                        # Low-magnitude event
    # 2: "hi",                        # High-magnitude event

}


    # start waiting bar    
    for t in tqdm(range(timescale-1)):
        

        #FP: define flow depth and flow velocity from flow_depth_calc
        # h, v = choose_flow_depth(ReachData, Slope, Q, t, indx_flo_depth)
        # flow_depth[t] = h
        
        #FP: Slope reduction functions
        # Slope = choose_slopeRed(ReachData, Slope, Q, t, h, indx_slope_red)

        # store velocities per reach and per class, for this time step
        # v_sed = np.zeros((n_classes, n_reaches))
        
        # deposit layer from previous timestep
        Qbi_dep_old = copy.deepcopy(Qbi_dep_0)


        # loop for all reaches:
        for n in Network['NH']:
            
            # if n==2 and t==3:
            #     print("stop")
                
            #---1) Extracts the deposit layer from the storage matrix and load the incoming cascades, in [m3/d]
            V_dep_old = Qbi_dep_old[n]# extract the deposit layer of the reach 

            # if Qbi_input[t,n,:].ndim == 1:
            #     vect = np.expand_dims(np.append(n, Qbi_input[t,n,:]), axis = 0)
            # else: 
            #     vect = np.c_[np.repeat(n, Qbi_input[t,n,:].shape[0]), Qbi_input[t,n,:]]
            
            vect = np.hstack((n, np.zeros(n_classes))).reshape(1,-1) # TO BE CHECKED #
                        
            Qbi_incoming  =  np.r_[(np.c_[np.array(range(n_reaches)), Qbi_tr[t][:, n,:]]), vect] # the material present at that time step + potential external mat
            Qbi_incoming  = np.delete(Qbi_incoming, np.sum(Qbi_incoming[:,1:], axis = 1) == 0, axis = 0) # sum all classes and delete the zeros  (rows represents provenance)
            
            if Qbi_incoming.size == 0:
                Qbi_incoming = np.hstack((n, np.zeros(n_classes))) # put an empty cascade if no incoming volumes are present (for computation)
            
            if Qbi_incoming.ndim == 1:
                Qbi_incoming = np.expand_dims(Qbi_incoming, axis = 0)

            # sort incoming matrix according to distance, in this way sediment coming from closer reaches will be deposited first 
            # Qbi_incoming = sortdistance(Qbi_incoming, Network['upstream_distance_list'][n] )
            
          
            #---2) Finds cascades to be included into the active layer in [m3/s], and use the cumulative GSD to compute tr_cap
            
            # define incoming matrix in [m3/s]
            # Qbi_incoming_per_s = copy.deepcopy(Qbi_incoming)
            # Qbi_incoming_per_s[:,1:] = Qbi_incoming_per_s[:,1:] / ts_length
                        
            # find the fraction of sediments in the active layer Fi_r_act. 
            # The active layer is made of incoming load in [m3/s], and if it needs to be completed, of deposit layer sediments
            # _,_,_, Fi_r_act[t,:,n] = layer_search(Qbi_incoming_per_s, V_dep_old, AL_vol_all[0,n], roundpar)
            
            # Calculate the D50 of the AL
            # D50_AL[t,n] = D_finder(Fi_r_act[t,:,n], 50, psi)
            

            # if np.sum(Fi_r_act[t,:,n]) == 0:
            #   Fi_r_act[t,:,n] = Fi_r_act[t-1,:,n] # in case the active layer is empty, i use the GSD of the previous timesteep
            
            
            #calculate transport capacity using the Fi of the active layer, the resulting tr_cap is in m3/s and is converted in m3/day
            # tr_cap_per_s, Qc = tr_cap_function(Fi_r_act[t][:,n] , D50_AL[t,n], Slope[t,n] , Q.iloc[t,n], ReachData['Wac'][n], v[n] , h[n], psi, indx_tr_cap, indx_partition)   
            
            ## select wood transfer function ##
            V_mob, V_dep = Global_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb(V_dep_old, Qbi_incoming, ReachData, n,t=t, recruitment_schedule=recruitment_schedule, barrier_schedule=barrier_schedule,default_retention=0.0)
            # V_mob, V_dep = HiLo_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb(V_dep_old, Qbi_incoming, ReachData, n,t=t, recruitment_schedule=recruitment_schedule, barrier_schedule=barrier_schedule, event_schedule=event_schedule, default_retention=0.0)
            # V_mob, V_dep = WoodClass_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb(V_dep_old, Qbi_incoming, ReachData, n, t=t,recruitment_schedule=recruitment_schedule, barrier_schedule=barrier_schedule, default_retention=0.0)
            # V_mob, V_dep = HiLo_WoodClass_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb(V_dep_old=V_dep_old, Qbi_incoming=Qbi_incoming, ReachData=ReachData, n=n, t=t, recruitment_schedule=recruitment_schedule, barrier_schedule=barrier_schedule, default_retention=0.0, event_schedule=event_schedule)
            


            # merge duplicate provenance IDs + remove zero rows
            V_mob = matrix_compact(V_mob)
            V_dep = matrix_compact(V_dep)

           # tr_cap=tr_cap_per_s * ts_length
           #  tr_cap_all[t,n,:] = tr_cap
           #  tr_cap_sum[t,n] = np.sum(tr_cap)
            # Ensure that V_mob[:,0] contains valid integers
            # V_mob_indices = np.round(V_mob[:, 0]).astype(int)  # Round and convert to integers
            # V_mob_indices = np.clip(V_mob_indices, 0, Qbi_tr[t+1].shape[0] - 1)  # Ensure indices are within bounds
            # print(f"V_mob_indices: {V_mob_indices}")
            # print(f"Valid index range for Qbi_tr: 0 to {Qbi_tr[t+1].shape[0] - 1}")
            # # Ensure that Network['Downstream_Node'][n] is an integer
            # downstream_node = int(Network['Downstream_Node'][n])  # Convert to integer if necessary
            # downstream_node = np.clip(downstream_node, 0, Qbi_tr[t+1].shape[1] - 1)
            # #print(f"downstream_node: {downstream_node}")
            # # Print detailed shape and index info
            # #print(f"V_mob[:, 1:].shape: {V_mob[:, 1:].shape}")
            # #print(f"Qbi_tr[t+1].shape: {Qbi_tr[t+1].shape}")
            # if any(V_mob_indices >= Qbi_tr[t+1].shape[0]) or downstream_node >= Qbi_tr[t+1].shape[1]:
            #     print(f"Skipping update due to out of bounds: V_mob_indices {V_mob_indices}, downstream_node {downstream_node}")

            # if downstream_node >= Qbi_tr[t+1].shape[1]:
            #     print(f"Error: downstream_node {downstream_node} is out of bounds")

            # print(f"downstream_node: {downstream_node}")
            # print(f"V_mob[:, 1:].shape: {V_mob[:, 1:].shape}")
            # if downstream_node >= Qbi_tr.shape[1]:
            #     print(f"Error: downstream_node {downstream_node} is out of bounds")
            # if any(V_mob_indices >= Qbi_tr.shape[0]):
            #     print(f"Error: V_mob_indices {V_mob_indices} are out of bounds")
            # # if indx_tr_cap == 7:
            # #     Qc_class_all[t,n,:]=Qc
            
            # # Check if V_mob_indices is a valid 1D array
            # # if V_mob_indices.ndim == 1:
            # #    print("V_mob_indices is a valid 1D array")

            # # Now use the validated indices for indexing
            #             if n_classes == 1:
            #                 Qbi_tr[t+1][V_mob_indices, downstream_node] = np.float32(V_mob[:, 1:].squeeze())
            #             else:
            #                 Qbi_tr[t+1][V_mob_indices, downstream_node, :] = np.float32(V_mob[:, 1:])

            # print(f"Qbi_tr[t+1].shape: {Qbi_tr[t+1].shape}") 
            
            # Now use the validated indices for indexing
            #Qbi_tr[t+1][V_mob_indices, downstream_node, :] = np.float32(V_mob[:, 1:])Qbi_tr[t+1][V_mob_indices, downstream_node, :] = np.float32(V_mob[:, 1:])
            
            # Compute velocity (in m/s) directly from tr_cap, using a section of height Hvel
            # if indx_velocity == 1 or indx_velocity == 2:
            #     # coef_AL_vel = 0.1
            #     # hVel = coef_AL_vel * h[n]                # the section height is proportional to the water height h
            #     hVel = AL_depth_all[t,n]                # the section height is the same as the active layer
            #     v_sed_n = sed_velocity(hVel, ReachData['Wac'].values[n], tr_cap_per_s, phi, indx_velocity, minvel)
            #     v_sed[:,n] = v_sed_n 
              
            #----3) Finds the volume of sediment from the total incoming load of that day [m3/d] and of the deposit layer to be included in the maximum erodible layer
            # V_inc_EL , V_dep_EL ,  V_dep , _ = layer_search(Qbi_incoming, V_dep_old, eros_max_vol[0,n], roundpar)
             
            # The tr_cap cumulated over the day [m3/day] is mobilised from the maximum erodible layer
            # V_mob is the volumes actually mobilised (if tr_cap reach the max erodible layer, Vmob < tr_cap)
            # V_dep is the remaining deposit layer
            # [V_mob, V_dep ] = tr_cap_deposit( V_inc_EL, V_dep_EL, V_dep, tr_cap, roundpar)
            
                               
            # (after this passage, V_mob contains only the volumes actually mobilized)     
            Qbi_dep_0[n] = np.float32(V_dep)
            
            #remove empty rows
            #Qbi_dep_0[n] = Qbi_dep_0[n][np.sum(Qbi_dep_0[n][:,1:],axis = 1)!= 0]

            # Qbi_mob contains the volume mobilized in the reach, that is about to be transfer downstream
            Qbi_mob[t][V_mob[:,0].astype(int),n,:] = np.float32(V_mob[:,1:])

            #if removing empty rows leaves only an Qbi_dep{t,n} empty
            # matrix, put an empty layer
            if  (Qbi_dep_0[n]).size == 0 :
                Qbi_dep_0[n] = np.float32(np.append(n, np.zeros(n_classes)).reshape(1,-1))
                
            ### Transferring V_mob to the next reach ###Qbi_tr[t+1] rather than Qbi_tr[t] will cause mobilized wood from upstream to not be used in downstream reach until next timestep
            if n != outlet:
                Qbi_tr[t][V_mob[:,0].astype(int), Network['Downstream_Node'][n], :] = np.float32(V_mob[:,1:]) # TO BE CHECKED #
            else:
                Q_out[t, V_mob[:,0].astype(int), :] = np.float32(V_mob[:,1:])
   
            #---- 4) Compute the changes in bed elevation
            # modify bed elevation according to increased deposit
            Delta_V = np.sum(Qbi_dep_0[n][:,1:]) -  np.sum(Qbi_dep_old[n][:,1:])
            
            # # #in case of changing slope
            # if update_slope == True:
            #      Node_el[t+1][n]= Node_el[t,n] + Delta_V/( np.sum(ReachData['Wac'][np.append(n, Network['Upstream_Node'][n])] * ReachData['Length'][np.append(n, Network['Upstream_Node'][n])]) * (1-phi) )
            
            #record Delta_V
            Delta_V_all[t,n] = Delta_V
            
            # Delta V per class
            Delta_V_class = np.sum(Qbi_dep_0[n][:,1:], axis=0) - np.sum(Qbi_dep_old[n][:,1:], axis=0)
            Delta_V_class_all[t,n,:] = Delta_V_class
            
        
            
        
        #Save Qbi_dep according to saving frequency
        if save_dep_layer == 'always':
            # Dynamically append the new time step data
            Qbi_dep.append(copy.deepcopy(Qbi_dep_0))
            
        if save_dep_layer == 'yearly':
            if int(t+2) % 365 == 0:
                Qbi_dep.append(copy.deepcopy(Qbi_dep_0))
                
        # end of the reach loop
        
        #---5) Move the mobilized volumes to the destination reaches according to the sediment velocity

        # for n in Network['NH']:
            
        #     #load mobilized volume for reach n
        #     V_mob = np.zeros((n_reaches,n_classes+1))
        #     V_mob[:,0] = np.arange(n_reaches)
        #     V_mob[:,1:n_classes+1] = np.squeeze(Qbi_mob[t][:,[n],:], axis = 1)
        #     V_mob = matrix_compact(V_mob)
            
        #     # # Calculate sediment velocity by re-aplying the transport capacity 
        #     # # formula on the mobilised volume Vmob, and in each reach. 
        #     # if indx_velocity == 3 or indx_velocity == 4:
        #     #     # GSD of mobilized volume
        #     #     Fi_mob = (np.sum(V_mob[:,1:], axis = 0)/np.sum(V_mob[:,1:]))[:,None] # EB: must be a column vector
        #     #     if np.isnan(Fi_mob).any():
        #     #         Fi_mob = Fi_r_act[t,:,n][:,None] # EB: must be a column vector
                
        #     #     v_sed = sed_velocity_OLD( np.matlib.repmat(Fi_mob, 1, n_reaches), Slope[t,:] , Q.iloc[t,:], ReachData['Wac'] , v , h ,psi,  minvel , phi , indx_tr_cap, indx_partition, indx_velocity )
            
        #     # #transfer the sediment volume downstream according to vsed in m/day
        #     # Qbi_tr_t, Q_out_t, setplace, setout = sed_transfer_simple(V_mob , n , v_sed * ts_length , ReachData['Length'], Network, psi)

        #     # Sum the volumes transported from reach n with all the other 
        #     # volumes mobilized by all the other reaches at time
        #     Qbi_tr[t+1] = Qbi_tr[t+1] + np.float32(Qbi_tr_t)
        #     Q_out[t] =  Q_out[t] + Q_out_t
            
        # store vsed per class and per reach, of this day, in m/day
        # V_sed[t] = v_sed * ts_length
            
        # del Qbi_tr_t,Q_out_t
        

        #in case of changing slope..
        if update_slope == True:
            #..change the slope accordingly to the bed elevation
            Slope[t+1,:], Node_el[t+1,:] = change_slope(Node_el[t+1,:] ,ReachData['Length'], Network, s = min_slope)
            
        #measure time of routing
        #time2   = clock;

        #if np.remainder(10, t) == 0:  # save time only at certain timesteps 
        #   timerout = etime(time2, time1);
        
        """end of the time loop"""
        

    # output processing
    # aggregated matrixes
    # if n_classes == 1:
    #     QB_mob_t = [np.sum(x, axis=1) for x in Qbi_mob[0:timescale-1]]  # No third axis, sum along the second axis
    # else:
    #     QB_mob_t = [np.sum(x, axis=2) for x in Qbi_mob[0:timescale-1]]  # Keep the third axis for multiple classes

    QB_mob_t = [np.sum(x, axis = 2) for x in Qbi_mob[0:timescale-1]] #sum along sediment classes
    # Qbi_mob_class = [np.sum(x, axis = 0) for x in Qbi_mob[0:timescale-1]] #sum along sediment classes
    QB_mob = np.rollaxis(np.dstack(QB_mob_t),-1) 
    QB_mob_sum = np.sum(QB_mob, 1) #total sediment mobilized in that reach for that time step (all sediment classes, from all reaches)
    
    #total sediment delivered in each reach (column), divided by reach provenance (row) 
    QB_tr_t = [np.sum(x, axis=2) for x in Qbi_tr[0:timescale]]

    QB_tr = np.rollaxis(np.dstack(QB_tr_t),-1)
    
    
    V_dep_sum = np.zeros((timescale, n_reaches ))  # EB : last time step would be equal to 0 - delete to avoid confusion 
    V_class_dep = [[np.expand_dims(np.zeros(n_classes+1), axis = 0) for _ in range(n_reaches)] for _ in range(timescale)]
   
    for t in range(timescale):
        for n in range(n_reaches): 
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



            
    
    # #--D50 of mobilised volume 
    # D50_mob = np.zeros((timescale-1, n_reaches))
 
    # for t in np.arange(len(Qbi_mob_class)):
    #     Fi_mob_t = Qbi_mob_class[t]/ (np.sum(Qbi_mob_class[t],axis = 1).reshape(-1,1))
    #     Fi_mob_t[np.isnan(Fi_mob_t)] = 0
    #     for i in np.arange(n_reaches):
    #         D50_mob[t,i] = D_finder(Fi_mob_t[i,:], 50, psi)
            
            
    # #--D50 of deposited volume 
    # dep_sed_temp = []
    # D50_dep = np.zeros((timescale-1, n_reaches))
    
    # # stack the deposited volume 
    # dep_sed_temp = []
    # D50_dep = np.zeros((timescale-1, n_reaches))
    
    # for t in np.arange(len(Qbi_dep_class)):
    #     Fi_dep_t = Qbi_dep_class[t]/ (np.sum(Qbi_dep_class[t],axis = 1).reshape(-1,1))
    #     Fi_dep_t[np.isnan(Fi_dep_t)] = 0
    #     for i in np.arange(n_reaches):
    #         D50_dep[t,i] = D_finder(Fi_dep_t[i,:], 50, psi)
            
            
    # --Total material in a reach in each timestep, divided by class (transported + dep)
    # tot_sed_class =  [np.empty((len(Qbi_dep), n_reaches)) for _ in range(n_classes)]
    # q_d = np.zeros((1, n_reaches))
    
    # ================================================================
    # OUTPUTS BY WOOD CLASS
    # Works for one or multiple classes
    # ================================================================
    
    # Total wood present per class:
    # deposited wood + wood transported into the reach
    tot_sed_class = [
        np.empty((len(Qbi_dep), n_reaches))
        for _ in range(n_classes)
    ]
    
    for c in range(n_classes):
        for t in range(len(Qbi_dep)):
            deposited_t = np.array([
                np.sum(reach[:, c + 1])
                for reach in Qbi_dep[t]
            ])
    
            transported_t = np.sum(
                Qbi_tr[t][:, :, c],
                axis=0
            )
    
            tot_sed_class[c][t, :] = deposited_t + transported_t
    
    
    # Deposited wood per class
    deposited_class = [
        np.empty((len(Qbi_dep), n_reaches))
        for _ in range(n_classes)
    ]
    
    for c in range(n_classes):
        for t in range(len(Qbi_dep)):
            deposited_class[c][t, :] = np.array([
                np.sum(reach[:, c + 1])
                for reach in Qbi_dep[t]
            ])
    
    
    # Mobilized wood per class
    mobilised_class = [
        np.empty((timescale - 1, n_reaches))
        for _ in range(n_classes)
    ]
    
    for c in range(n_classes):
        for t in range(timescale - 1):
            mobilised_class[c][t, :] = np.sum(
                Qbi_mob[t][:, :, c],
                axis=0
            )
    
    
    # Transported wood per class after barrier retention
    transported_class = [
        np.empty((timescale - 1, n_reaches))
        for _ in range(n_classes)
    ]
    
    for c in range(n_classes):
        for t in range(timescale - 1):
            transported_class[c][t, :] = np.sum(
                Qbi_tr[t][:, :, c],
                axis=0
            )
    
    
    # Change in deposited wood per class
    delta_deposited_class = [
        Delta_V_class_all[:, :, c]
        for c in range(n_classes)
    ]
                        
    # #--Tranport capacity per class (put in same format as mob and trans per class)
    # tr_cap_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    # for c in range(n_classes): 
    #     for t in range(timescale-1): 
    #         q_m = tr_cap_all[t,:,c]
    #         tr_cap_class[c][t,:] = q_m     
    
    # #--Critical discharge per class (put in same format as mob and trans per class)
    # if indx_tr_cap == 7:   
    #     Qc_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    #     for c in range(n_classes): 
    #         for t in range(timescale-1): 
    #             q_m = Qc_class_all[t,:,c]
    #             Qc_class[c][t,:] = q_m  
            
    # Q_out_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    # for c in range(n_classes): 
    #     for t in range(timescale-1): 
    #         if n_classes > 1:
    #             q_m = Q_out[t][:,:,c]
    #         else:
    #             q_m = Q_out[t][:,:]
            
    #         Q_out_class[c][t,:] = q_m 
    
    
    # V_sed_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    # for t in range(timescale-1):
    #     for c in range(n_classes):
    #         q_m = V_sed[t,c,:]
    #         V_sed_class[c][t, :] = q_m
        
    #--Total sediment volume leaving the network
    outcum_tot = np.array([np.sum(x) for x in Q_out])
    
    
    
    #set all NaN transport capacity to 0
    # tr_cap_sum[np.isnan(tr_cap_sum)] = 0 
    
    # #set all NaN active layer D50 to 0; 
    # D50_AL[np.isnan(D50_AL)] = 0
    
    # Q = np.array(Q)
    
    
        
    #--Output struct definition 
    #data_plot contains the most important D_CASCADE outputs 
    data_output = { # 'Channel Width [m]': np.array([ReachData[f'Wac_{t}'].values for t in range(timescale)]).T,  # Modified for varying channel widths in input
                   'Channel Width [m]': np.repeat(np.array(ReachData['Wac']).reshape(1,-1),len(Qbi_dep), axis = 0),  #Original
                   'Reach Slope' : Slope,   
                   # 'Discharge [m^3/s]': Q[0:timescale,:],                    
                   'Mobilized [m^3]' : QB_mob_sum,
                   'Transported [m^3]':  tot_tranported,                   
                   'Transported + deposited [m^3]':tot_sed,   
                   # 'D50 deposit layer [m]' :D50_dep, 
                   # 'D50 mobilised layer [m]':D50_mob,
                   # 'D50 active layer [m]' :D50_AL,  
                   # 'Transport capacity [m^3]': tr_cap_sum,                   
                   'Deposit layer [m^3]': V_dep_sum, 
                   'Delta deposit layer [m^3]' : Delta_V_all,
                   'Transported + deposited - per class [m^3]':  tot_sed_class, 
                   'Deposited - per class [m^3]' : deposited_class,
                   'Mobilised - per class [m^3]': mobilised_class,
                   'Transported- per class [m^3]': transported_class,
                   'Delta deposit layer - per class [m^3]': Delta_V_class,
                   # 'Transport capacity - per class [m^3]': tr_cap_class,
                   # 'Sed_velocity [m/day]': V_sed,
                   # 'Sed_velocity - per class [m/day]': V_sed_class,
                   # 'Flow depth': flow_depth,
                   # 'Active layer [m]': AL_depth_all,
                   # 'Maximum erosion layer [m]': eros_max_all,
                   'Q_out [m^3]' : Q_out,
                   # 'Q_out_class [m^3]' : Q_out_class,  
                   'Q_out_tot' : outcum_tot
                   }

    # if indx_tr_cap == 7:
    #     data_output["Qc - per class"] = Qc_class
         
    #all other outputs are included in the extended_output cell variable 
    extended_output = { 'Qbi_tr': Qbi_tr,  
                   'Qbi_mob' : Qbi_mob  , 
                   'Q_out' : Q_out ,  
                   'Qbi_dep':Qbi_dep, 
                   # 'Fi_r_ac' :Fi_r_act ,  
                   'Node_el' : Node_el, 
                   }
    
    return data_output,extended_output


















