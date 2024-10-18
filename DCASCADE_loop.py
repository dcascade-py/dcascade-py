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

from supporting_functions import D_finder
from supporting_functions import sortdistance
from supporting_functions import layer_search
from supporting_functions import tr_cap_deposit
from supporting_functions import matrix_compact 
# from supporting_functions import sed_transfer_simple
from supporting_functions import change_slope
from supporting_functions import stop_or_not
from supporting_functions import deposit_from_passing_sediments
from transport_capacity_computation import tr_cap_function
from transport_capacity_computation import sed_velocity
from transport_capacity_computation import sed_velocity_OLD
from flow_depth_calc import choose_flow_depth
from slope_reduction import choose_slopeRed
import itertools

np.seterr(divide='ignore', invalid='ignore')
             
""" MAIN FUNCTION SECTION """


class Cascade:
    def __init__(self, provenance, elapsed_time, volume):
        self.provenance = provenance
        self.elapsed_time = elapsed_time
        self.volume = volume
        # To be filled during the time step
        self.velocities = np.nan
        
class ReachData:
    def __init__(self, geodataframe):
        self.n_reaches = len(geodataframe)
        
        # Mandatory attributes
        self.from_n = geodataframe['FromN'].astype(int).values
        self.to_n = geodataframe['ToN'].astype(int).values
        self.slope = geodataframe['Slope'].astype(float).values
        self.wac = geodataframe['Wac'].astype(float).values
        self.n = geodataframe['n'].astype(float).values
        self.D16 = geodataframe['D16'].astype(float).values
        self.D50 = geodataframe['D50'].astype(float).values
        self.D84 = geodataframe['D84'].astype(float).values
        self.length = geodataframe['Length'].astype(float).values
        self.el_fn = geodataframe['el_FN'].astype(float).values
        self.el_tn = geodataframe['el_TN'].astype(float).values

        # Optional attributes
        self.reach_id = geodataframe['reach_id'].values if 'reach_id' in geodataframe.columns else np.nan
        self.id = geodataframe['Id'].values if 'Id' in geodataframe.columns else np.nan
        self.q = geodataframe['Q'].values if 'Q' in geodataframe.columns else np.nan
        self.wac_bf = geodataframe['Wac_BF'].values if 'Wac_BF' in geodataframe.columns else np.nan
        self.D90 = geodataframe['D90'].values if 'D90' in geodataframe.columns else np.nan
        self.s_lr_gis = geodataframe['S_LR_GIS'].values if 'S_LR_GIS' in geodataframe.columns else np.nan
        self.tr_limit = geodataframe['tr_limit'].values if 'tr_limit' in geodataframe.columns else np.nan
        self.x_fn = geodataframe['x_FN'].values if 'x_FN' in geodataframe.columns else np.nan
        self.y_fn = geodataframe['y_FN'].values if 'y_FN' in geodataframe.columns else np.nan
        self.x_tn = geodataframe['x_TN'].values if 'x_TN' in geodataframe.columns else np.nan
        self.y_tn = geodataframe['y_TN'].values if 'y_TN' in geodataframe.columns else np.nan
        self.ad = geodataframe['Ad'].values if 'Ad' in geodataframe.columns else np.nan
        self.direct_ad = geodataframe['directAd'].values if 'directAd' in geodataframe.columns else np.nan
        self.strO = geodataframe['StrO'].values if 'StrO' in geodataframe.columns else np.nan
        self.deposit = geodataframe['deposit'].values if 'deposit' in geodataframe.columns else np.nan
        self.geometry = geodataframe['geometry'].values if 'geometry' in geodataframe.columns else np.nan
        
    def sort_values_by(self, sorting_array):
        """
        Function to sort the Reaches by the array given in input.
        """
        # Making sure the array given has the right length
        assert(len(sorting_array) == self.n_reaches)
        
        # Get the indices that would sort sorting_array
        sorted_indices = np.argsort(sorting_array)
        
        # Loop through all attributes
        for attr_name in vars(self):

            # Check if these are reach attributes
            attr_value = vars(self)[attr_name]
            if isinstance(attr_value, np.ndarray) and len(attr_value) == self.n_reaches:
                vars(self)[attr_name] = attr_value[sorted_indices]
                
        return sorted_indices



def compute_cascades_velocities(reach_cascades_list, 
                               indx_velocity, indx_velocity_partitioning, hVel,
                               indx_tr_cap, indx_partition,
                               reach_width, reach_slope, Q_reach, v, h,
                               phi, minvel, psi,
                               reach_Vdep, active_layer_volume,
                               roundpar):
    ''' Compute the velocity of the cascades in reach_cascade_list.
    The velocity must be assessed by re-calculating the transport capacity 
    in the present reach, considering the effect of the arriving cascade(s).
    Two methods are proposed to re-evaluated the transport capacity, chosen 
    by the indx_velocity. 
    First method: the simplest, we re-calculate the transport capacity on the cascade itself.
    Second method: we consider the active layer volume, to complete, if needed, 
    the list of cascade by some reach material. If the cascade volume is more 
    than the active layer, we consider all the cascade volume.
    '''
    if indx_velocity == 1:
        velocities_list = []
        for cascade in reach_cascades_list:
            cascade.velocities = volume_velocities(cascade.volume, 
                                                   indx_velocity_partitioning, 
                                                   hVel, phi, minvel, psi,
                                                   indx_tr_cap, indx_partition,
                                                   reach_width, reach_slope,
                                                   Q_reach, v, h)
            velocities_list.append(cascade.velocities)
        # In this case, we store the averaged velocities obtained among all the cascades
        velocities = np.mean(np.array(velocities_list), axis = 0)
            
    if indx_velocity == 2:
        # concatenate cascades in one volume, and compact it by original provenance
        # DD: should the cascade volume be in [m3/s] ?
        volume_all_cascades = np.concatenate([cascade.volume for cascade in reach_cascades_list], axis=0) 
        volume_all_cascades = matrix_compact(volume_all_cascades)
        
        volume_total = np.sum(volume_all_cascades[:,1:])
        if volume_total < active_layer_volume:
            _, Vdep_active, _, _ = layer_search(volume_all_cascades, reach_Vdep, 
                                   active_layer_volume, roundpar)
            volume_all_cascades = np.concatenate([volume_all_cascades, Vdep_active], axis=0) 

        velocities = volume_velocities(volume_all_cascades, indx_velocity_partitioning, 
                                       hVel, phi, minvel, psi,
                                       indx_tr_cap, indx_partition,
                                       reach_width, reach_slope,
                                       Q_reach, v, h)
        
        for cascade in reach_cascades_list:
            cascade.velocities = velocities
    
    if indx_velocity == 3: 
        # We want to reproduce old cascade results here, compute velocity from Vdep only
        incoming_active, Vdep_active, _, _ = layer_search(reach_Vdep, 
                                                          active_layer_volume, roundpar)
        
        velocities = volume_velocities(Vdep_active, indx_velocity_partitioning, 
                                       hVel, phi, minvel, psi,
                                       indx_tr_cap, indx_partition,
                                       reach_width, reach_slope,
                                       Q_reach, v, h)
        
        for cascade in reach_cascades_list:
            cascade.velocities = velocities
    
    return velocities
        
            
def volume_velocities(volume, indx_velocity_partitioning, hVel, phi, minvel, psi,
                      indx_tr_cap, indx_partition,
                      reach_width, reach_slope, Q_reach, v, h):
    
    ''' Compute the velocity of the volume of sediments. The transport capacity [m3/s]
    is calculated on this volume, and the velocity is calculated by dividing the 
    transport capacity by a section (hVel x width x (1 - porosity)). 
    For partionning the section among the different sediment class in the volume, 
    two methods are proposed. The first one put the same velocity to all classes.
    The second divides the section equally, so the velocity stays proportional to 
    the transport capacity of that class.
    
    '''
    # Find volume sediment class fractions and D50
    volume_total = np.sum(volume[:,1:])
    volume_total_per_class = np.sum(volume[:,1:], axis = 0)
    sed_class_fraction = volume_total_per_class / volume_total
    D50 = float(D_finder(sed_class_fraction, 50, psi))
    
    # Compute the transport capacity
    [ tr_cap_per_s, pci ] = tr_cap_function(sed_class_fraction, D50,  
                                       reach_slope, Q_reach, reach_width,
                                       v , h, psi, 
                                       indx_tr_cap, indx_partition)
    
    Svel = hVel * reach_width * (1 - phi)  # the global section where all sediments pass through

    if indx_velocity_partitioning == 1:
        velocity_same = np.sum(tr_cap_per_s) / Svel     # same velocity for each class
        velocity_same = np.maximum(velocity_same , minvel)    # apply the min vel threshold
        velocities = np.full(len(tr_cap_per_s), velocity_same) # put the same value for all classes
    elif indx_velocity_partitioning == 2:
        Si = Svel / len(tr_cap_per_s)             # same section for all sediments
        velocities = np.maximum(tr_cap_per_s/Si , minvel)

    return velocities


def compute_time_lag(cascade_list, n_classes):
    ''' The time lag is the time we use to mobilise from the reach, 
    before cascades from upstream reaches arrive at the outlet of the present reach.
    We take it as the time for the first cascade to arrive at the outet.
    '''
    
    if cascade_list == []:
        time_lag = np.ones(n_classes) # the time lag is the entire time step as no other cascade reach the outlet
    else:
        time_arrays = np.array([cascade.elapsed_time for cascade in cascade_list])
        time_lag = np.min(time_arrays, axis=0) 
        
    return time_lag


def cascades_end_time_or_not(cascade_list_old, reach_length, ts_length):
    ''' Fonction to decide if the traveling cascades in cascade list stop in 
    the reach or not, due to the end of the time step.
    Inputs:
        cascade_list_old:    list of traveling cascades
        reach_length:           reach physical length
        ts_length:              time step length
        
    Return:
        cascade_list_new:       same cascade list updated. Stopping cascades or 
                                partial volumes have been removed
                                                          
        depositing_volume:      the volume to be deposited in this reach. 
                                They are ordered according to their arrival time
                                at the inlet, so that volume arriving first 
                                deposit first.
    '''   
    # Order cascades according to their arrival time, so that first arriving 
    # cascade are deposited first 
    # Note: in the deposit layer matrix, the upper most layers are the last
    cascade_list_old = sorted(cascade_list_old, key=lambda x: np.mean(x.elapsed_time))
    
    depositing_volume_list = []
    arrival_mean_time = []
    cascades_to_be_completely_removed = []
    
    for cascade in cascade_list_old:
        # Time in, time travel, and time out in time step unit (not seconds)
        t_in = cascade.elapsed_time
        t_travel_n = reach_length / (cascade.velocities * ts_length)
        t_out = t_in + t_travel_n
        # Vm_stop is the stopping part of the cascade volume
        # Vm_continue is the continuing part
        Vm_stop, Vm_continue = stop_or_not(t_out, cascade.volume)
        
        if Vm_stop is not None:
            depositing_volume_list.append(Vm_stop)
            arrival_mean_time.append(np.mean(t_in))
            
            if Vm_continue is None: 
                # no part of the volume continues, we remove the entire cascade
                cascades_to_be_completely_removed.append(cascade)
            else: 
                # some part of the volume continues, we update the volume
                cascade.volume = Vm_continue
                                
        if Vm_continue is not None:
            # update time for continuing cascades
            cascade.elapsed_time = t_out   
    
    # If they are, remove complete cascades:
    cascade_list_new = [casc for casc in cascade_list_old if casc not in cascades_to_be_completely_removed]   
    
    # If they are, concatenate the deposited volumes in the reverse arrival time order
    if depositing_volume_list != []:
        depositing_volume = np.concatenate(depositing_volume_list, axis=0)
        depositing_volume = matrix_compact(depositing_volume)
    else:
        depositing_volume = None
    
    return cascade_list_new, depositing_volume
               




def DCASCADE_main(indx_tr_cap, indx_partition, indx_flo_depth, indx_slope_red, 
                  indx_velocity, indx_velocity_partition,
                  reach_data, network, Q, Qbi_input, Qbi_dep_in, timescale, psi, roundpar, 
                  update_slope, eros_max, save_dep_layer, ts_length,
                  consider_overtaking_sed_in_outputs,
                  compare_with_tr_cap, time_lag_for_mobilised):
    """
    Main function of the D-CASCADE software.
    
    INPUT :
    indx_tr_cap    = the index indicating the transport capacity formula
    indx_partition = the index indicating the type of sediment flux partitioning
    indx_flo_depth = the index indicating the flow depth formula
    indx_slope_red = the index indicating the slope reduction formula
    reach_data     = nx1 Struct defining the features of the network reaches
    network        = 1x1 struct containing for each node info on upstream and downstream nodes
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
    extended_output  = struct collecting the raw D-CASCADE output datasets
    """
    
    
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
        
        #FP: define flow depth and flow velocity from flow_depth_calc
        h, v = choose_flow_depth(reach_data, slope, Q, t, indx_flo_depth)
        flow_depth[t] = h
        
        #FP: slope reduction functions
        slope = choose_slopeRed(reach_data, slope, Q, t, h, indx_slope_red)

        # # store velocities per reach and per class, for this time step
        # v_sed = np.zeros((n_reaches, n_classes))
        
        # deposit layer from previous timestep
        Qbi_dep_old = copy.deepcopy(Qbi_dep_0)
        
        # volumes of sediment passing through a reach in this timestep,
        # ready to go to the next reach in the same time step.
        Qbi_pass = [[] for n in range(n_reaches)]
        
        # loop for all reaches:
        for n in network['n_hier']:  
            if t==35 and n==30:
                print('ok')
                
            # Find reach downstream of reach n    
            if n != int(network['outlet']):
                n_down = np.squeeze(network['downstream_node'][n], axis = 1) #DD: outlet case
                n_down = int(n_down) # DD: This is wrong if there is more than 1 reach downstream (to do later)
            else:
                n_down = None
                
            # Extracts the deposit layer left in previous time step          
            V_dep_init = Qbi_dep_old[n] # extract the deposit layer of the reach 
            
            
            if consider_overtaking_sed_in_outputs == False:
                if Qbi_input[t,n,:].ndim == 1:
                    vect = np.expand_dims(np.append(n, Qbi_input[t,n,:]), axis = 0)
                else: 
                    vect = np.c_[np.repeat(n, Qbi_input[t,n,:].shape[0]), Qbi_input[t,n,:]]
                
                Qbi_incoming  =  np.r_[(np.c_[np.array(range(n_reaches)), Qbi_tr[t][:, n,:]]), vect] # the material present at that time step + potential external mat
                Qbi_incoming  = np.delete(Qbi_incoming, np.sum(Qbi_incoming[:,1:], axis = 1)==0, axis = 0) # sum all classes and delete the zeros  (rows represents provenance)
                
                if Qbi_incoming.size == 0:
                    Qbi_incoming = np.hstack((n, np.zeros(n_classes))) # put an empty cascade if no incoming volumes are present (for computation)
                
                if Qbi_incoming.ndim == 1:
                    Qbi_incoming = np.expand_dims(Qbi_incoming, axis = 0)
    
                # sort incoming matrix according to distance, in this way sediment coming from closer reaches will be deposited first (DD: relative to original provenance ?)
                Qbi_incoming = sortdistance(Qbi_incoming, network['upstream_distance_list'][n])
            else:
                Qbi_incoming = None 
                
            
            ###------Step 1 : Cascades generated from the reaches upstream during 
            # the present time step, are passing the inlet of the reach
            # (stored in Qbi_pass[n]).
            # This step make them pass to the outlet of the reach or stop there,
            # depending if they reach the end of the time step or not.
                        
            # Store the arriving cascades in the transported matrix (Qbi_tr)
            # Note: we store the volume by original provenance
            if consider_overtaking_sed_in_outputs == True:
                for cascade in Qbi_pass[n]:
                    Qbi_tr[t][[cascade.volume[:,0].astype(int)], n, :] += cascade.volume[:, 1:]
                    # DD: If we want to store instead the direct provenance
                    # Qbi_tr[t][cascade.provenance, n, :] += np.sum(cascade.volume[:, 1:], axis = 0)  
                
                
            # Compute the velocity of the cascades in this reach [m/s] 
            if Qbi_pass[n] != []:
                # coef_AL_vel = 0.1
                # hVel = coef_AL_vel * h                # the section height is proportional to the water height h
                hVel = al_depth_all[t,n]  
    
                velocities = compute_cascades_velocities(Qbi_pass[n], 
                                           indx_velocity, indx_velocity_partition, hVel,
                                           indx_tr_cap, indx_partition,
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
                                                    
            ###------Step 2 : Mobilise volumes from the reach material before 
            # considering the passing cascades. 
            # The parameter "time_lag" is the proportion of the time step where this
            # mobilisation occurs, i.e. before the first possible cascade arrives 
            # at the outlet.
            # For now this step 2 is optional.
            if compare_with_tr_cap == True:
                if time_lag_for_mobilised == True:
                    time_lag = compute_time_lag(Qbi_pass[n], n_classes)
                else:
                    # in this condition (we compare with tr cap at the outlet,
                    # but no time lag is considered), we don't mobilised from the
                    # reach before the possible cascades arrive
                    time_lag = np.zeros(n_classes)
            else:
                # in this condition (compare_with_tr_cap = False), 
                # we always mobilise from the reach itself and 
                # the passing cascades are passing the outlet, without 
                # checking the energy available to make them pass,
                # like in version 1 of the code
                time_lag = np.ones(n_classes)                      
                           
            # In the case time_lag is not all zeros, we mobilise from the reach
            # before the cascades arrive.
            if np.any(time_lag != 0):
                # Get the sediment fraction and D50 in the active layer.                          
                _,_,_, Fi_r_act[t,n,:] = layer_search(V_dep_init, al_vol_all[0,n], roundpar, Qbi_incoming = Qbi_incoming)
                D50_AL[t,n] = D_finder(Fi_r_act[t,n,:], 50, psi)           
                # In case the active layer is empty, I use the GSD of the previous timestep
                if np.sum(Fi_r_act[t,n,:]) == 0:
                   Fi_r_act[t,n,:] = Fi_r_act[t-1,n,:] 
                # Transport capacity in m3/s 
                tr_cap_per_s, Qc = tr_cap_function(Fi_r_act[t,n,:] , D50_AL[t,n], slope[t,n] , Q[t,n], reach_data.wac[n], v[n] , h[n], psi, indx_tr_cap, indx_partition)                                   
                # Store tr_cap (and Qc)
                tr_cap_all[t,n,:] = tr_cap_per_s
                tr_cap_sum[t,n] = np.sum(tr_cap_per_s)
                if indx_tr_cap == 7:
                    Qc_class_all[t,n,:]=Qc   
                    
                # Total volume possibly mobilised during the time lag
                time_for_mobilising = time_lag * ts_length
                volume_mobilisable = tr_cap_per_s * time_for_mobilising
                
                # Erosion maximum during the time lag 
                # (we take the mean time lag among the classes)
                eros_max_vol_t = eros_max_vol[0,n] * np.mean(time_lag)
                
                # Mobilise from the reach layers
                V_inc_EL, V_dep_EL, V_dep_init, _ = layer_search(V_dep_init, eros_max_vol_t, roundpar, Qbi_incoming = Qbi_incoming)
                [V_mob, V_dep_init] = tr_cap_deposit(V_inc_EL, V_dep_EL, V_dep_init, volume_mobilisable, roundpar)
                
                # Store this volume only (if option 1 is False):
                if consider_overtaking_sed_in_outputs == False:
                    Qbi_mob[t][[V_mob[:,0].astype(int)], n, :] += V_mob[:, 1:]
                
                # The Vmob is added to the arriving sediment for the reach downstream
                # (Only if the present reach is not the outlet)
                if n != int(network['outlet']):  
                    if np.any(V_mob[:,1:] != 0):
                        elapsed_time = np.zeros(n_classes) #it elapsed time is 0
                        provenance = n
                        Qbi_pass[n_down].append(Cascade(provenance, elapsed_time, V_mob))
            
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
                _,_,_, Fi_r = layer_search(V_dep_after_tlag, al_vol_all[0,n], roundpar, Qbi_incoming = Q_pass_volume_per_s) 
                D50_2 = float(D_finder(Fi_r, 50, psi))   
                tr_cap_per_s, Qc = tr_cap_function(Fi_r , D50_2, slope[t,n] , Q[t,n], reach_data.wac[n], v[n] , h[n], psi, indx_tr_cap, indx_partition)                                   
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
                    V_inc_EL, V_dep_EL, V_dep_after_tlag, _ = layer_search(V_dep_after_tlag, eros_max_vol_t, roundpar, Qbi_incoming = Q_pass_volume_per_s)
                    [V_mob, V_dep_after_tlag] = tr_cap_deposit(V_inc_EL, V_dep_EL, V_dep_after_tlag, diff_pos, roundpar)
                    
                    # The Vmob is added to the arriving sediment for the reach downstream
                    # (Only if the present reach is not the outlet)
                    if n != int(network['outlet']):
                        elapsed_time = time_lag # it start its journey at the end of the time lag
                        provenance = n
                        Qbi_pass[n_down].append(Cascade(provenance, elapsed_time, V_mob))
                                        
                # Sediment classes with negative values in diff_with_capacity
                # are over capacity
                # They are deposited, i.e. directly added to Vdep
                diff_neg = -np.where(diff_with_capacity > 0, 0, diff_with_capacity)     
                if np.any(diff_neg):  
                    Vm_removed, Qbi_pass[n], residual = deposit_from_passing_sediments(np.copy(diff_neg), Qbi_pass[n], roundpar)
                    V_dep_after_tlag = np.concatenate([V_dep_after_tlag, Vm_removed], axis=0) 
                    
            # After this step, Qbi_pass[n] contains the traveling volumes that effectively 
            # reach and pass the outlet of the reach. They are passed to the next reach. 
            # (Only if the present reach is not the outlet)
            if n != int(network['outlet']):       
                Qbi_pass[n_down].extend(Qbi_pass[n]) 
                
            # Update the deposit layer
            V_dep_final = V_dep_after_tlag

            ###-----Step 4: Finalisation.
            # Deposit now the stopping cascades in Vdep 
            if to_be_deposited is not None:     
                if consider_overtaking_sed_in_outputs == False:
                    Qbi_tr[t+1][[to_be_deposited[:,0].astype(int)], n, :] += to_be_deposited[:, 1:]
                else:
                    V_dep_final = np.concatenate([V_dep_final, to_be_deposited], axis=0)
                
            # Store Vdep for next time step
            Qbi_dep_0[n] = np.float32(V_dep_final)

            # Store the mobilised volumes (Qbi_mob),
            if consider_overtaking_sed_in_outputs == True:
                for cascade in Qbi_pass[n]:
                    Qbi_mob[t][[cascade.volume[:,0].astype(int)], n, :] += cascade.volume[:, 1:]
                    # DD: If we want to store instead the direct provenance
                    # Qbi_mob[t][cascade.provenance, n, :] += np.sum(cascade.volume[:, 1:], axis = 0)
            
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
                            
        # Compute Qout for this time step.
        outlet = int(network['outlet'])
        for cascade in Qbi_pass[outlet]:
            Vm = cascade.volume
            Q_out[t, [Vm[:,0].astype(int)], :] += Vm[:,1:]
            
        # in case of changing slope..
        if update_slope == True:
            #..change the slope accordingly to the bed elevation
            slope[t+1,:], node_el[t+1,:] = change_slope(node_el[t+1,:] ,reach_data.length, network, s = min_slope)
                
    """end of the time loop"""                

# ##########################################         
#             #---Extract the cascades, that had stopped in reach n at t-1 [m3/d]
#             if Qbi_input[t,n,:].ndim == 1:
#                 vect = np.expand_dims(np.append(n, Qbi_input[t,n,:]), axis = 0)
#             else: 
#                 vect = np.c_[np.repeat(n, Qbi_input[t,n,:].shape[0]), Qbi_input[t,n,:]]
            
#             Qbi_incoming  =  np.r_[(np.c_[np.array(range(n_reaches)), Qbi_tr[t][:, n,:]]), vect] # the material present at that time step + potential external mat
#             Qbi_incoming  = np.delete(Qbi_incoming, np.sum(Qbi_incoming[:,1:], axis = 1)==0, axis = 0) # sum all classes and delete the zeros  (rows represents provenance)
            
#             if Qbi_incoming.size == 0:
#                 Qbi_incoming = np.hstack((n, np.zeros(n_classes))) # put an empty cascade if no incoming volumes are present (for computation)
            
#             if Qbi_incoming.ndim == 1:
#                 Qbi_incoming = np.expand_dims(Qbi_incoming, axis = 0)

#             # sort incoming matrix according to distance, in this way sediment coming from closer reaches will be deposited first (DD: relative to original provenance ?)
#             Qbi_incoming = sortdistance(Qbi_incoming, network['upstream_distance_list'][n])
            
                       
  
            
#             if consider_passing_sed_in_tr_cap == True:
#                 # Include the cascades arriving from the reach(es) upstream in this time step (Qbi_pass_from_n_up) 
#                 # to cascasde that stopped here at t-1 (Qbi_incoming), 
#                 # in order to compute the transport capacity.
#                 # This way we compute the transport capacity based on all the sediments 
#                 # present in the reach during that time step.
#                 if len(Qbi_pass_from_n_up) != 0:
#                     Qbi_pass_t = np.concatenate([cascade.volume for cascade in Qbi_pass_from_n_up], axis=0)             
#                     Qbi_incoming = np.concatenate([Qbi_pass_t, Qbi_incoming], axis=0)
#                     Qbi_incoming = matrix_compact(Qbi_incoming)         # group layers by initial provenance            

#             #---Finds cascades of the incoming load in [m3/s], 
#             # and of the deposit layer, to be included into the active layer, 
#             # and use its cumulative GSD to compute tr_cap            
#             Qbi_incoming_per_s = copy.deepcopy(Qbi_incoming)
#             Qbi_incoming_per_s[:,1:] = Qbi_incoming_per_s[:,1:] / ts_length                                            
#             # Fraction of sediments in the active layer Fi_r_act. 
#             _,_,_, Fi_r_act[t,n,:] = layer_search(Qbi_incoming_per_s, V_dep_old, al_vol_all[0,n], roundpar, n,t, temp_idx='per_sec')            
#             # Calculate the D50 of the AL
#             D50_AL[t,n] = D_finder(Fi_r_act[t,n,:], 50, psi)
            
#             # In case the active layer is empty, I use the GSD of the previous timestep
#             if np.sum(Fi_r_act[t,n,:]) == 0:
#                Fi_r_act[t,n,:] = Fi_r_act[t-1,n,:] 
                      

#             # Total volume possibly mobilised in the time step
#             tr_cap = tr_cap_per_s * ts_length
#             # Store tr_cap
#             tr_cap_all[t,n,:] = tr_cap
#             tr_cap_sum[t,n] = np.sum(tr_cap)
            
#             if indx_tr_cap == 7:
#                 Qc_class_all[t,n,:]=Qc
            
#             # Compute velocity (in m/s) directly from tr_cap, using a section of height Hvel
#             if indx_velocity == 1 or indx_velocity == 2:
#                 # coef_AL_vel = 0.1
#                 # hVel = coef_AL_vel * h                # the section height is proportional to the water height h
#                 hVel = al_depth_all[t,n]                # the section height is the same as the active layer
#                 v_sed_n = sed_velocity(hVel, reach_data.wac[n], tr_cap_per_s, phi, indx_velocity, minvel)
#                 v_sed[n,:] = v_sed_n
                
                
#             #---Finds cascades from the total incoming load of that day [m3/d] 
#             # and of the deposit layer to be included in the maximum erodible layer
#             V_inc_EL, V_dep_EL, V_dep, _ = layer_search(Qbi_incoming, V_dep_old, eros_max_vol[0,n], roundpar, n, t, temp_idx='daily')
#             #--> DD: verify that this step does not modify Qbi_incoming and V_dep_old 
                
#             # Update the time in each cascade, by adding the time to pass 
#             # trought the current reach.
#             # Stop if the new time is larger than the time step, 
#             # i.e. add the cascade to Qbi_tr[t+1].
#             # If the cascade does not stop, 
#             # i.e. add the volume to Qbi_pass[n] and to Qbi_tr[t]
#             # Note: time is expressed in time_step units (not seconds)
#             t_travel_n = reach_data.length[n] / (v_sed[n,:] * ts_length)
#             for cascade in Qbi_pass_from_n_up:
#                 t_new = cascade.elapsed_time + t_travel_n
#                 Vm_stop, Vm_continue = stop_or_not(t_new, cascade.volume)
#                 if Vm_stop is not None:
#                     Qbi_tr[t+1][[Vm_stop[:,0].astype(int)], n, :] += Vm_stop[:, 1:]                  
#                 if Vm_continue is not None:
#                     Qbi_pass[n].append(Cascade(cascade.provenance, t_new, Vm_continue))
#                     if consider_overtaking_sed_in_outputs == True:
#                         Qbi_tr[t][[Vm_continue[:,0].astype(int)], n, :] += Vm_continue[:, 1:]
#                 # DD: think about this strange fact, that we put the transported
#                 # volumes either at t or t+1 depending if they finish the day there or not
            
#             # Compare sum of Qbi_pass[n] to tr_cap volume (for each sediment class)
#             if compare_with_tr_cap == True:           
#                 if Qbi_pass[n] == []:
#                     sum_pass = np.zeros(len(tr_cap))
#                 else:
#                     concat_cascades = np.concatenate([cascade.volume for cascade in Qbi_pass[n]], axis=0)
#                     sum_pass = np.sum(concat_cascades[:,1:], axis=0)
#                 diff_with_capacity = tr_cap - sum_pass
                
#                 # In addition, consider a time lag during which we mobilise sediments 
#                 # in the reach n, while the first cascade travels to the ToN of reach n
#                 if time_lag_for_Vmob == True:
#                     # Note: time lag is in time step units (not in seconds)
#                     if Qbi_pass[n] == []:
#                         time_lag = 1 # the complete time step is considered 
#                     else:
#                         time_arrays = np.array([cascade.elapsed_time for cascade in Qbi_pass[n]])
#                         time_lag = np.min(time_arrays, axis=0)
                          
#                     # Quantities to mobilise during the time lag
#                     # volume_time_lag = np.round(tr_cap * time_lag , decimals=roundpar)
#                     volume_time_lag = tr_cap * time_lag
#                     # The volume mobilised during the time lag is considered in the budget
#                     diff_with_capacity -= volume_time_lag
                          
#                 # Sediment classes with negative values in diff_with_capacity are deposited, 
#                 # i.e. directly added to Qbi_dep
#                 diff_neg=-np.where(diff_with_capacity > 0, 0, diff_with_capacity)     
#                 if np.any(diff_neg):  
#                     Vm_removed, Qbi_pass[n], residual = deposit_from_passing_sediments(np.copy(diff_neg), Qbi_pass[n], roundpar)
#                     # Vm_removed is directly added to Qbi_dep for the next time step
#                     # but we do it after we mobilise from the reach
#                     to_be_deposited = Vm_removed
#                     # Qbi_tr[t+1][[Vm_removed[:,0].astype(int)], n, :] += Vm_removed[:, 1:]
#                 else:
#                     to_be_deposited = None
                    
#                 # Sediment classes with positive values are mobilised from the reach n,
#                 # V_mob is the mobilised cascade, V_dep is the new deposit layer           
#                 diff_pos = np.where(diff_with_capacity < 0, 0, diff_with_capacity)
#                 # We also add the volume mobilised during the time lag, 
#                 # because it is also mobilised from the reach n
#                 if time_lag_for_Vmob == True:
#                     diff_pos += volume_time_lag        
#                 if np.any(diff_pos):                               
#                     [V_mob, V_dep] = tr_cap_deposit(V_inc_EL, V_dep_EL, V_dep, diff_pos, roundpar)         
#                 else:
#                     # Case when there is no mobilised volume from the reach n
#                     V_mob = None
#                     V_dep = V_dep_old #in this case, V_dep has not changed in reach n
#             else:
#                 # Case where we don't compare to tr_cap 
#                 [V_mob, V_dep] = tr_cap_deposit(V_inc_EL, V_dep_EL, V_dep, tr_cap, roundpar)                       
#                 to_be_deposited = None
                
#                 # def a_des_doublons(vecteur):
#                 #     return len(vecteur) != len(set(vecteur))
#                 # if a_des_doublons(V_mob[:,0]):
#                 #     print("The vecteur has several time the same provenance.")
                
#             # Add the possible V_mob cascade coming from reach n to Qbi_pass[n]
#             if V_mob is not None:
#                 elapsed_time = np.zeros(n_classes) #it elapsed time is 0
#                 provenance = n
#                 Qbi_pass[n].append(Cascade(provenance, elapsed_time, V_mob))
#                 # We only consider Vmob of the reach if this option is False.
#                 if consider_overtaking_sed_in_outputs == False:
#                     Qbi_mob[t][V_mob[:,0].astype(int), n, :] += np.float32(V_mob[:, 1:])
                      
#             # Store all the Qbi_pass[n] cascades in Qbi_mob of reach n
#             if consider_overtaking_sed_in_outputs == True:
#                 for cascade in Qbi_pass[n]:
#                     Vm = cascade.volume
#                     Qbi_mob[t][Vm[:,0].astype(int), n, :] += np.float32(Vm[:, 1:])
            
#             # Add the to_be_deposited volume from passing sediment to the V_dep
#             if to_be_deposited is not None:
#                 V_dep = np.concatenate([V_dep, to_be_deposited], axis=0)
                              
#             # Update Qbi_dep_0 to be used for the next time step.
#             Qbi_dep_0[n] = np.float32(V_dep)
#             # Remove empty rows from Qbi_dep_0
#             Qbi_dep_0[n] = Qbi_dep_0[n][np.sum(Qbi_dep_0[n][:,1:],axis = 1)!= 0]
#             # If removing empty rows leaves only an empty matrix, 
#             # put an empty layer
#             if  (Qbi_dep_0[n]).size == 0 :
#                 Qbi_dep_0[n] = np.float32(np.append(n, np.zeros(n_classes)).reshape(1,-1))
                
   
#             #---- 4) Compute the changes in bed elevation
#             # Modify bed elevation according to increased deposit
#             Delta_V = np.sum(Qbi_dep_0[n][:,1:]) -  np.sum(Qbi_dep_old[n][:,1:])
#             # Record Delta_V
#             Delta_V_all[t,n] = Delta_V 
#             # And Delta_V per class
#             Delta_V_class = np.sum(Qbi_dep_0[n][:,1:], axis=0) - np.sum(Qbi_dep_old[n][:,1:], axis=0)
#             Delta_V_class_all[t,n,:] = Delta_V_class            
            
#             # Update slope if required.
#             if update_slope == True:
#                 node_el[t+1][n]= node_el[t,n] + Delta_V/( np.sum(reach_data.Wac[np.append(n, network['upstream_node'][n])] * reach_data.length[np.append(n, network['Upstream_Node'][n])]) * (1-phi) )
            
#         """End of the reach loop"""
            
#         # Compute Qout for this time step.
#         outlet = network['n_hier'][-1]
#         for cascade in Qbi_pass[outlet]:
#             Vm = cascade.volume
#             Q_out[t, [Vm[:,0].astype(int)], :] += Vm[:,1:]
            
#         #Save Qbi_dep according to saving frequency
#         if save_dep_layer == 'always':
#             Qbi_dep[t+1] = copy.deepcopy(Qbi_dep_0)            
#         if save_dep_layer == 'yearly':
#             if int(t+2) % 365 == 0 and t != 0:
#                 t_y = int((t+2)/365)
#                 Qbi_dep[t_y] = copy.deepcopy(Qbi_dep_0)
                
        # #---5) Move the mobilized volumes to the destination reaches according to the sediment velocity

        # for n in network['n_hier']:
        #     #load mobilized volume for reach n
            
        #     V_mob = np.zeros((n_reaches,n_classes+1))
        #     V_mob[:,0] = np.arange(n_reaches)
        #     V_mob[:,1:n_classes+1] = np.squeeze(Qbi_mob[t,:,[n],:], axis = 0)
        #     V_mob = matrix_compact(V_mob)
            
        #     # # OLD: calculate GSD of mobilized volume
        #     # Fi_mob = (np.sum(V_mob[:,1:],axis = 0)/np.sum(V_mob[:,1:]))[:,None] # EB: must be a column vector
        #     # if np.isnan(Fi_mob).any():
        #     #     Fi_mob = Fi_r_act[t,:,n]
                
        #     # #OLD: calculate sediment velocity for the mobilized volume in each reach
              # v_sed = sed_velocity( np.matlib.repmat(Fi_mob, 1, n_reaches), slope[t,:] , Q.iloc[t,:], reach_data.Wac , v , h ,psi,  minvel , phi , indx_tr_cap, indx_partition, indx_velocity )
            
        #     #transfer the sediment volume downstream according to vsed in m/day
        #     Qbi_tr_t, Q_out_t, setplace, setout = sed_transfer_simple(V_mob , n , v_sed * ts_length , reach_data.length, network, psi)

        #     # Sum the volumes transported from reach n with all the other 
        #     # volumes mobilized by all the other reaches at time
        #     Qbi_tr[t+1] = Qbi_tr[t+1] + np.float32(Qbi_tr_t)
        #     Q_out[t] =  Q_out[t] + Q_out_t
            
        # # store vsed per class and per reach, of this day, in m/day
        # V_sed[t] = v_sed * ts_length
            
        # del Qbi_tr_t,Q_out_t
        

        # #in case of changing slope..
        # if update_slope == True:
        #     #..change the slope accordingly to the bed elevation
        #     slope[t+1,:], node_el[t+1,:] = change_slope(node_el[t+1,:] ,reach_data.length, network, s = min_slope)
            
        #measure time of routing
        #time2   = clock;

        #if np.remainder(10, t) == 0:  # save time only at certain timesteps 
        #   timerout = etime(time2, time1);
        
        # """end of the time loop"""
        

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
    df.to_csv('Refactoring_test_file_AnneLaure.txt')
    
    #set all NaN transport capacity to 0
    tr_cap_sum[np.isnan(tr_cap_sum)] = 0 
    
    #set all NaN active layer D50 to 0; 
    D50_AL[np.isnan(D50_AL)] = 0
    
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
                   'D50 active layer [m]': D50_AL,
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


















