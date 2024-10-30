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
        self.maxwac = geodataframe['maxWac'].astype(float).values
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
        self.width_a = geodataframe['a'].astype(float).values
        self.width_b = geodataframe['b'].astype(float).values
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
            _, Vdep_active, _, _ = layer_search(reach_Vdep, active_layer_volume,
                                    roundpar, Qbi_incoming = volume_all_cascades)
            volume_all_cascades = np.concatenate([volume_all_cascades, Vdep_active], axis=0) 

        velocities = volume_velocities(volume_all_cascades, indx_velocity_partitioning, 
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
        # DD: to be improved, the section Svel should be divided by the number of non zeros in tr_cap_per_s

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
    # Note: in the deposit layer matrix, the uppermost layers are the last
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
                  update_slope, eros_max, save_dep_layer, ts_length,vary_width,vary_roughness,
                  consider_overtaking_sed_in_outputs = True,
                  compare_with_tr_cap = True, time_lag_for_mobilised = True):
    
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
    #JR additions
    vary_width
    vary_roughness
    
    OUTPUT: 
    data_output      = struct collecting the main aggregated output matrices 
    extended_output  = struct collecting the raw D-CASCADE output datasets
    """
    
    hypsolayers = True #move this to calling options when ready, ensure it works false!
    ################### Fixed parameters
    phi = 0.4 # sediment porosity in the maximum active layer
    minvel = 0.0000001
    outlet = network['n_hier'][-1] #outlet reach ID identification
    n_reaches = reach_data.n_reaches
    n_classes = len(psi)
    n_layers = Qbi_dep_in.shape[1] 
    sand_indices = np.where(psi > -1)[0]
    
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
    if save_dep_layer=='monthhour':
        dep_save_number = int(timescale/720)+1 #+1 because we also keep t0.        
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
    T_record_days = np.zeros((timescale)) #time storage, days #ccJR
    
    tr_cap_all = np.zeros((timescale, n_reaches, n_classes)) #transport capacity per each sediment class
    tr_cap_sum = np.zeros((timescale, n_reaches)) #total transport capacity 

    Qc_class_all = np.zeros((timescale, n_reaches, n_classes))
    flow_depth = np.zeros((timescale, n_reaches)) 
    
    Delta_V_all = np.zeros((timescale, n_reaches)) # reach mass balance (volumes eroded or deposited)
    Delta_V_class_all = np.zeros((timescale, n_reaches, n_classes))
    
    Qbi_FiLayers = np.zeros((dep_save_number,n_reaches,n_layers,n_classes))
    
    # In case of constant slope
    if update_slope == False:
        slope[:,:] = slope[0,:]
        node_el[:,: ] = node_el[0,:]
    
    # Initialise sediment deposit in the reaches 
    #ccJR - the first part is integer 'n' which is reach. 
    #Qbi_dep_0 = [np.expand_dims(np.zeros(n_classes+1, dtype=numpy.float32), axis = 0) for _ in range(n_reaches)]
   
    Qbi_dep_0 = [np.zeros((n_layers, n_classes + 1), dtype=np.float32) for _ in range(n_reaches)]  # Assuming 4 layers

    for n in network['n_hier']:  
        # if no inputs are defined, initialize deposit layer with a single cascade with no volume and GSD equal to 0
        q_bin = np.array(Qbi_dep_in[n])
        if not q_bin.any(): #if all zeros 
            Qbi_dep_0[n] = np.hstack((n, np.zeros(n_classes))).reshape(1,-1)
        else:           
            
            for nl in range(n_layers):
                Qbi_dep_0[n][nl] = np.float32(np.hstack((n, Qbi_dep_in[n, nl]))).reshape(1,-1)
                Fi_r_act[0,n,:] = np.sum(q_bin, axis=0) / np.sum(q_bin)
                D50_AL[0,n] = D_finder(Fi_r_act[0,n,:], 50, psi)

    #can I initialize a layer system? Will it be destroyed? How are cascades deposited, and can I put them at the 'bottom' or 'middle'?
    #layercake
    
    Qbi_dep[0] = copy.deepcopy(Qbi_dep_0) #store init condition of dep layer

    # Set maximum volume in meters that can be eroded for each reach, for each time step.
    eros_max_all = np.ones((1, n_reaches)) * eros_max
    eros_max_vol = np.round(eros_max_all * reach_data.maxwac * reach_data.length, roundpar)

    # Set active layer volume, the one used for calculating the tr_cap in [m3/s]
    # corresponds to the depth that the river can see every second (more like a continuum carpet ...)  
    # defined here as 2.D90 [Parker 2008]
    al_vol_all = np.zeros((timescale, n_reaches)) #store the volumes
    Vfracsave = np.zeros((timescale, n_reaches)) #store the volumes
    al_depth_all = np.zeros((timescale, n_reaches)) #store also the depths 
    # We take the input D90, or if not provided, the D84:
    if ~np.isnan(reach_data.D90):
        reference_d = reach_data.D90
    else:
        reference_d = reach_data.D84
    for n in network['n_hier']:
        al_depth = np.maximum(2*reference_d[n], 0.01)
        al_vol = al_depth * reach_data.maxwac[n] * reach_data.length[n]
        #al_vol_all[:,n] = np.repeat(al_vol, timescale, axis=0) #calculating in time loop. 
        al_depth_all[:,n] = np.repeat(al_depth, timescale, axis=0)
    if vary_width:
        wacsave = np.zeros((timescale, n_reaches)) 
    # start waiting bar    
    for t in tqdm(range(timescale-1)):
        T_record_days[t]=t*ts_length / (60*60*24);
        
    
        if vary_width: #ccJR carying width with hydraulic geometry. Replace this with hypsometric hydraulics next. 
            wacsave[t,:] =  reach_data.width_a[:] * Q[t,:]**reach_data.width_b[:]
            #this is a good place to raplace Q[t,n] with a 'single' Q to test the old constant Wac
            reach_data.wac[:] = wacsave[t,:]        
        
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
                               

            al_vol_all[t,n] = reach_data.wac[n] * reach_data.length[n] * np.maximum(2*reach_data.D84[n], 0.01) 

            #diag stuff
            jrdiag = False
            if jrdiag:
                al_vol_all[t,n] / Qbi_dep_0[n].sum()  #AL fraction of remainder
                Qbi_dep_0[n].sum() / reach_data.length[n] / reach_data.maxwac[n]  #depth of sediment in meters, total
                
                for nl in range(n_layers*10):  #this will loop in increments. could go finer. and could turn into the whole hypso transport curve.
                    slicevol = 0.05*(nl+1) * reach_data.wac[n]* reach_data.length[n]
                    _,V_dep_init,V_dep_overburden, Fi_slice = layer_search(Qbi_dep_old[n], slicevol, roundpar)
                    D50_custom = D_finder(Fi_slice, 50, psi) 
                    #print(1000*D50_custom, (Fi_slice[5:7].sum()*100).round(3) ) #great - the sand is at the top,
                    #V_dep_init has the bed 'under' hypso_V_above. V_dep_overburden has what we need to add at the end. 
                    V_dep_init.sum() / Qbi_dep_0[n].sum()
                    
            #ccJR hypso - REMOVE 'overburden' which is volume of (wacmax - wac) for now. 
            if hypsolayers:
                hypso_V_above = (reach_data.maxwac[n] - reach_data.wac[n])* reach_data.length[n] * 2.0 #original bed depth 2m. this controls the range; do I want to b
                Vfracsave[t,n] = hypso_V_above / Qbi_dep_0[n].sum() #fraction we are removing.
            
                #strip off overburden here:
                _,V_dep_init,V_dep_overburden, Fi_slice = layer_search(Qbi_dep_old[n], hypso_V_above, roundpar)
            else:
                # Extracts the deposit layer left in previous time step          
                V_dep_init = Qbi_dep_old[n] # extract the deposit layer of the reach 
            
            #ccJR - add our inputs to the bed, for the next timestep to deal with. it will at least be available and mass conserving..
            #V_dep_init[0, 1:] += Qbi_input[t,n,:] #didn't work
            if t>1 and Qbi_input[t,n,:].sum()>0:
                # The mobilised cascade is added to a temporary container
                et0 = np.ones(n_classes) # its elapsed time is 0
                #how it's done:                 mobilized_cascades.append(Cascade(provenance, elapsed_time, V_mob))
                arrgh = Qbi_input[t, n, :] #what's in the [0]? 1-7 are grain sizes. 
                Qtoadd =  np.insert(arrgh, 0, 0)
                Qtoadd = Qtoadd.reshape(1, -1)
                Qbi_pass[n].append(Cascade(n, et0, Qtoadd)) #ccJR adding source volume here. 
            
                
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
                # hVel = coef_AL_vel * h                
                hVel = al_depth_all[t,n]  
                    
                velocities = compute_cascades_velocities(Qbi_pass[n], 
                                           indx_velocity, indx_velocity_partition, hVel,
                                           indx_tr_cap, indx_partition,
                                           reach_data.wac[n], slope[t,n], Q[t,n], v[n], h[n],
                                           phi, minvel, psi,
                                           V_dep_init, al_vol_all[t,n],
                                           roundpar)
                # Store velocities
                V_sed[t, n, :] = velocities * ts_length
                #ccJR test moving sand much faster in prep for a suspended sand routine. 
                #V_sed[t, n, psi > - 1] = v[n] * ts_length * 0.1 #moving at 0.1 of the water velocity. 
            # Decides whether cascades, or parts of cascades, 
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
                    # reach before the possible cascades arrive.
                    # At the exception that no cascades arrive at the outlet.
                    if Qbi_pass[n] != []:
                        time_lag = np.zeros(n_classes)
                    else: 
                        # If no cascades arrive at the outlet,
                        # we mobilise from the reach itself
                        time_lag = np.ones(n_classes)
            else:
                # in this condition (compare_with_tr_cap = False), 
                # we always mobilise from the reach itself and 
                # the passing cascades are passing the outlet, without 
                # checking the energy available to make them pass,
                # like in version 1 of the code
                time_lag = np.ones(n_classes)                      
                           
            # In the case time_lag is not all zeros, we mobilise from the reach
            # before the cascades arrive, proportionally to this time_lag
            mobilized_cascades = []
            if np.any(time_lag != 0):
                # Get the sediment fraction and D50 in the active layer. 
                # DD: to be discussed if we keep the active layer (2D90) or a bigger active layer,
                # since we don't consider any incoming sediment now                          
                _,_,_, Fi_r_act[t,n,:] = layer_search(V_dep_init, al_vol_all[t,n], roundpar)
                
                
                #ccJR can I work here to get different Fi at different 'depths'? YES.
                #a small active layer volume returns the material on top - basically sand.
                #I can split up the bed by active layer 'hypso position' using incremental wac returned by hypsometry
                #the first 'cascade' in n=4 has index 4 and the most volume - that's it's own-sourced volume
                debugJR = False
                if n==4 and t>50 and debugJR:
             #       print(V_dep_init.shape)
                    foo_v = 50
                    
                    V_inc2act,V_dep2act,V_dep, Fi_custom = layer_search(V_dep_init, foo_v, roundpar)
                    D50_custom = D_finder(Fi_custom, 50, psi) 
                    print(1000*D50_custom)
                  
                    #if I run this many times, I dig down V_dep to the sublayer.:
                    V_inc2act,V_dep2act,V_dep, Fi_custom = layer_search(V_dep, foo_v, roundpar)
                    D50_custom = D_finder(Fi_custom, 50, psi) 
                    print(1000*D50_custom)
                    
                    
                
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
                    
                # Total volume possibly mobilised during the time lag (per class)
                time_for_mobilising = time_lag * ts_length
                volume_mobilisable = tr_cap_per_s * time_for_mobilising
                
                
                # Erosion maximum during the time lag 
                # (we take the mean time lag among the classes)
                eros_max_vol_t = eros_max_vol[0,n] * np.mean(time_lag)
                
                #ccJR attempting to limit the channel's access to sand.
                sand_access_mod = reach_data.wac[n]**2 / reach_data.maxwac[n]**2 #fraction of maxwac as availability modifier. more powerful - squared ratio. CUT small channels access to sand
                #sand_access_mod = reach_data.wac[n] / reach_data.maxwac[n] #fraction of maxwac as availability modifier. more powerful - squared ratio. CUT small channels access to sand
                #eros_max_vol_t[sand_indices] = sand_access_mod * eros_max_vol_t[sand_indices] %this failed - does not have dimension for psi
                
                #ccJR if I modify V_dep_init here it will kill mass. if I move thingsa after layer search from EL to not EL
                
                # Mobilise from the reach layers
                V_inc_EL, V_dep_EL, V_dep_not_EL, _ = layer_search(V_dep_init, eros_max_vol_t, roundpar)
                
                #here I can move deposited sand that layer_search found active, into inactive. currently by a width ratio- next by hypsometric active.
                #ahhh, but there are multiple cascades in V_dep_EL. 
                #V_dep_not_EL[0,sand_indices+1] += V_dep_EL[0,sand_indices+1] * (1-sand_access_mod)
                #then remove same from active. This MAY retain sand on the bed until width increases enough to access it. 
                #V_dep_EL[0,sand_indices+1] *= sand_access_mod
                #print(V_dep_EL.shape)
                
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
                Q_pass_volume_per_s[:,1:] = Q_pass_volume_per_s[:,1:] / ts_length                                            
                _,_,_, Fi_r = layer_search(V_dep_after_tlag, al_vol_all[t,n], roundpar, Qbi_incoming = Q_pass_volume_per_s) 
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
                    
                    #ccJR attempting to limit the channel's access to sand.
                    #sand_access_mod = reach_data.wac[n]**2 / reach_data.maxwac[n]**2 #fraction of maxwac as availability modifier. more powerful - squared ratio. CUT small channels access to sand
                    
                    # Mobilise from the reach layers
                    V_inc_EL, V_dep_EL, V_dep_not_EL, _ = layer_search(V_dep_after_tlag, eros_max_vol_t, roundpar, Qbi_incoming = Q_pass_volume)
                    
                    # how to shield sand from erosion despite capacity? matrix_compact(Q_pass_volume)
                    #here I can move deposited sand that layer_search found active, into inactive. currently by a width ratio- next by hypsometric active.
                    #V_dep_not_EL[0,sand_indices+1] += V_dep_EL[0,sand_indices+1] * (1-sand_access_mod)
                    #then remove same from active
                    #V_dep_EL[0,sand_indices+1] *= sand_access_mod                    
                    #print(V_dep_EL.shape)
                    
                    [V_mob, V_dep_after_tlag] = tr_cap_deposit(V_inc_EL, V_dep_EL, V_dep_not_EL, diff_pos, roundpar)
                    
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
            
            #ccJR hypso: here if I split the bed at my active hypso position, I can reassemble the 'untouched overburden'
            if hypsolayers:
                Qbi_dep_0[n] = np.concatenate([np.float32(V_dep_final), V_dep_overburden], axis=0)
                #this puts overburden we sliced off the top (representing inactive width), back on top. 
            else:
                Qbi_dep_0[n] = np.float32(V_dep_final)
            # Finally, pass these cascades to the next reach (if we are not at the outlet)
            # Find reach downstream:   
            if n != int(network['outlet']):
                n_down = np.squeeze(network['downstream_node'][n], axis = 1) 
                n_down = int(n_down) # Note: This is wrong if there is more than 1 reach downstream (to consider later)
            else:
                n_down = None
            
            if n != int(network['outlet']):       
                Qbi_pass[n_down].extend(Qbi_pass[n]) 
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
                node_el[t+1][n]= node_el[t,n] + Delta_V/( np.sum(reach_data.maxwac[np.append(n, network['upstream_node'][n])] * reach_data.length[np.append(n, network['upstream_node'][n])]) * (1-phi) )
            
            
            vary_roughness_inloop = True 
            #update roughness within the time loop. each vector is length n_reaches
            if vary_roughness_inloop:
                #need updated D84 at least
                #reach_data.D16[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 16, psi)
                #reach_data.D50[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 50, psi)
                reach_data.D84[n] = D_finder(Fi_r_act[t,n,:], 84, psi)  
                
                Hbar = h[n]
                s8f_keul = (1 / 0.41) * np.log((12.14 * Hbar) / (3*reach_data.D84[n]))
                C_keul = s8f_keul * np.sqrt(9.81)
                n_keul = Hbar**(1/6) / C_keul
                reach_data.n[n] = n_keul            
        """End of the reach loop"""
        
        # Save Qbi_dep according to saving frequency
        if save_dep_layer == 'always':
            Qbi_dep[t+1] = copy.deepcopy(Qbi_dep_0)            
        if save_dep_layer == 'yearly':
            if int(t+2) % 365 == 0 and t != 0:
                t_y = int((t+2)/365)
                Qbi_dep[t_y] = copy.deepcopy(Qbi_dep_0)
        if save_dep_layer == 'monthhour':
            if int(t+2) % 720 == 0 and t != 0:
                t_y = int((t+2)/720)
                Qbi_dep[t_y] = copy.deepcopy(Qbi_dep_0)
                #save stratigraphy of Fi, by using layer_search.
                #think really hard about which way is up. layer_search works from the top. nl 0 is the top then?
                if hypsolayers:
                    for n in network['n_hier']:            
                        for nl in range(n_layers):  #this will loop in increments. could go finer. and could turn into the whole hypso transport curve.
                            slicevol = (nl+1)* (Qbi_dep_0[n].sum() / n_layers)  #even slices
                            _,V_dep_init,V_dep_overburden, Fi_slice = layer_search(Qbi_dep_old[n], slicevol, roundpar)
                            D50_custom = D_finder(Fi_slice, 50, psi) 
                            #print(1000*D50_custom, (Fi_slice[5:7].sum()*100).round(3) ) #great - the sand is at the top,
                            Qbi_FiLayers[t_y,n,nl,:] = Fi_slice #might lose a bit at the bottom if major aggradation?

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
    
    # Fi_bed_slices = [np.empty((len(Qbi_dep), n_reaches, n_layers)) for _ in range(n_classes)]
#I need to get the dimensions right here; I want to save Fi at the end time. then dig down by slice to get Fi vs depth
    # for t in range(len(Qbi_dep)): 
    #     q_t = Qbi_dep[t]        
    #     for n in network['n_hier']:            
    #         for nl in range(n_layers):  #this will loop in increments. could go finer. and could turn into the whole hypso transport curve.
    #             slicevol = (nl+1)* (Qbi_dep[n].sum() / n_layers)  #even slices
    #             _,V_dep_init,V_dep_overburden, Fi_slice = layer_search(Qbi_dep_old[n], slicevol, roundpar)
    #             D50_custom = D_finder(Fi_slice, 50, psi) 
    #             Fi_bed_slices[t][n,nl,:]
    #             print(1000*D50_custom, (Fi_slice[5:7].sum()*100).round(3) ) #great - the sand is at the top,
        
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
    data_output = {
        'wac': wacsave,  # Channel Width [m] which is now [m] and shape [t,nreaches]
        'slope': slope,  # Reach slope
        'Q': Q[0:timescale, :],  # Discharge [m^3/s]
        'QB_mob_sum': QB_mob_sum,  # Mobilized [m^3]
        'tot_tranported': tot_tranported,  # Transported [m^3]
        'tot_sed': tot_sed,  # Transported + deposited [m^3]
        'D50_dep': D50_dep,  # D50 deposit layer [m]
        'D50_mob': D50_mob,  # D50 mobilised layer [m]
        'D50_AL': D50_AL,  # D50 active layer [m]
        'tr_cap_sum': tr_cap_sum,  # Transport capacity [m^3]
        'V_dep_sum': V_dep_sum,  # Deposit layer [m^3]
        'Delta_V_all': Delta_V_all,  # Delta deposit layer [m^3]
        'tot_sed_class': tot_sed_class,  # Transported + deposited - per class [m^3]
        'deposited_class': deposited_class,  # Deposited - per class [m^3]
        'mobilised_class': mobilised_class,  # Mobilised - per class [m^3]
        'transported_class': transported_class,  # Transported - per class [m^3]
        'Delta_V_class': Delta_V_class_all,  # Delta deposit layer - per class [m^3] #ccJR added _all for complete change array. 
        'tr_cap_class': tr_cap_class,  # Transport capacity - per class [m^3]
        'V_sed': V_sed,  # Sed_velocity [m/dt]
        'V_sed_class': V_sed_class,  # Sed_velocity - per class [m/day]
        'flow_depth': flow_depth,  # Flow depth
        'al_depth_all': al_depth_all,  # Active layer [m]
        'eros_max_all': eros_max_all,  # Maximum erosion layer [m]
        'Q_out': Q_out,  # Q_out [m^3]
        'Q_out_class': Q_out_class,  # Q_out_class [m^3]
        'outcum_tot': outcum_tot,  # Q_out_tot [m^3]
        'Qbi_FiLayers': Qbi_FiLayers, #Fi in layers.  
        }

    if indx_tr_cap == 7:
        data_output["Qc - per class"] = Qc_class
    dmi = 2**(-psi).reshape(-1,1)      
    #all other outputs are included in the extended_output cell variable 
    extended_output = {'Qbi_tr': Qbi_tr,
                       'Qbi_mob': Qbi_mob,
                       'Q_out': Q_out,
                       'Qbi_dep': Qbi_dep,
                       'Fi_r_ac': Fi_r_act,
                       'Fi_mob_t': Fi_mob_t,
                       'Node_el': node_el,
                       'Length': reach_data.length,
                       'D16': reach_data.D16,
                       'D50': reach_data.D50,
                       'D84': reach_data.D84,
                       'psi': psi,
                       'dmi': dmi,
                       'Tdays': T_record_days,
                       'Vfracsave': Vfracsave,
                       }
    
    return data_output, extended_output


















