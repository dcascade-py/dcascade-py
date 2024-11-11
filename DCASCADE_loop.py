# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:00:36 2022

This script contains the time-space loop which assess the sediment routing through the network 

This script was adapted from the Matlab version by Marco Tangi 

@author: Elisa Bozzolan
"""

""" Libraries to import """


from supporting_classes import ReachData, Cascade, DCASCADE

import itertools


           
""" MAIN FUNCTION SECTION """

def DCASCADE_main(reach_data, network, Q, Qbi_input, Qbi_dep_in, timescale, psi, roundpar, 
                  update_slope, eros_max, save_dep_layer, ts_length,
                  indx_tr_cap, indx_tr_partition, indx_flo_depth,                 
                  indx_velocity = 2, 
                  indx_vel_partition = 1,                   
                  indx_slope_red = 1,                  
                  passing_cascade_in_outputs = True,
                  passing_cascade_in_trcap = True, 
                  time_lag_for_mobilised = True):
    
    """
    Main function of the D-CASCADE software.
    
    INPUT :
    
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
    
    indx_tr_cap         = the index indicating the transport capacity formula
    indx_tr_partition   = the index indicating the type of sediment flux partitioning
    indx_flo_depth      = the index indicating the flow depth formula, default 1 is Manning
    

    
    OPTIONAL:
    indx_velocity       = the index indicating the method for calculating velocity (see compute_cascades_velocities)
    indx_vel_partition  = the index indicating the type of partitioning in the section used to compute velocity
    indx_slope_red      = the index indicating the slope reduction formula, default 1 is no reduction
    
    Options for the dcascade algorithme (if all False, we reproduce the version 1)
    passing_cascade_in_outputs          = Bool to activate or not this option (default True)
    passing_cascade_in_trcap            = Bool to activate or not this option (default True)
    time_lag_for_mobilised              = Bool to activate or not this option (default True)
    
    OUTPUT: 
    data_output      = struct collecting the main aggregated output matrices 
    extended_output  = struct collecting the raw D-CASCADE output datasets
    """
    
    dcascade = DCASCADE(timescale, ts_length, save_dep_layer, update_slope,                     
                       indx_flo_depth, indx_slope_red, 
                       indx_tr_cap, indx_tr_partition,
                       indx_velocity, indx_vel_partition,
                       passing_cascade_in_outputs,
                       passing_cascade_in_trcap, 
                       time_lag_for_mobilised)
    
    dcascade.initialize_sedim_system(phi=0.4, minvel=0.0000001, outlet=network['n_hier'][-1],
                                    n_reaches=reach_data.n_reaches, psi=psi)
    dcascade.initialize_slopes(reach_data)
    dcascade.initialize_elevations(reach_data)
    dcascade.initialize_sediment_variables()
    
    dcascade.set_sediment_deposit(network, Qbi_dep_in)
    dcascade.set_erosion_maximum(eros_max, reach_data, roundpar)
    dcascade.set_active_layer(reach_data, network)    
    
    dcascade.run(reach_data, network, Q, roundpar)
    
    data_output, extended_output = dcascade.output_processing(reach_data, Q)
    
    return data_output, extended_output


















