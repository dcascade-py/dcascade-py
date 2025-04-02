"""
Created on Mon Oct 10 18:00:36 2022

This script contains the time-space loop which assess the sediment routing through the network

This script was adapted from the Matlab version by Marco Tangi

@author: Elisa Bozzolan, Diane Doolaeghe, Anne-Laure Argentin
"""

""" Libraries to import """


import itertools

from DCASCADE_class import DCASCADE
from supporting_classes import Cascade, ReachData, SedimentarySystem

""" MAIN FUNCTION SECTION """

def DCASCADE_main(reach_data, network, Q, psi, timescale, ts_length, al_depth,
                  indx_tr_cap, indx_tr_partition, Qbi_dep_in,                     
                  save_dep_layer = 'always', 
                  eros_max = None,
                  vel_height = '2D90',
                  indx_flo_depth = 1,                  
                  indx_velocity = 2,
                  indx_vel_partition = 1,
                  indx_slope_red = 1,
                  indx_width_calc = 1,
                  update_slope = False,
                  roundpar = 0,
                  
                  external_inputs = None,
                  force_pass_external_inputs = False,
                  
                  passing_cascade_in_outputs = True,
                  passing_cascade_in_trcap = True,
                  time_lag_for_mobilised = False):
    
    
    DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length, al_depth,
                                                 indx_tr_cap , indx_tr_partition, Qbi_dep_in,
                                                 **kwargs)

    if eros_max is None:
        eros_max = al_depth
        
    """
    Main function of the D-CASCADE software.

    INPUT :

    reach_data          = nx1 Struct defining the features of the network reaches
    network             = 1x1 struct containing for each node info on upstream and downstream nodes
    Q                   = txn matrix reporting the discharge for each timestep
    external_input      = txnxn_c matrix. Per each reach and per each timestep is defined an external sediment input of a certain sediment class
    Qbi_dep_in          = sediment material volume available in reaches at the beginning of the simulation
                        (it could be that for the same reach id, there are two strata defined so two rows of the dataframe with the top row is the deepest strata)
    timescale           = length for the time horizion considered
    psi                 = sediment classes considered (from coarse to fine)
    roundpar            = mimimum volume to be considered for mobilization of subcascade
                         (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)
    update_slope        = bool to chose if we change slope trought time or not. If Flase, constant slope. If True, slope changes according to sediment deposit.
    eros_max            = maximum erosion depth per time step [m], is equal to the active layer depth by default
    save_dep_layer      = saving option of the deposit layer for each time step
    ts_length           = the length in seconds of the timestep (60*60*24 for daily timesteps)

    indx_tr_cap         = the index indicating the transport capacity formula
    indx_tr_partition   = the index indicating the type of sediment flux partitioning
    indx_flo_depth      = the index indicating the flow depth formula, default 1 is Manning



    OPTIONAL:
    indx_velocity       = the index indicating the method for calculating velocity (see compute_cascades_velocities)
    indx_vel_partition  = the index indicating the type of partitioning in the section used to compute velocity
    indx_slope_red      = the index indicating the slope reduction formula, default 1 is no reduction

    Options for the dcascade algorithm (if all False, we reproduce the version 1)
    passing_cascade_in_outputs          = if True, we consider ovepassing sediment in the output (Qbimob and Qbitr).
                                            But this does not change the way sediment move. (default True)
    passing_cascade_in_trcap            = If True, we now include present cascades from upstream + reach material
                                            in the transport capacity calculation, to check if they should pass or not. (default True)
    time_lag_for_mobilised              = option in progress (default False). If True, we consider a time lag between the beginning of the time step,
                                            and the arrival of the first cascade to the ToN of the reach, during which we are able to mobilise from the reach itself

    OUTPUT:
    data_output      = struct collecting the main aggregated output matrices
    extended_output  = struct collecting the raw D-CASCADE output datasets
    """

    # Create sedimentary system
    sedimentary_system = SedimentarySystem(reach_data, network, timescale, ts_length,
                                           save_dep_layer, update_slope, psi)
    sedimentary_system.initialize_slopes()
    sedimentary_system.initialize_widths(indx_width_calc)
    sedimentary_system.initialize_elevations()
    sedimentary_system.initialize_storing_matrices()
    sedimentary_system.set_sediment_initial_deposit(Qbi_dep_in)
    sedimentary_system.set_external_input(external_inputs, force_pass_external_inputs, roundpar)
    sedimentary_system.set_erosion_maximum(eros_max, roundpar)
    sedimentary_system.set_active_layer(al_depth)



    # Create DCASCADE solver
    dcascade = DCASCADE(sedimentary_system, indx_flo_depth, indx_slope_red, indx_width_calc)

    dcascade.set_transport_indexes(indx_tr_cap, indx_tr_partition)
    dcascade.set_velocity_options(indx_velocity, indx_vel_partition, vel_height)
    dcascade.set_algorithm_options(passing_cascade_in_outputs, passing_cascade_in_trcap,
                                   time_lag_for_mobilised)
    # Run
    dcascade.run(Q, roundpar)

    # Post process
    data_output, extended_output = dcascade.output_processing(Q)

    return data_output, extended_output


















