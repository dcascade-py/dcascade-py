
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:21:34 2022

Input that are required in the ReachData class which define your river network:
- reach FromN - ToN (From Node - To Node) which define the relation between
  reaches (from upstream to downstream), these must be ordered from the smaller
  to the greater (e.g. first reach Id = 0, fromN = 1, ToN = 2)
- el_FN and el_TN (elevation fromN and ToN)
- Length, Wac (active channel width) in meters and Slope of the reach
- deposit = initial deposit layer expressed in m3/m2 - this value will be then
  multiplied by the reach width and length 
- D16, D50, D84 diameters expressed in [m] - will define the diameter distributions
  of the sediments present in the reach at t = 0 (i.e. of the deposit)
- Q = initial water discharge per reach in [m3/s]
- n = Manning coefficient for the calculation of the flow velocity 


Then you will also need a Dataframe which provides the water discharge per reach per time step: 
    rows = timestep
    columns = reaches 

Optional: you can provide external sediment sources per timestep, per reach and
per class of sediments. This variable is defined by Qbi_input 

This script was adapted from the Matlab version by Marco Tangi 

@author: Elisa Bozzolan
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add source (src) folder in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from GSD_curvefit import GSDcurvefit
from main import DCASCADE_main
from preprocessing import extract_Q, graph_preprocessing, read_network
from reach_data import ReachData
from widget import read_user_input
from plot_function import dynamic_plot


def DCASCADE_run(filename_river_network, filename_q, path_results, timescale, sed_range, n_classes, indx_tr_cap , indx_tr_partition, al_depth = 0.3, vel_height = '2D90'):
                   
    #---Option to save extended outputs or not
    save_extended = True
    dynamic_display = False
            
    #---Length of timestep in seconds - 60*60*24 = daily; 60*60 = hourly        
    ts_length = 60 * 60 * 24    
    
    #---Initial deposit layer [m]
    deposit_layer = 100000

    #---Transport capacity formula
    if indx_tr_cap == 3:
      formula = "Engelund and Hansen 1967"
      if indx_tr_partition == 1:
        partition = "Direct"
      if indx_tr_partition == 2:
        partition = "Bed material fraction (BMF)"
      if indx_tr_partition == 3:
        partition = "Transport capacity function (TCF)"
      if indx_tr_partition == 4:
        raise Exception("You can not use this partitioning with this transport formula")
     
    if indx_tr_cap == 6:
      formula = "Ackers and White 1990"
      if indx_tr_partition == 1:
        partition = "Direct"
      if indx_tr_partition == 2:
        partition = "Bed material fraction (BMF)"
      if indx_tr_partition == 3:
        partition = "Transport capacity function (TCF)"
      if indx_tr_partition == 4:
        raise Exception("You can not use this partitioning with this transport formula")
            
    if indx_tr_cap == 2:
      formula = "Wilkock and Crowe 2003"
      if indx_tr_partition == 4:
        partition = "Shear stress correction approach"
      else:
        raise Exception("You can not use this partitioning with this transport formula")
        
    if indx_tr_cap != 3 and indx_tr_cap != 2 and indx_tr_cap != 6:
      raise Exception("Other formulations are not implemented yet")
    print("Transport formula: " +str(formula)+"; Partitioning: "+str(partition)) 
                        
    #---Storing Deposit layer
    save_dep_layer = 'never' # 'yearly', 'always', 'never'.  Choose to save or not, the entire time deposit matrix
    
    
    #-------------------2) List of optional defined parameters of the simulation
    # These parameter are setted by default in the model with the following values
    # But they can also be changed by the user
    
    # eros_max = 1                  # Maximum depth that can be eroded in one time step from the reach, in meters. 
                                    # It is by default equal to the active layer, but can be larger for some case study 
    
    # al_depth_method = 1           # method to count the al_depth, 1: from the reach deposit layer top, the possible passing through cascade are then added at the top
                                    #                               2: from the top, including possible passing cascades. In this case, al_depth and eros_max, even if they are equal
                                    #                                   do not include the same layers
                                        
    # vel_height = '2D90'           # Section height for velocity calculation. 
                                    # Options: '2D90', '0.1_hw' (10% of water height), or any fixed value)
                                    
    # indx_flo_depth = 1            # Index for the flow calculation, default 1 = Manning 
                                    # (alternatives where developed for accounting for mountain stream roughness)
    
    
    # indx_velocity = 2             # method for calculating velocity (1: computed on each cascade individually, 2: on whole active layer)
    # indx_vel_partition = 1        # velocity section partitionning (1: same velocity for all classes, 2: section shared equally for all classes)
    
    
    # indx_slope_red = 1            # Slope reduction index, default 1 = None 
                                    # (alternatives where developed for accounting for mountain stream roughness)
    
    # indx_width_calc = 1           # Index for varying the width, default None
    
    # update_slope = False          # if False: slope is constant, if True, slope changes according to sediment deposit
    
    # roundpar = 0 # mimimum volume to be considered for mobilization of subcascade (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)
    
            
            
    ################ MAIN ###############
    # If the transport capacity formula is not chosen manually:
    #if 'indx_tr_cap' not in globals() or 'indx_tr_partition' not in globals():
        #indx_tr_cap, indx_tr_partition = read_user_input()
    
    # Read the network 
    reach_data_df = read_network(filename_river_network)
    reach_data = ReachData(reach_data_df)
    
    # Define the initial deposit layer per each reach in [m3/m]
    reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
    
    
    # # Set reach deposit to 0 except sources
    # not_source_idx = np.where(np.isin(ReachData.from_n, np.arange(3, 44+1, 1)))
    # ReachData.deposit[not_source_idx] = 0
    
    # Read/define the water discharge  
    Q = extract_Q(filename_q)
    
    # Sort ReachData according to the from_n, and organise the Q file accordingly
    sorted_indices = reach_data.sort_values_by(reach_data.from_n)
    Q_new = np.zeros((Q.shape))
    for i, idx in enumerate(sorted_indices): 
        Q_new[:,i] = Q.iloc[:,idx]
    Q = Q_new
            
    
    # Extract network properties
    network = graph_preprocessing(reach_data)
    
    # Sediment classes defined in Krumbein phi (φ) scale   
    psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)
    
    # Sediment classes in mm
    dmi = 2**(-psi).reshape(-1,1)
    
    # Check requirements. Classes must be compatible with D16, D50, D84 defined for the reaches - i.e. max sed class cannot be lower than D16
    print(min(reach_data.D16) * 1000, ' must be greater than ', np.percentile(dmi, 10, method='midpoint'))
    print(max(reach_data.D84) * 1000, ' must be lower than ',  np.percentile(dmi, 90, method='midpoint'))
       
    
    # External sediment for all reaches, all classes and all timesteps 
    # external_inputs = np.zeros((timescale, reach_data.n_reaches, n_classes))
    # external_inputs = pd.read_pickle(open(filename_ext , "rb"))
    # force_pass_external_inputs = True
    
    # Define input sediment load in the deposit layer
    deposit = reach_data.deposit * reach_data.length * reach_data.wac
    
    
    # Modify some tributaries GSD
    # Puting Ticino back to its old GSD
    FromN_Ticino = 51
    reach_data.D16[FromN_Ticino - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Ticino, 'D16_old']
    reach_data.D50[FromN_Ticino - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Ticino, 'D50_old']
    reach_data.D84[FromN_Ticino - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Ticino, 'D84_old']
    
    # Puting Lambro back to its old GSD
    FromN_Lambro = 53
    reach_data.D16[FromN_Lambro - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Lambro, 'D16_old']
    reach_data.D50[FromN_Lambro - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Lambro, 'D50_old']
    reach_data.D84[FromN_Lambro - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Lambro, 'D84_old']
    
    # Puting Nure back to its old GSD
    FromN_Nure = 55
    reach_data.D16[FromN_Nure - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Nure, 'D16_old']
    reach_data.D50[FromN_Nure - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Nure, 'D50_old']
    reach_data.D84[FromN_Nure - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Nure, 'D84_old']
    
    
    # Puting Dora Baltea back to its old GSD
    FromN_DB = 47
    reach_data.D16[FromN_DB - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_DB, 'D16_old']
    reach_data.D50[FromN_DB - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_DB, 'D50_old']
    reach_data.D84[FromN_DB - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_DB, 'D84_old']
    
    # Puting Sesia back to its old GSD
    FromN_Sesia = 48
    reach_data.D16[FromN_Sesia - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Sesia, 'D16_old']
    reach_data.D50[FromN_Sesia - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Sesia, 'D50_old']
    reach_data.D84[FromN_Sesia - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Sesia, 'D84_old']
    
    # Puting Tanaro back to its old GSD
    FromN_Tanaro = 49
    reach_data.D16[FromN_Tanaro - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Tanaro, 'D16_old']
    reach_data.D50[FromN_Tanaro - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Tanaro, 'D50_old']
    reach_data.D84[FromN_Tanaro - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Tanaro, 'D84_old']
    
    # Puting Scrivia back to its old GSD
    FromN_Scrivia = 50
    reach_data.D16[FromN_Scrivia - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Scrivia, 'D16_old']
    reach_data.D50[FromN_Scrivia - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Scrivia, 'D50_old']
    reach_data.D84[FromN_Scrivia - 1] = reach_data_df.loc[reach_data_df['FromN'] == FromN_Scrivia, 'D84_old']
    
    
    
    # # Putting all tributary GSD back to the old values
    # trib_from_n_list = [i for i in range(45, 65)]
    # for tr_fn in trib_from_n_list:
    #     reach_data.D16[tr_fn - 1] = reach_data_df.loc[reach_data_df['FromN'] == tr_fn, 'D16_old']
    #     reach_data.D50[tr_fn - 1] = reach_data_df.loc[reach_data_df['FromN'] == tr_fn, 'D50_old']
    #     reach_data.D84[tr_fn - 1] = reach_data_df.loc[reach_data_df['FromN'] == tr_fn, 'D84_old']
        
    
    # Define initial sediment fractions per class in each reaches, using a Rosin distribution
    Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi)
    
    # path_fir = Path("../../03-Po_case_16y/57-5y_2008-2012_DEM") / f"figures_{name}_{slope_name}_BF"
    # name_file_fir = path_fir / f"{name}_{slope_name}_BF.p"
    # Fi_r = np.load(path_fir / 'Fi_al_end.npy')
    
    # np.save('Fi_r_init.npy', Fi_r)
    
    # # Add sand in Fir of Dora Baltea, Sesia, Tanaro, Scrivia 
    # FromN_tribs = [47, 48, 49, 50]
    # Rs_Vezzoli = [0.3, 0.8, 0.35, 0.7]
    # Rs_tuned = Rs_Vezzoli#[0.06, 0.5, 0.3, 0.4]
    # for FromN in FromN_tribs:
    #     idx_trib = FromN_tribs.index(FromN)    
    #     Fi_r[FromN-1,7:] = Rs_tuned[idx_trib]/len(Fi_r[FromN-1,7:])
    #     # Make sure that the non sand make 1-Rs in total
    #     sum_non_sand = np.sum(Fi_r[FromN-1,:7])
    #     ratio = (1 - Rs_tuned[idx_trib])/sum_non_sand
    #     Fi_r[FromN-1,:7] = Fi_r[FromN-1,:7] * ratio
        
    #     #check sum makes 1
    #     print(np.sum(Fi_r[FromN-1,:]))
        
    # np.save('Fi_r_tuned.npy', Fi_r)
    
        
    # Initialise deposit layer 
    Qbi_dep_in = np.zeros((reach_data.n_reaches, 1, n_classes))
    for n in range(reach_data.n_reaches):
        Qbi_dep_in[n] = deposit[n] * Fi_r[n,:]
        
    
    # Prepare optionnal paramaters (possibly not given by the user) for calling the DCASCADE_main function
    kwargs = {}
    
    if 'eros_max' in globals():
        kwargs['eros_max'] = globals().get('eros_max')
        
    if 'al_depth_method' in globals():
        kwargs['al_depth_method'] = globals().get('al_depth_method')
    
    if 'vel_height' in globals():
        kwargs['vel_height'] = globals().get('vel_height')
        
    if 'indx_flo_depth' in globals():
        kwargs['indx_flo_depth'] = globals().get('indx_flo_depth')
        
    if 'indx_velocity' in globals():
        kwargs['indx_velocity'] = globals().get('indx_velocity')
        
    if 'indx_vel_partition' in globals():
        kwargs['indx_vel_partition'] = globals().get('indx_vel_partition')
        
    if 'indx_slope_red' in globals():
        kwargs['indx_slope_red'] = globals().get('indx_slope_red')
        
    if 'indx_width_calc' in globals():
        kwargs['indx_width_calc'] = globals().get('indx_width_calc')
        
    if 'update_slope' in globals():
        kwargs['update_slope'] = globals().get('update_slope')
        
    if 'roundpar' in globals():
        kwargs['roundpar'] = globals().get('roundpar')
        
    if 'save_dep_layer' in globals():
        kwargs['save_dep_layer'] = globals().get('save_dep_layer')
    
    if 'external_inputs' in globals():
        kwargs['external_inputs'] = globals().get('external_inputs')
    
    if 'force_pass_external_inputs' in globals():
        kwargs['force_pass_external_inputs'] = globals().get('force_pass_external_inputs')
    
    
    # Call dcascade main
    data_output, extended_output = DCASCADE_main(reach_data, network, Q, psi, timescale, ts_length, al_depth,
                                                 indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 **kwargs)
                                                    
    
    # Save results as pickled files
    import pickle
    
    if not os.path.exists(path_results):   
        os.makedirs(path_results)          
    

    pickle.dump(data_output, open(name_file , "wb"))  # save it into a file named save.p
    
    if save_extended: 
        pickle.dump(extended_output , open(name_file_ext , "wb"))  # save it into a file named save.p
        
    # Plot results
    if dynamic_display:
        keep_slider = dynamic_plot(data_output, reach_data_df)
        
        
