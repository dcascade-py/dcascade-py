
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

from d_finder import D_finder


# List to loop over transport-capacity + partitionning formulas: 
#  - Wilcock and Crowes
#  - Ackers and White - Bed Material Fraction partitionning
#  - Ackers and White - Molinas rates partitionning

GSD_perc_list = [1.0] # [0.8, 0.9, 1.0, 1.1, 1.2]

transport_law = [(6, 2)]   #(6, 2),(2, 4) , (6, 3)]  
name_list = ['AW_BMF']         #'WC', , 'AW_Molinas']              

# List to loop over the two width scenarios:
#  - Bankfull
#  - Mean flow
# Note that, width is given externally from a csv file.

width_input_name_list = ['bf_width']#, 'smoothed_wat_w']          
width_output_name_list = ['BF_smooth']#, 'MF_smooth'] 

for GSD_perc in GSD_perc_list:
    for width_input_name, width_output_name in zip(width_input_name_list, width_output_name_list):
        for tl, name in zip(transport_law, name_list):
            
            '''user defined input data'''
                
            #-------River shape files 
            path_river_network = Path('../../03-Po_case_16y/Inputs/08-shp_with_slope_tributaries_updated/before_IS_slope_hydro') #lin reg DEM 1m and slope HECRAS
            name_river_network = 'Po_river_network.shp'
            filename_river_network = path_river_network / name_river_network
            
            #--------Discharge files
            path_q = Path('../inputs/Input_Po_case_PGS_2025/')
            # csv file that specifies the water flows in m3/s as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)
            name_q = 'Po_Qdaily_16y.csv'
            filename_q = path_q / name_q
            
            
            # #--------Path to the output folder
            # path_results = Path("../cascade_results/")
            # name_file = path_results / f"{name}_{width_output_name}.p"
            # name_file_ext = path_results / f"{name}_{width_output_name}_ext.p"
            
            #--------Path to the output folder
            path_results = Path("../cascade_results/")
            name_file = path_results / f"{name}_GSD_mult_{GSD_perc}.p"
            name_file_ext = path_results / f"{name}_GSD_mult_{GSD_perc}_ext.p"
            
            
            #--------Width path
            width_path = Path("../inputs/Input_Po_case_PGS_2025")
            name_width = width_path / 'Widths_dcascade.csv'
            name_width_type = width_input_name
        
            #---Option to save extended outputs or not
            save_extended = False
            dynamic_display = False
            
            
            #--------Parameters of the simulation
            
            #---Sediment classes definition 
            # defines the sediment sizes considered in the simulation
            #(must be compatible with D16, D50, D84 defined for the reach - i.e. max sed class cannot be lower than D16)
            sed_range = [-6, 2]     # range of sediment sizes - in Krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5). 
            n_classes = 12          # number of classes
            
            #---Timescale 
            timescale = 5843       # 5843 days, 1095, 1825
            ts_length = 60 * 60 * 24    # length of timestep in seconds - 60*60*24 = daily; 60*60 = hourly
            
            #---Transport capacity formula and partitioning
            indx_tr_cap = tl[0]                 # 2: Wilkock and Crowe 2003; 
                                            # 3: Engelund and Hansen; 
                                            # 6: Ackers and White
            
            indx_tr_partition = tl[1]           # 1: Direct calculation summing fractionnal load; 
                                            # 2: BMF: "Bed Material" Fraction weighting of fractionnal loads; 
                                            # 3: Molinas rates: weighting on total load
                                            # 4: Shear stress correction (only for Formula already partitionned, e.g., W&C)
            
            #---Initial layer sizes
            deposit_layer = 100000      # Initial deposit layer [m]. Warning: will overwrite the deposit column in the reach_data file
            al_depth = 0.3           # Active layer depth (Possibilities: '2D90', or any fixed value)
            
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
            if 'indx_tr_cap' not in globals() or 'indx_tr_partition' not in globals():
                indx_tr_cap, indx_tr_partition = read_user_input()
            
            # Read the network 
            reach_data_df = read_network(filename_river_network)
            reach_data = ReachData(reach_data_df)
            
            
            # Read/define the water discharge  
            Q = extract_Q(filename_q)
            
            # Sort ReachData according to the from_n, and organise the Q file accordingly
            sorted_indices = reach_data.sort_values_by(reach_data.from_n)
            Q_new = np.zeros((Q.shape))
            for i, idx in enumerate(sorted_indices): 
                Q_new[:,i] = Q.iloc[:,idx]
            Q = Q_new
            
            
            # Define the initial deposit layer per each reach in [m3/m]
            reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
            
            # # I want to put to 0 the reaches that are not sources
            # fromn_start = 4
            # fromn_end = 44
            # for FromN in range(fromn_start, fromn_end + 1):
            #     reach_data.deposit[FromN - 1] = 0
            
            
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
                
            
            # Multiplying all Po GSD + trib
            Po_from_n_list = [1]
            Po_from_n_list.extend(i for i in range(45, 65))
            for Po_fn in Po_from_n_list:
                reach_data.D16[Po_fn - 1] *= GSD_perc
                reach_data.D50[Po_fn - 1] *= GSD_perc
                reach_data.D84[Po_fn - 1] *= GSD_perc
                        
            # Put width from Sentinel 2 measurements (and smoothed)
            # This will replace the width in the reach data file (measured from orthophotos)
            widths = pd.read_csv(name_width)
            for FromN in range(2, 45):
                reach_data.wac[FromN - 1] = widths.loc[widths['FromN'] == FromN, name_width_type]
                
            
            # Define initial sediment fractions per class in each reaches, using a Rosin distribution
            Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi)
            
            # D50_init = D_finder(Fi_r, 50, psi)
            
            # df = pd.DataFrame()
            # df['D50_init'] = D50_init
            # df['D50_meas'] = reach_data.D50
            
            # df.to_csv('D50_check_init.csv')
            
            # import matplotlib.pyplot as plt
            
            # plt.plot(D50_init, '*', label = 'init')
            # plt.plot(reach_data.D50, 'v', label = 'measured')
            # plt.legend()
            # plt.show()
            
                
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
        
        
