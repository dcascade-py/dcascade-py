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

'''user defined input data'''


# #-------River shape files
# path_river_network = Path('../inputs/input_solda_workshop_insbruck/')
# # path_river_network = Path('C:\\Users\\FPitscheider\\OneDrive - Scientific Network South Tyrol\Desktop\\Projects\\ALTROCLIMA\\Solda\\RN4Model\\')
# name_river_network = 'Solda_RN_Straightlines_ETRS89UTM_ok.shp'
# # name_river_network = 'Solda_RN_Straightlines_ETRS89UTM_ok_PS.shp' # Only PS reach - For test tr_cap
# filename_river_network = path_river_network / name_river_network

# #--------Discharge files
# path_q = Path('C:\\Users\\FPitscheider\\OneDrive - Scientific Network South Tyrol\Desktop\\Projects\\ALTROCLIMA\\Solda\\DischargeSims\\')
# # csv file that specifies the water flows in m3/s as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)
# # name_q = 'Sims_2014-22_v2\\discharges_allReaches_m3_per_s_noNode29.csv'
# # name_q = 'Sims_2014-22_v2\\discharges_allReaches_m3_per_s_noNode29_2015_22_x2.csv'
# #name_q = 'Sims_2014-22_v2\\discharges_allReaches_m3_per_s_noNode29_1Yinit.csv' # 365 days of initiation - same as 2014
# # name_q = 'Sims_2014-22_v2\\discharges_reach28_PS.csv' # Only PS reach - For test tr_cap
# name_q = 'discharge_2009_15_20.csv'
# # name_q = 'discharge_2009_15_20_PSonly.csv'
# # q_init = [
# #     # 'discharge_15_20.xlsx',                    # just 2015–2020
# #     # 'discharge_2014_15_20.xlsx',               # 1x 2015
# #     # 'discharge_2013_15_20.xlsx',               # 2x 2015
# #     # 'discharge_2012_15_20.xlsx',               # 1x 2016 + 2x 2015
# #     # 'discharge_2011_15_20.xlsx',
# #     # 'discharge_2010_15_20.xlsx',
# #     'discharge_2009_15_20.xlsx',
# #     # 'discharge_2008_15_20.xlsx',
# #     # 'discharge_2007_15_20.xlsx',
# #     # 'discharge_2006_15_20.xlsx',
# #     # 'discharge_2005_15_20.xlsx',
# #     # 'discharge_2004_15_20.xlsx',
# #     # 'discharge_2003_15_20.xlsx',
# #     # 'discharge_2002_15_20.xlsx',
# #     # 'discharge_2001_15_20.xlsx',
# #     # 'discharge_2000_15_20.xlsx'
# # ]
# # name_q = q_init


# path_q = Path('../inputs/input_solda_workshop_insbruck/')
# # csv file that specifies the water flows in m3/s as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)
# name_q = 'discharges_allReaches_m3_per_s_noNode29.csv'

# filename_q = path_q / name_q
# filename_q = path_q / name_q

# # #--------Path to the output folder
# # path_results = Path("C:\\Users\\FPitscheider\\OneDrive - Scientific Network South Tyrol\Desktop\\Projects\\ALTROCLIMA\\Solda\\dCascade_Results\\Simualtions\\2015_20_v1Apr25\\")

# #---Path to the output folder
# path_results = Path("../cascade_results/")
# name_file = path_results / 'save_all.p'


# #--------Parameters of the simulation

# #---Sediment classes definition
# # defines the sediment sizes considered in the simulation
# #(must be compatible with D16, D50, D84 defined for the reach - i.e. max sed class cannot be lower than D16)
# sed_range = [-10, -1]  # range of sediment sizes - in Krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5).
# n_classes = 10       # number of classes

# #---Timescale
# timescale = 365#len(name_q) #5844 #3287 # 3652
# # print(timescale)
# ts_length = 60 * 60 * 24 # length of timestep in seconds - 60*60*24 = daily; 60*60 = hourly

# #---Change slope or not
# update_slope = False # if False: slope is constant, if True, slope changes according to sediment deposit

# #---Initial layer sizes
# deposit_layer = 0.1     # Initial deposit layer [m]. Warning: will overwrite the deposit column in the reach_data file
# eros_max = 0.3               # Maximum depth (threshold) that can be eroded in one time step (here one day), in meters.
# al_depth = '2D90'              # Active layer depth (Possibilities: '2D90', or any fixed value)
# vel_height = '2D90'         # Section for velocity calculation
#                             #Possibilities: '2D90', '0.1_hw' (10% of water height), or any fixed value)

# #---Storing Deposit layer
# save_dep_layer = 'always' # 'yearly', 'always', 'never'.  Choose to save or not, the entire time deposit matrix

# #---Others
# roundpar = 0 # mimimum volume to be considered for mobilization of subcascade (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)


# indx_tr_cap = 7
# indx_tr_partition = 2


def DCASCADE_run(filename_river_network, filename_q, path_results, timescale, sed_range, n_classes, indx_tr_cap , indx_tr_partition, al_depth = 0.3, vel_height = '2D90'):
    
    # Parameters that I did not set in the colab, but we could:
    ts_length = 60 * 60 * 24
    deposit_layer = 0.1


    ################ MAIN ###############
    # # If the transport capacity formula is not chosen manually:
    # if 'indx_tr_cap' not in globals() or 'indx_tr_partition' not in globals():
    #     indx_tr_cap, indx_tr_partition = read_user_input()

    # Read the network
    reach_data_df = read_network(filename_river_network)
    reach_data = ReachData(reach_data_df)

    # Define the initial deposit layer per each reach in [m3/m]
    reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)

    # FromN for reaches with depositions
    deposit_nodes = np.array([
        # Solda
        1, 3, 4, 5, 7, 8,
        # # Trafoi
        29, 31, 32, 34, 36, 37,
    
        # for transport limited in PS
        # 28
    ])

    reach_data.deposit[deposit_nodes-1] = 100000

    reach_data.wac = reach_data.wac_bf

    # Read/define the water discharge
    Q = extract_Q(filename_q)

    # Sort reach_data according to the from_n, and organise the Q file accordingly
    sorted_indices = reach_data.sort_values_by(reach_data.from_n)
    Q_new = np.zeros(Q.shape)
    for i, idx in enumerate(sorted_indices):
        Q_new[:,i] = Q.iloc[:,idx]
    Q = Q_new

    # timescale = len(Q) #5844 #3287 # 3652
    # print(timescale)

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
    external_inputs = np.zeros((timescale, reach_data.n_reaches, n_classes))
    
    # Define input sediment load in the deposit layer
    deposit = reach_data.deposit * reach_data.length
    
    # Define initial sediment fractions per class in each reaches, using a Rosin distribution
    Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi)
    
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
    
    
    
    # Call dcascade main
    data_output, extended_output = DCASCADE_main(reach_data, network, Q, psi, timescale, ts_length, al_depth,
                                                 indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 **kwargs)





    # Exclude variables not included in the plotting yet (sediment divided into classes)
    data_output_t = copy.deepcopy(data_output)
    variable_names = [data for data in data_output_t.keys() if data.endswith('per class [m^3/s]')]
    for item in variable_names:
        del data_output_t[item]


    # Save results as pickled files
    import pickle
    
    if not os.path.exists(path_results):   #does the output folder exist ?
        os.makedirs(path_results)          # if not, create it.

    name_file = path_results / 'RR_VFW_Init_6.p'
    pickle.dump(data_output, open(name_file , "wb"))  # save it into a file named save.p

    #name_file_ext = path_results + 'save_all_ext.p'
    #pickle.dump(extended_output , open(name_file_ext , "wb"))  # save it into a file named save.p
    
    
    # ## Plot results
    # keep_slider = dynamic_plot(data_output_t, reach_data, psi)
    
# DCASCADE_run(filename_river_network, filename_q, path_results, timescale, sed_range, n_classes, indx_tr_cap , indx_tr_partition, al_depth = 0.3, vel_height = '2D90')
