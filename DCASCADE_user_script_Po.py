<<<<<<< Updated upstream
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 18:47:53 2025

@author: Sahansila
"""

=======
>>>>>>> Stashed changes

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

# import libraries 
import numpy as np
import geopandas as gpd
import pandas as pd
# from plot_function import dynamic_plot
import copy
from numpy import random

# import ad hoc functions
from GSD import GSDcurvefit
from preprocessing import graph_preprocessing, extract_Q
from DCASCADE_main_script import DCASCADE_main, ReachData
from widget import read_user_input
import profile
import os
from pathlib import Path



'''user defined input data'''


#-------River shape files 
path_river_network = Path("C:\\Sahansila\\cascade\\input\\shp_file_slopes_hydro_and_LR\\02-shp_trib_GSD_updated\\")
name_river_network = 'Po_river_network.shp'
   
filename_river_network = path_river_network / name_river_network

#--------Discharge files
path_q = Path('C:\\Sahansila\\cascade\\input\\Discharge\\')
# csv file that specifies the water flows in m3/s as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)

name_q = 'Po_Qdaily_3y.csv'
filename_q = path_q / name_q

#--------External input (here sand for Dora Baltea, Sesia, Tanaro, Scrivia)
# path_extern_input = Path('../../Po_case_16y/Inputs/sand_input/sand_to_add_Vezzoli_2nd_method/')
# # csv file that specifies the water flows in m3/s as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)
# name_extern_input = 'sand_to_add_5.p'
# filename_ext = path_extern_input / name_extern_input


#--------Path to the output folder
<<<<<<< Updated upstream
path_results = Path("C:\\Sahansila\\Cascade_v2.0\\cascade_results\\")
name_file = path_results / 'save_3yr.p'
=======
path_results = Path("C:\\Sahansila\\cascade\\SAFE_output\\data_output_batch_test\\")
# name_file = path_results / 'save_all.p'
>>>>>>> Stashed changes
# name_file_ext = path_results / 'save_all_ext.p'


#--------Parameters of the simulation

#---Sediment classes definition 
# defines the sediment sizes considered in the simulation
#(must be compatible with D16, D50, D84 defined for the reach - i.e. max sed class cannot be lower than D16)
sed_range = [-8, 3]  # range of sediment sizes - in Krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5). 
<<<<<<< Updated upstream
n_classes = 6       # number of classes

#---Timescale 
timescale = 1095 #5843 days 1095
=======
n_classes = 6        # number of classes

#---Timescale 
timescale =1095 #5843 days 1095
>>>>>>> Stashed changes
ts_length = 60 * 60 * 24 # length of timestep in seconds - 60*60*24 = daily; 60*60 = hourly

#---Transport capacity formula and partitioning
indx_tr_cap = 3 # 2: Wilkock and Crowe 2003; 3: Engelund and Hansen; 6: Ackers and White
indx_tr_partition = 2 # 2: BMF; 4: Shear stress correction

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
    indx_tr_cap, indx_tr_partition, indx_flo_depth = read_user_input()

# Read the network 
network = gpd.GeoDataFrame.from_file(filename_river_network) #read shapefine from shp format
reach_data = ReachData(network)

# Define the initial deposit layer per each reach in [m3/m]
reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)

# Read/define the water discharge  
Q = extract_Q(filename_q)

# Sort reach_data according to the from_n, and organise the Q file accordingly
sorted_indices = reach_data.sort_values_by(reach_data.from_n)
Q_new = np.zeros((Q.shape))
for i, idx in enumerate(sorted_indices): 
    Q_new[:,i] = Q.iloc[:,idx]
Q = Q_new

# Extract network properties
Network = graph_preprocessing(reach_data)

# Sediment classes defined in Krumbein phi (φ) scale   
psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)

# Sediment classes in mm
dmi = 2**(-psi).reshape(-1,1)

# Check requirements. Classes must be compatible with D16, D50, D84 defined for the reaches - i.e. max sed class cannot be lower than D16
print(min(reach_data.D16) * 1000, ' must be greater than ', np.percentile(dmi, 10, method='midpoint'))
print(max(reach_data.D84) * 1000, ' must be lower than ',  np.percentile(dmi, 90, method='midpoint'))
   

# External sediment for all reaches, all classes and all timesteps 
external_inputs = np.zeros((timescale, reach_data.n_reaches, n_classes))
#external_inputs = pd.read_pickle(open(filename_ext , "rb"))
#force_pass_external_inputs = True

# Define input sediment load in the deposit layer
deposit = reach_data.deposit * reach_data.length

# Puting Ticino back to its old GSD
FromN_Ticino = 51
reach_data.D16[FromN_Ticino - 1] = network.loc[network['FromN'] == FromN_Ticino, 'D16_old']
reach_data.D50[FromN_Ticino - 1] = network.loc[network['FromN'] == FromN_Ticino, 'D50_old']
reach_data.D84[FromN_Ticino - 1] = network.loc[network['FromN'] == FromN_Ticino, 'D84_old']

# Puting Lambro back to its old GSD
FromN_Lambro = 53
reach_data.D16[FromN_Lambro - 1] = network.loc[network['FromN'] == FromN_Lambro, 'D16_old']
reach_data.D50[FromN_Lambro - 1] = network.loc[network['FromN'] == FromN_Lambro, 'D50_old']
reach_data.D84[FromN_Lambro - 1] = network.loc[network['FromN'] == FromN_Lambro, 'D84_old']


# Smooth Po main stem slopes
# def smooth_vector_manual_3(vector):
    
#     smoothed = np.zeros_like(vector)
    
#     smoothed[0] = (vector[0] + vector[1]) / 2
    
#     smoothed[1] = (vector[0] + vector[1]) / 2

#     for i in range(1, len(vector) - 1):
#         smoothed[i] = (vector[i - 1] + vector[i] + vector[i + 1]) / 3

#     smoothed[-1] = (vector[-1] + vector[-2]) / 2

#     return smoothed


# def smooth_vector_manual_4(vector):
    
#     smoothed = np.zeros_like(vector)
    
#     smoothed[0] = (vector[0] + vector[1] + vector[2]) / 3
    
#     for i in range(1, len(vector) - 2):
#         smoothed[i] = (vector[i - 1] + vector[i] + vector[i + 1] + vector[i + 2]) / 4

#     smoothed[-2] = (vector[-3] + vector[-2] + vector[-1]) / 3
    
#     smoothed[-1] = (vector[-2] + vector[-1]) / 2

#     return smoothed

# Po_only = network.loc[network['River'] == 'Po']
# reach_data_Po = ReachData(Po_only)
# _ = reach_data_Po.sort_values_by(reach_data_Po.from_n)
# Po_slopes = reach_data_Po.slope

# smoothed_slopes = smooth_vector_manual_4(Po_slopes)


   

# Define initial sediment fractions per class in each reaches, using a Rosin distribution
Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi)

<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
# # Add sand in Fir of Dora Baltea, Sesia, Tanaro, Scrivia 
# FromN_tribs = [47, 48, 49, 50]
# # Rs_Vezzoli = [0.43, 0.35, 0.26, 0.22]
# Rs_tuned = [0.06, 0.5, 0.3, 0.4]
# for FromN in FromN_tribs:
#     idx_trib = FromN_tribs.index(FromN)    
#     Fi_r[FromN-1,7:] = Rs_tuned[idx_trib]/len(Fi_r[FromN-1,7:])
#     # Make sure that the non sand make 1-Rs in total
#     sum_non_sand = np.sum(Fi_r[FromN-1,:7])
#     ratio = (1 - Rs_tuned[idx_trib])/sum_non_sand
#     Fi_r[FromN-1,:7] = Fi_r[FromN-1,:7] * ratio
    
#     #check sum makes 1
#     print(np.sum(Fi_r[FromN-1,:]))
    


    
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
data_output, extended_output = DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length, al_depth,
                                             indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                             **kwargs)
                                                



# Exclude variables not included in the plotting yet (sediment divided into classes)
<<<<<<< Updated upstream
data_output_t = copy.deepcopy(data_output)
variable_names = [data for data in data_output_t.keys() if data.endswith('per class [m^3/s]')]
for item in variable_names: 
    del data_output_t[item]
=======
# data_output_t = copy.deepcopy(data_output)
# variable_names = [data for data in data_output_t.keys() if data.endswith('per class [m^3/s]')]
# for item in variable_names: 
#     del data_output_t[item]
>>>>>>> Stashed changes
    

# Save results as pickled files     
import pickle 

<<<<<<< Updated upstream
=======
name_file = 'C:\\Sahansila\\Cascade_v2.0\\save_reults_EH_2019-21'


>>>>>>> Stashed changes
if not os.path.exists(path_results):   #does the output folder exist ?   
    os.makedirs(path_results)          # if not, create it.

pickle.dump(data_output, open(name_file , "wb"))  # save it into a file named save.p

# pickle.dump(extended_output , open(name_file_ext , "wb"))  # save it into a file named save.p


# ## Plot results 
# keep_slider = dynamic_plot(data_output_t, reach_data, psi)


