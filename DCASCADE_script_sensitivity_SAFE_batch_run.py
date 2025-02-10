 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:21:34 2022

Input that are required in the ReachData dataframe which define your river network:
- reach FromN - ToN (From Node - To Node) which define the relation between reaches (from upstream to downstream), these must be ordered from the smaller to the greater (e.g. first reach Id = 0, fromN = 1, ToN = 2)
- el_FN and el_TN (elevation fromN and ToN)
- Length, Wac (active channel width) in meters and Slope of the reach
- deposit = initial deposit layer expressed in m3/m2 - this value will be then multiplied by the reach width and length 
- D16, D50, D84 diameters expressed in [m] - will define the diameter distributions of the sediments present in the reach at t = 0 (i.e. of the deposit)
- Q = initial water discharge per reach in [m3/s]
- n = Manning coefficient for the calculation of the flow velocity 


Then you will also need a Dataframe which provides the water discharge per reach per time step: 
    rows = timestep
    columns = reaches 

Optional: you can provide external sediment sources per timestep, per reach and per class of sediments. This variable is defined by Qbi_input 

This script was adapted from the Matlab version by Marco Tangi 

@author: Elisa Bozzolan
"""
  
# import libraries 
import sys, os
import numpy as np
import geopandas as gpd
import pandas as pd
# from plot_function import dynamic_plot
import copy
from numpy import random

# import ad hoc functions
from GSD import GSDcurvefit
from preprocessing import graph_preprocessing
from DCASCADE_loop import DCASCADE_main
import profile
import os
import pickle


'''user defined input data'''


def DCASCADE_script_PAWN (i):
    #-------River shape files original network 
    path_river_network = "E:\\Sahansila\\input\\shp_file_slopes_hydro_and_LR\\02-shp_trib_GSD_updated\\"
    name_river_network = 'Po_river_network.shp'
    
    # Read the original network to sort the Q values
    ReachData_original = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefine from shp format
    
    
    #-------River shape files modified network  
    path_ReachData = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\ReachData_modified\\ReachData_modified_X_10000\\"
=======
>>>>>>> Stashed changes
    
    # Construct the filename or variable name for ReachData_modified
    reach_data_file = f'ReachData_modified_{i}.shp'
                    
    
    #--------Q files
    path_q = 'E:\\Sahansila\\input\\Discharge\\'
    # csv file that specifies the water flows in m3/s as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)
    name_q = 'Po_Qdaily_3y_only_2020.csv'
=======
    name_q = 'Po_Qdaily_3y.csv'
>>>>>>> Stashed changes
    
    
    #--------Parameters of the simulation
    
    #---Sediment classes definition 
    # defines the sediment sizes considered in the simulation
    #(must be compatible with D16, D50, D84 defined for the reach - i.e. max sed class cannot be lower than D16)
    sed_range = [-8, 3]  # range of sediment sizes - in Krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5). 
    n_classes = 6        # number of classes
    
    
    #---Timescale 
    timescale = 365 # days 
    ts_length = 60*60*24 # length of timestep in seconds - 60*60*24 = daily; 60*60 = hourly
    
    
    #---Change slope or not
    update_slope = False #if False: slope is constant, if True, slope changes according to sediment deposit
    
    #---Initial layer sizes
    deposit_layer = 100000   # Initial deposit layer [m]. Warning: will overwrite the deposit column in the ReachData file
    eros_max = 0.3             # Maximum depth (threshold) that can be eroded in one time step (here one day), in meters. 
    
    #---Storing Deposit layer
    save_dep_layer = 'never' #'yearly', 'always', 'never'.  Choose to save or not, the entire time deposit matrix
    
    #---Others
    roundpar = 0 #mimimum volume to be considered for mobilization of subcascade (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)
    
    
    ################ MAIN ###############
    
    # read the network 
    ReachData_modified = gpd.GeoDataFrame.from_file(path_ReachData + reach_data_file) #read shapefine from shp format
        
    # Define the initial deposit layer per each reach in [m3/m]
    ReachData_modified['deposit'] = np.repeat(deposit_layer, len(ReachData_modified))
    
    
    
    # Read/define the water discharge 
    # but first, we check automatically the delimiter (; or ,) and if Q file has headers or not:
    Q_check = pd.read_csv(path_q + name_q , header = None) # read from external csv file
    if Q_check.iloc[0,:].size == 1: 
        my_delimiter = ';'
    else:
        my_delimiter = ','
    Q_check2 = pd.read_csv(path_q + name_q, header=None, sep=my_delimiter)  
    if Q_check2.iloc[0,0]=='yyyy/mm/dd':
        Q = pd.read_csv(path_q + name_q, header = 0, sep=my_delimiter, index_col = 'yyyy/mm/dd')  
    else:
        Q = pd.read_csv(path_q + name_q, header = None, sep=my_delimiter)
    
    
    
    # Sort ReachData according to the FromN, and organise the Q file accordingly
    ReachData_original = ReachData_original.sort_values(by = 'FromN')
    Q_new = np.zeros((Q.shape))
    for i, idx in enumerate(ReachData_original.index): 
        Q_new[:,i] = Q.iloc[:,idx]
    Q = pd.DataFrame(Q_new)
    # ReachData_original = ReachData_original.sort_values(by = 'FromN', ignore_index = True)
    
    # Extract network properties
    Network = graph_preprocessing(ReachData_modified)
    
    # Sediment classes defined in Krumbein phi (φ) scale   
    psi=np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)
    
    # Sediment classes in mm
    dmi = 2**(-psi).reshape(-1,1)
    
    # Check requirements. Classes must be compatible with D16, D50, D84 defined for the reaches - i.e. max sed class cannot be lower than D16
    print(min(ReachData_modified['D16'])*1000, ' must be greater than ', np.percentile(dmi,10, method='midpoint'))
    print(max(ReachData_modified['D84'])*1000, ' must be lower than ',  np.percentile(dmi,90, method='midpoint'))
       
    
    n_reaches = len(ReachData_modified)
    # External sediment for all reaches, all classes and all timesteps +
    Qbi_input = np.zeros((timescale,n_reaches,n_classes))
    
    # Define input sediment load in the deposit layer
    deposit = ReachData_modified.deposit*ReachData_modified.Length
    
    # Define initial sediment fractions per class in each reaches, using a Rosin distribution
    Fi_r,_,_ = GSDcurvefit( ReachData_modified.D16, ReachData_modified.D50, ReachData_modified.D84 , psi) 
    
    # Initialise deposit layer 
    Qbi_dep_in = np.zeros((n_reaches,1,n_classes))
    for n in range(len(ReachData_modified)):
        Qbi_dep_in[n] = deposit[n]*Fi_r[n,:]
    
    # Formula selection
    # indx_tr_cap , indx_partition, indx_flo_depth, indx_slope_red = read_user_input()
    # If you want to fix indexes, comment the line above and fix manually the indexes
    indx_tr_cap = 3 # EH
    indx_partition = 2 # BMF
    indx_flo_depth = 1 # Manning
    indx_slope_red = 1 # None
    indx_velocity = 1 # same velocity for all classes
    
    # call dcascade
    data_output, extended_output = DCASCADE_main(indx_tr_cap , indx_partition, indx_flo_depth, indx_slope_red, indx_velocity,                            
                                                ReachData_modified, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                roundpar, update_slope, eros_max, save_dep_layer, ts_length)
    
    
    
    path_output = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\data_output_batch_test\\"
    # Construct the output file name without the '.shp' extension 
    base_filename = reach_data_file.replace('.shp', '')  # Remove the '.shp' extension
    output_file_name = f"data_output_{base_filename}.pkl"
       
    # Save the output data for each percentage change as pickled files
    output_file_path = path_output + output_file_name
    pickle.dump(data_output, open(output_file_path, "wb"))

if __name__== "__main__":
 	DCASCADE_script_PAWN(int(sys.argv[1]))
   

