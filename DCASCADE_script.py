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
import numpy as np
import geopandas as gpd
import pandas as pd
from plot_function import dynamic_plot
import copy
from numpy import random

# import ad hoc functions
from GSD import GSDcurvefit
from preprocessing import graph_preprocessing
from DCASCADE_loop import DCASCADE_main
import profile
import os


'''user defined input data'''


#-------River shape files 
path_river_network = 'C:\\VirtualBoxShared\\outputs_rect\\'
name_river_network = 'Solda_RN_Straightlines_ETRS89UTM_ok.shp'

#--------Q files
path_q = 'C:\\Users\\FPitscheider\\OneDrive - Scientific Network South Tyrol\Desktop\\Projects\\ALTROCLIMA\\Solda\\DischargeSims\\'
name_q = 'Sims_2014-22_v2\\discharges_allReaches_m3_per_s_noNode29.csv' # timestep = 3287
#name_q = 'Sims_2014-22_v2\\discharges_allReaches_m3_per_s_noNode29_10yInitiation.csv' # timestep = 6937

#--------path to the output folder
path_results = 'C:\\Users\\FPitscheider\\OneDrive - Scientific Network South Tyrol\Desktop\\Projects\\ALTROCLIMA\\Solda\\dCascade_Results\\Simualtions\\2014-22_daily_sep24\\'


#--------Parameters of the simulation

#---Sediment classes definition 
# defines the sediment sizes considered in the simulation
#(must be compatible with D16, D50, D84 defined for the reach - i.e. max sed class cannot be lower than D16)
sed_range = [-10, -1]  # range of sediment sizes - in Krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5). 
n_classes = 10       # number of classes


#---Timescale 
timescale = 50 #3287  # days 
timestep = 60*60*24 # seconds per timescale unit - 60*60*24 = daily; 60*60 = hourly

#---Change slope or not
update_slope = False #if False: slope is constant, if True, slope changes according to sediment deposit

#---Initial layer sizes
deposit_layer = 0.1   # Initial deposit layer [m]. Warning: will overwrite the deposit column in the ReachData file
eros_max = 0.3             # Maximum depth (threshold) that can be eroded in one time step (here one day), in meters. 

#---Storing Deposit layer
save_dep_layer = 'never' #'yearly', 'always', 'never'.  Choose to save or not, the entire time deposit matrix

#---Others
roundpar = 0 #mimimum volume to be considered for mobilization of subcascade (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)



################ MAIN ###############

# Read the network 
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefine from shp format

# Define the initial deposit layer per each reach in [m3/m]
ReachData['deposit'] = np.repeat(deposit_layer, len(ReachData))

deposit_nodes = [
    #Solda
    1, 3, 4, 5, 6, 7, 8, 
    #Trafoi             
    29, 30, 31, 32, 34, 37, 38              
                 ] #FromN

#ReachData['deposit'] = 0
ReachData.loc[ReachData['FromN'].isin(deposit_nodes), 'deposit'] = 100000


increaseGSD_nodes = [
    #Solda
    21, 22, 23,             
                 ]

ReachData.loc[ReachData['FromN'].isin(increaseGSD_nodes), 'D16'] *= 2
ReachData.loc[ReachData['FromN'].isin(increaseGSD_nodes), 'D50'] *= 2
ReachData.loc[ReachData['FromN'].isin(increaseGSD_nodes), 'D84'] *= 2
ReachData.loc[ReachData['FromN'].isin(increaseGSD_nodes), 'D90'] *= 2


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
ReachData = ReachData.sort_values(by = 'FromN')
Q_new = np.zeros((Q.shape))
for i, idx in enumerate(ReachData.index): 
    Q_new[:,i] = Q.iloc[:,idx]
Q = pd.DataFrame(Q_new)
ReachData = ReachData.sort_values(by = 'FromN', ignore_index = True)

# Extract network properties
Network = graph_preprocessing(ReachData)

# Sediment classes defined in Krumbein phi (φ) scale   
psi=np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)

# Sediment classes in mm
dmi = 2**(-psi).reshape(-1,1)

# Check requirements. Classes must be compatible with D16, D50, D84 defined for the reaches - i.e. max sed class cannot be lower than D16
print(min(ReachData['D16'])*1000, ' must be greater than ', np.percentile(dmi,10, method='midpoint'))
print(max(ReachData['D84'])*1000, ' must be lower than ',  np.percentile(dmi,90, method='midpoint'))
   

n_reaches = len(ReachData)
# External sediment for all reaches, all classes and all timesteps 
Qbi_input = [np.zeros((n_reaches,n_classes)) for _ in range(timescale)]

# Define input sediment load in the deposit layer
deposit = ReachData.deposit*ReachData.Length

# Define initial sediment fractions per class in each reaches, using a Rosin distribution
Fi_r,_,_ = GSDcurvefit( ReachData.D16, ReachData.D50, ReachData.D84 , psi) 

# Initialise deposit layer 
Qbi_dep_in = [np.zeros((1,n_classes)) for _ in range(n_reaches)] 
for n in range(len(ReachData)):
    Qbi_dep_in[n] = deposit[n]*Fi_r[n,:]

# Call dcascade main
data_output, extended_output = DCASCADE_main(ReachData, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi, roundpar, update_slope, eros_max, save_dep_layer, timestep) 

# Exclude variables not included in the plotting yet (sediment divided into classes)
data_output_t = copy.deepcopy(data_output)
variable_names = [data for data in data_output_t.keys() if data.endswith('per class [m^3/s]')]
for item in variable_names: 
    del data_output_t[item]
    
'''
# Save results as pickled files     
import pickle 

if not os.path.exists(path_results):   #does the output folder exist ?   
    os.makedirs(path_results)          # if not, create it.
    
name_file = path_results + 'Solda_14-22_WC_SSC_hFerg_SredR_a1p5_dep0p1.p'
pickle.dump(data_output, open(name_file , "wb"))  # save it into a file named save.p

#name_file_ext = path_results + 'save_all_ext.p'
#pickle.dump(extended_output , open(name_file_ext , "wb"))  # save it into a file named save.p
'''

# ## Plot results 
#keep_slider = dynamic_plot(data_output_t, ReachData, psi)

