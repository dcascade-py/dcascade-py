# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:21:34 2022

Input that are required in the ReachData dataframe which define your river network:
- reach FromN - ToN (FRom Node - To Node) which define the relation between reaches (from upstream to downstream), these must be ordered from the smaller to the greater (e.g. first reach Id = 0, fromN = 1, ToN = 2)
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

'''user defined input data'''


#Shape files 
#path_river_network = 'C:\\Users\\user1\\Documents\\dcascade_py\\Input\\input_trial\\'
#name_river_network = 'River_network.shp'

path_river_network = "E:\\UNIPD\\shp_file_slopes_hydro_and_LR\\"
name_river_network = "Po_rivernet_grainsze_new_d.shp"


# Q files
#path_q = 'C:\\Users\\user1\\Documents\\dcascade_py\\Input\\input_trial\\'
#name_q = 'Q_Vjosa.csv' # csv file that specifies the water flows as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)

path_q = "E:\\cascade\\input\\"
name_q = 'Po_Qdaily_latest_cascade.csv' 

path_results = "E:\\cascade\\cascade_results\\"

roundpar = 0 #mimimum volume to be considered for mobilization of subcascade (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)

#Sediment classes definition (must be compatible with D16, D50, D84 defined for the reach - i.e. max sed class cannot be lower than D16)
sed_range = [-8, 5]  #range of sediment sizes considered in the model - in log scale where each number is the average diameter of that sediment class (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5). 
class_size = 2.5  # amplitude of the sediment classes



# read the network 
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefine from shp format

# define the initial deposit layer per each reach in [m3/m]
ReachData['deposit'] = np.repeat(100000, len(ReachData))


# read/define the water discharge 
Q = pd.read_csv(path_q + name_q , header = None, sep=',') # read from external csv file


################ MAIN ###############

n_reaches = len(ReachData)

#Order the ReachData network by FromNode (needed for graph operations)
ReachData = ReachData.sort_values(by = 'FromN')
# order flow rates accordingly 
Q_new = np.zeros((Q.shape))
for i, idx in enumerate(ReachData.index): 
    Q_new[:,i] = Q.iloc[:,idx]
Q = pd.DataFrame(Q_new)

Q.to_csv(path_q + 'Q_latest_reordered.csv', index=False)

#timescale 

# timescale = 10 # days 

timescale = len(Q) # days 

ReachData = ReachData.sort_values(by = 'FromN', ignore_index = True)


# mwindow 
ReachData.rename(columns={'Slope': 'Slope_or'}, inplace=True)
ReachData['Slope'] = np.where(ReachData['River'] == 'Po', ReachData['Slope_or'].rolling(window=2).mean(), ReachData['Slope_or'])


#extract network properties
Network = graph_preprocessing(ReachData)

# sediment classes defined in Krumbein phi (φ) scale   
psi = np.arange(sed_range[0], sed_range[-1], class_size)

# check requirement  
dmi = 2**(-psi).reshape(-1,1)
print(min(ReachData['D16_05']), ' must be greater than ', np.percentile(dmi,10, method='midpoint'))
print(max(ReachData['D84_05']), ' must be lower than ',  np.percentile(dmi,90, method='midpoint'))
   

n_classes = len(psi)
del sed_range, class_size
  

# external sediment for all reaches, all classes and all timesteps 
Qbi_input = [np.zeros((n_reaches,n_classes)) for _ in range(timescale)]

# define input sediment load in the deposit layer
deposit = ReachData.deposit*ReachData.Length
Fi_r,_,_ = GSDcurvefit( ReachData.D16, ReachData.D50, ReachData.D84 , psi) # per each reach, Rosin distribution of sediments for the diameters specified in sed_range 

#initialise deposit layer 
Qbi_dep_in = [np.zeros((1,n_classes)) for _ in range(n_reaches)] # initialise the deposit layer 
for n in range(len(ReachData)):
    Qbi_dep_in[n] = deposit[n]*Fi_r[n,:]
    
# to add deposit layer at a given reach 
#row = np.array(range(n_classes)).reshape(1,n_classes)  
#Qbi_dep_in[0] = np.append(Qbi_dep_in[0],row,axis= 0)

# call dcascade 
data_output, extended_output = DCASCADE_main(ReachData, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi, roundpar) 
  
# # exclude variables not included in the plotting yet (sediment divided into classes)
# data_output_t = copy.deepcopy(data_output)
# variable_names = [data for data in data_output_t.keys() if data.endswith('per class [m^3/s]')]
# for item in variable_names: 
#     del data_output_t[item]

# ## plot results 
# keep_slider = dynamic_plot(data_output_t, ReachData, psi)

    
""" save results as pickled files 
     
import pickle 
name_file = path_results + 'Po_results_H03.p'
pickle.dump(data_output, open(name_file , "wb"))  # save it into a file named save.p

name_file_ext = path_results + 'Po_results_extended.p'
pickle.dump(extended_output, open(name_file_ext , "wb"))  # save it into a file named save.p

# load outout 
extended_output = pickle.load(open(name_file_ext , "rb"))
data_output = pickle.load(open(name_file , "rb"))

 """
 
#a = profile.run('main()', sort=2)