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
#from plot_function import dynamic_plot
import scipy.io
import copy
from numpy import random

# import ad hoc functions
from GSD import GSDcurvefit
from preprocessing import graph_preprocessing, extract_Q
from DCASCADE_loop_vWac import DCASCADE_main, ReachData
from widget import read_user_input
import profile
import os
from pathlib import Path



'''user defined input data'''


#-------River shape files 
path_river_network = Path('Input/RangitataFC_dH/')
name_river_network = 'River_Network5.shp' #has width hydraulic geometry a and b in form Bpred = a .* Q^b % [m from m3/s]
filename_river_network = path_river_network / name_river_network
#matlab: outq(:,2) = outq(:,2)*.9; is upper main valley
#        outq(:,1) = outq(:,1)*.1; is upper Forest Ck
#so, any incoming sediment ought to join not at [0] which is ForCk but at [1] in python.
#and to remove erosion of the gorge, zero sediment at index 4 (matlab reach 5)

#--------Discharge files
path_q = Path('Input/RangitataFC_dH/')
# csv file that specifies the water flows in m3/s as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)
#name_q = 'q_Apr2024_1060.csv'
name_q = 'q_2024.csv'
filename_q = path_q / name_q

#csv file with the size of q timeseries. could simplify to just specific reaches, but let's keep full structure for now. 
name_qs = 'qsand_40pct_gravUpper68_2024.csv'
filename_qs = path_q / name_qs


#--------Path to the output folder
path_results = Path("Oct25RangitataFC_dH/Rev2_5pctsand_1234/")
name_file = path_results / 'save_all.p'

#--------Parameters of the simulation

#---Sediment classes definition 
# defines the sediment sizes considered in the simulation
#(must be compatible with D16, D50, D84 defined for the reach - i.e. max sed class cannot be lower than D16)
sed_range = [-9, 3]  # range of sediment sizes - in Krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5). 
n_classes = 7        # number of classes

#---Timescale 
nrepeats = 5
#timescale =  385 # hours   #420
timescale =  2880 # hours   #420
ts_length = 60 * 60 # length of timestep in seconds - 60*60*24 = daily; 60*60 = hourly

#---Change slope or not
update_slope = True # if False: slope is constant, if True, slope changes according to sediment deposit

#---Initial layer sizes #ccJR chaged this to a nominal width * depth. which is why 1000 didn't work, too wide for that!
deposit_layer = 1000 * 10   # Initial deposit layer [m]. Warning: will overwrite the deposit column in the reach_data file
eros_max = 1             # Maximum depth (threshold) that can be eroded in one time step (here one day), in meters. 

#---Storing Deposit layer
save_dep_layer = 'yearly' # 'yearly', 'always', 'never'.  Choose to save or not, the entire time deposit matrix

#---Others
roundpar = 1 # mimimum volume to be considered for mobilization of subcascade (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)



################ MAIN ###############

# Read the network 
network = gpd.GeoDataFrame.from_file(filename_river_network) #read shapefine from shp format
reach_data = ReachData(network)

# Define the initial deposit layer per each reach in [m3/m] ccJR so this incorporates width. 
reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
reach_data.deposit[4] = 1  #ccJR hardcoded GORGE no sources, just whta I give it. 
# Read/define the water discharge  
Q = extract_Q(filename_q)

# Sort reach_data according to the from_n, and organise the Q file accordingly
#also this turns it from a dataframe into a numpy array. which I understand, so convert Qs similary
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
# Read/define the sediment discharge  
Qs_dframe = extract_Q(filename_qs)
print('Applying Qs to Reach 2, input shape', Qs_dframe.shape)
Qs = Qs_dframe.to_numpy()
Qbi_input = np.zeros((timescale, reach_data.n_reaches, n_classes))
#ccJR - put in a constant source of the finest sediment in reach 3
#ccJR UNITS are [m3/timestep] of 'pure sediment.' We here throw out any of Qs that is AFTER 'timescale' timesteps. 

Qbi_input[:,1,5:6] = Qs[0:timescale,5:6] * 0.5 #ccJR HARDCODED aha - adding much more as it is going straight to the bed. 
Qbi_input[:,2,5:6] = Qs[0:timescale,5:6] * 0.5 #ccJR HARDCODED bin 6 125 micron for now. added bin 5. 
Qbi_input[:,3,5:6] = Qs[0:timescale,5:6] * 0.5 #ccJR HARDCODED 100% of half is 50% of the natural sand load.
Qbi_input[:,4,5:6] = Qs[0:timescale,5:6] * 0.5 #ccJR HARDCODED I should probably nix the 0.5mm sand and just keep a 250.

# Define input sediment load in the deposit layer
deposit = reach_data.deposit * reach_data.length

# Define initial sediment fractions per class in each reaches, using a Rosin distribution
Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi) 

# Initialise deposit layer 
Qbi_dep_in = np.zeros((reach_data.n_reaches, 1, n_classes))
for n in range(reach_data.n_reaches):
    Qbi_dep_in[n] = deposit[n] * Fi_r[n,:]

# Formula selection
# indx_tr_cap , indx_partition, indx_flo_depth, indx_slope_red = read_user_input()
# If you want to fix indexes, comment the line above and fix manually the indexes
indx_tr_cap = 2 # Wilcock and Crowe 2003 is 2, Parker is 1
indx_partition = 4 # Shear stress correction
indx_flo_depth = 1 # Manning
indx_slope_red = 1 # None
indx_velocity = 2 # 3 and 4 seem to be gone. 2 should vary virtual velocity by grain size. JR to add some sand suspension boosts to this?
indx_velocity_partition = 1 # same velocity for all classes

# Options for the cascade algorithm:        
# If all these option are False, we are normally reproducing the results
# of the old version. These option appear in order of complexity. 

# Option 1: consider overtaking sediments in outputs (if false, we reproduce the 
# old version)
consider_overtaking_sed_in_outputs = True

# Option 2: If True, we add a comparison to tr_cap to test if overpassing
# volumes must be partly deposited or if completed by the reach
compare_with_tr_cap = True

# Option 3: If True, we consider a time lag between the beginning of the time step,
# and the arrival of the first cascade to the ToN of the reach, 
# during which we are able to mobilise from the reach itself
time_lag_for_Vmob = True

#JR addition - variable Wac. Currently with easy to set up hydraulic geometry a,b stored in ReachData
#width hydraulic geometry a and b in form Bpred = a .* Q^b % [m from m3/s]
vary_width = True
if vary_width:
    Bcheck = reach_data.width_a * Q.max()**reach_data.width_b
    
    if any(reach_data.maxwac - Bcheck < 0):
        print('maxWac too low')
        Bcheck / reach_data.maxwac
        raise SystemExit()

reach_data_original = copy.deepcopy(reach_data)  

# Call dcascade main
data_output, extended_output = DCASCADE_main(indx_tr_cap , indx_partition, indx_flo_depth, indx_slope_red, 
                                             indx_velocity, indx_velocity_partition,                           
                                             reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                             roundpar, update_slope, eros_max, save_dep_layer, ts_length,vary_width,
                                             consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                             time_lag_for_Vmob)


# Exclude variables not included in the plotting yet (sediment divided into classes)
data_output_t = copy.deepcopy(data_output)
variable_names = [data for data in data_output_t.keys() if data.endswith('per class [m^3/s]')]
for item in variable_names: 
    del data_output_t[item]
    

# Save results as pickled files     
import pickle 

if not os.path.exists(path_results):   #does the output folder exist ?   
    os.makedirs(path_results)          # if not, create it.

pickle.dump(data_output, open(name_file , "wb"))  # save it into a file named save.p

#name_file_ext = path_results + 'save_all_ext.p'
#pickle.dump(extended_output , open(name_file_ext , "wb"))  # save it into a file named save.p
name_file = path_results / 'save_all_0.mat'
# Save using scipy.io (MATLAB-style)
scipy.io.savemat(name_file, {'data_output': data_output, 'extended_output': extended_output})



# %% testing recovering final volume / Fi for an annual re-run 'bed composition' hotstart

#input volume of top substrate
vol0 = (Qbi_dep_in[0,:]).sum()
#first timestep "
#vol1 = data_output['tot_sed'][0,0].sum()
#first timestep "
#vol1end = data_output['tot_sed'][-1,0].sum()
#vol1end - vol1

#starting sand volume:
(data_output['tot_sed_class'][6])[0,0]

#ending sand volume:
#(data_output['tot_sed_class'][6])[timescale-1,0]

#ending volume all claasses:
#(data_output['tot_sed_class'][6])[timescale-1,0]  #works
#(data_output['tot_sed_class'][5:6])[timescale-1,0]  #fails, can't index a list, need to loop:
# %% Annual loops after first run gets things going. Each 'year' or interval, we
#update the ENTIRE bed composition (resetting some important stratigraphy I am sure - don't neglect this step
#and from that starting point, re-run. This is meant to reach a steady state for a certain hydrograph

from supporting_functions import D_finder

for NR in range(nrepeats):    
    tot_m = 0
    # Initialise new deposit layer 
    Qbi_dep_in2 = np.zeros((reach_data.n_reaches, 1, n_classes))
    #write a Fs change matrix before the next loop. how many to equilibrate?
    np.set_printoptions(precision=2,suppress=True)
    print(100*(extended_output['Fi_r_ac'][timescale-2] - extended_output['Fi_r_ac'][1]))
    
    
    #new Qbi_dep_in. Tried tot_sed_class but    that is different from the Fi_active. Until I understand stratigraphy, let's
    #instead set all to new Fi as active layer's new Fi was getting averaged out with buried original material.
    # for i in range(len(psi)):  # This loops through grain sizes and collects volumes
    #     element = extended_output['Fi_r_ac'][i]
    #     value = element[timescale-1, 0] / reach_data.length[0]
    #     tot_m = tot_m + value
    #     print(f"reach 0 mm= {dmi[i]} Value: {value}")
    nsave_dep_layer = data_output['V_dep_sum'].shape[0]-1
    for n in range(reach_data.n_reaches):
        #Qbi_dep_in[n] = deposit[n] * Fi_r[n,:]    #template from above    
        #reassign new volumes to every reach from final timestep:
            #                this is like original var 'deposit'      * writes as if active Fi everywhere. 
        Qbi_dep_in2[n,0,:] = data_output['V_dep_sum'][nsave_dep_layer][n] * extended_output['Fi_r_ac'][timescale-2][n]
        reach_data.D16[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 16, psi)
        reach_data.D50[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 50, psi)
        reach_data.D84[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 84, psi)
        
    #update roughness. each vector is length n_reaches
    Hbar = (data_output['Q'][0:-1]).mean(0) / data_output['wac'][0:-1].mean(0)
    s8f_keul = (1 / 0.41) * np.log((12.14 * Hbar) / (3*reach_data.D84))
    C_keul = s8f_keul * np.sqrt(9.81)
    n_keul = Hbar**(1/6) / C_keul
    reach_data.n = n_keul
    # new el_fn and el_tn. Consider keeping the top reaches, and perhaps our gorge, from moving?
    reach_data.el_fn = extended_output['Node_el'][-1][0:reach_data.n_reaches]
    reach_data.el_tn = extended_output['Node_el'][-1][1:reach_data.n_reaches+1]
    
    reach_data.slope = data_output['slope'][-1]
    slope_length_delta = reach_data.length * (data_output['slope'][-1] - data_output['slope'][1]) #meters of increase this run
    
    
    # Call dcascade main
    data_output, extended_output = DCASCADE_main(indx_tr_cap , indx_partition, indx_flo_depth, indx_slope_red, 
                                               indx_velocity, indx_velocity_partition,                           
                                               reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                               roundpar, update_slope, eros_max, save_dep_layer, ts_length,vary_width,
                                               consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                               time_lag_for_Vmob)


    name_file = f"{path_results}/save_all_{NR+1}.mat"
    # Save using scipy.io (MATLAB-style)
    scipy.io.savemat(name_file, {'data_output': data_output, 'extended_output': extended_output})
