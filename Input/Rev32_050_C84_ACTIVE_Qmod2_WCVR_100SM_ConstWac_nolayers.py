# -*- coding: utf-8 -*-
"""
19 Nov 2024
JR brought in changes from C:\bin\cascade\dcascade_py-JR_oct29_dWac_dVactive\Rangitata_script_vWac_upper_dVactHypso_02.py

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
import sys
import os
import shutil
sys.path.insert(0, os.path.abspath(".."))

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
from DCASCADE_main_script import DCASCADE_main
from supporting_classes import ReachData
from widget import read_user_input
#import profile

from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
#from line_profiler import profile

'''user defined input data'''
np.set_printoptions(precision=3,suppress=True)
print(__file__)
#-------River shape files 
path_river_network = Path('../../Rangitata_Rev20_inputs/')
name_river_network = 'River_Network10.shp' #fixed a slope issue near the gorge, from rev8
filename_river_network = path_river_network / name_river_network

#--------Discharge files
path_q = Path('../../Rangitata_Rev20_inputs/')
# csv file that specifies the water flows in m3/s as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)
name_q = 'qmod2_2024_1Jan_1Jul_cel2p25_Atten22.csv'
filename_q = path_q / name_q

#csv file with the size of q timeseries. could simplify to just specific reaches, but let's keep full structure for now. 
name_qs = 'qsand_2024_1Jan_1Jul_cel2p25.csv'
filename_qs = path_q / name_qs

#--------Path to the output folder
path_results = Path("../Results/Rev32/01-Qmod2_ACTIVE_Ferg_050_C84_100SM_ConstWAC_nolayers/")
name_file = path_results / 'save_all.p'

#--------Parameters of the simulation

#---Sediment classes definition 
# defines the sediment sizes considered in the simulation
#(must be compatible with D16, D50, D84 defined for the reach - i.e. max sed class cannot be lower than D16)
sed_range = [-9, 3]  # range of sediment sizes - in Krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5). 
n_classes = 7        # number of classes

#---Timescale 
timescale = 4369 # hours   #4369  2882
ts_length = 60 * 60 * 1 # length of timestep in seconds - 60*60*24 = daily; 60*60 = hourly
nrepeats = 100 # number of times to repeat the hydrograph. think 'years' ?
#---Change slope or not
update_slope = True # if False: slope is constant, if True, slope changes according to sediment deposit

#---Initial layer sizes #ccJR chaged this to a nominal width * depth. which is why 1000 didn't work, too wide for that!
#what are the units now?
deposit_layer = 1.00   # Initial deposit layer thickness [m]. Warning: will overwrite the deposit column in the reach_data file
nlayers_init = 50 #ccJR split up deposit, considered as WIDTH. best to have these a different number than n_classes
eros_max = .10             # Maximum depth (threshold) that can be eroded in one time step  in meters. 

#---Storing Deposit layer
save_dep_layer = 'monthhour' # 'yearly', 'always', 'never' 'monthhour'.  Choose to save or not, the entire time deposit matrix

#---Others
roundpar = 0 # mimimum volume to be considered for mobilization of subcascade (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)



################ MAIN ###############

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
#psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)
psi = np.array([-9.,-8, -7.,-6., -5., -3.,  2.])
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
#ccJR - put in a constant source of fine sediment in specific reaches

#ccJR UNITS are [m3/timestep] of 'pure sediment.' We here throw out any Qs we read that is AFTER 'timescale' timesteps. 
add_Qbi=True #switch to turn on or off Qbi_input code
if add_Qbi:
    #original load. now all in 250 micron bin. 
    Qbi_input[:,4,6] = Qs[0:timescale,6] #ccJR HARDCODED bin 6 250 micron 
    ##Qbi_input[:,3,6] = Qs[0:timescale,6] * (1/3) #ccJR HARDCODED 100% of half is 50% of the natural sand load.
    #Qbi_input[:,4,6] = Qs[0:timescale,6] * (1/3) #ccJR 
   


#TEST SENSITIVIY of my width resolving code to changes in wac_bf ccJR 
print(reach_data.wac_bf[:])
reach_data.wac_bf[:] = max(reach_data.wac_bf[:])
print(reach_data.wac_bf[:])

# Define input sediment load in EACH deposit layer. JR making width explicit.
deposit = reach_data.deposit * reach_data.length * reach_data.wac_bf
# Define initial sediment fractions per class in each reaches, using a Rosin distribution
Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi) 

#cJR no sand from above, higher IC in flat bit.
Fi_r[:,6:7] = 0.00 #low sand IC : layer. initial condition testing. 0 is the bottom, nlayers_init-1 is the top (thalweg)
Fi_r[0,6:7] = 0
Fi_r[4,6:7] = 0.05
Fi_r[5:,6:7] = 0.01

# Initialise deposit layer 
hypsolayers = False

if hypsolayers == True:  #total Qbi_dep_in is deposit, times length, times width in meters. this is then split up by the # of width 'layers'
    Qbi_dep_in = np.zeros((reach_data.n_reaches, nlayers_init, n_classes))
    for n in range(reach_data.n_reaches):
        for nl in range(nlayers_init):
            Fi_r[n] /= Fi_r[n].sum()  # Renormalize  
            Qbi_dep_in[n,nl,:] = deposit[n] * Fi_r[n,:] * (1/nlayers_init) #each layer has an equal proportion of the width. so layer size accordions with wac_bf. 
            #diagnostic code. using boulder material to just do some orienting of myself. 
            #Qbi_dep_in[n,nl,0] = 111 * nl 
        
else:
    nlayers_init = 1
    Qbi_dep_in = np.zeros((reach_data.n_reaches, nlayers_init, n_classes))
    for n in range(reach_data.n_reaches):
        for nl in range(nlayers_init):
            Qbi_dep_in[n,nl,:] = deposit[n] * Fi_r[n,:]


# Compulsory indexes to choose:
# Indexes for the transport capacity:
indx_tr_cap = 28 # 2: Wilkock and Crowe 2003; 3: Engelund and Hansen.
indx_tr_partition = 4 # 2: BMF; 4: Shear stress correction

# Index for the flow calculation: 
indx_flo_depth = 2 # Manning (alternatives where developed for accounting for mountain stream roughness)

# If these variable are not chosen manually: 
if 'indx_tr_cap' not in globals() or 'indx_tr_partition' not in globals() or 'indx_flo_depth' not in globals():
    indx_tr_cap, indx_tr_partition, indx_flo_depth = read_user_input()
    
# Velocity indexes:
indx_velocity = 2 # method for calculating velocity (1: computed on each cascade individually, 2: on whole active layer)
indx_vel_partition = 2 # velocity section partitionning (1: same velocity for all classes, 2: section shared equally for all classes)

# Slope index:
indx_slope_red = 1 # None (alternatives where developed for accounting for mountain stream roughness)

# Options for the cascade algorithm (by default, they are all True):        
# If all these options are False, we are reproducing the algorithme of 
# the old version. Which means that cascades are all passing if the time step 
# is not finished for them (= based on their velocities) + overpassing cascades are 
# not considered in the mobilised volume nor transported

# Option 1: If True, we consider ovepassing sediment in the output (Qbimob and Qbitr).
# But this does not change the way sediment move.
op1 = True

# Option 2: If True, we now include present cascades from upstream + reach material
# in the transport capacity calculation, to check if they should pass or not. 
op2 = True

# Option 3: If True, we consider a time lag between the beginning of the time step,
# and the arrival of the first cascade to the ToN of the reach, 
# during which we are able to mobilise from the reach itself
op3 = False

#JR addition - variable Wac. Currently with easy to set up hydraulic geometry a,b stored in ReachData
#width hydraulic geometry a and b in form Bpred = a .* Q^b % [m from m3/s]
vary_width = False

#JR addition - recalculate roughness from changing GSD. Question - do this annually, or per timestep?
vary_roughness = False

#ROUGHNESS ALTERATION and JMR hydraulics edits
reach_data.C84_fac[:] = 0.50
reach_data.SUSP_MULT = 1.00
reach_data.slope_h_red_fac = 0.15


if vary_width:
    Bcheck = reach_data.width_a * Q.max()**reach_data.width_b
    
    if any(reach_data.wac_bf - Bcheck < 0):
        print('Wac_BF too low')
        Bcheck / reach_data.wac_bf
        #raise SystemExit()

        
#------ reach hypsometry tables
reach_hypsometry = np.zeros(reach_data.wac.shape,dtype = bool)
#hard code which ones exist. could read from dir..
 
# reach_hypsometry[5:14] = True


reach_hypsometry_data = {}
# Loop through each reach
for i in range(len(reach_hypsometry)):
    
    if reach_hypsometry[i]:
        ip1 = i+1 #naming convention for matlab exported reaches
        qstepsfilename = path_q / "Qsteps_Table.csv"
        #qstepsfilename = f"../Rangitata_FC_dH_MannThresh/Qsteps_Table.csv"
        qdf = pd.read_csv(qstepsfilename, header=0)  # assuming no header in the CSV files
        Qindex = qdf.iloc[:, 0].astype(float).values
        Qsteps = qdf.iloc[:, 1].astype(float).values
        # file name from reach index
        filename = path_q / f"Active_Hypsometry_q1_23_{ip1}.csv"
        #filename = path_q / f"Wet_Hypsometry_q1_23_{ip1}.csv"
        
    
        df = pd.read_csv(filename, header=0)  # assuming no header in the CSV files
        #tried to put this in reach_data but didn't get it right
        Zvec = df.iloc[:, 0].astype(float).values  # Convert to float if necessary
        Wvec = df.iloc[:, 1:].astype(float).values
    
        # Store in the dictionary
        reach_hypsometry_data[i] = {
            'Zvec': Zvec,
            'Wvec': Wvec,
            'Hvec': Zvec-Zvec.min(),
            'Qsteps': Qsteps,
        }

reach_data_original = copy.deepcopy(reach_data)  


# Call dcascade main
data_output, extended_output = DCASCADE_main(reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                             reach_hypsometry, reach_hypsometry_data,
                                             vary_width,vary_roughness,hypsolayers,
                                             roundpar, update_slope, eros_max, save_dep_layer, ts_length,
                                             indx_tr_cap , indx_tr_partition, indx_flo_depth,
                                             indx_velocity = indx_velocity, 
                                             indx_vel_partition = indx_vel_partition,
                                             indx_slope_red = indx_slope_red,
                                             passing_cascade_in_outputs = op1,
                                             passing_cascade_in_trcap = op2,
                                             time_lag_for_mobilised = op3)


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

name_file = path_results / 'save_all_0.mat'
# Save using scipy.io (MATLAB-style)
try:
    extended_output["pyfile"] = __file__
    backup_path = os.path.join(path_results, f"backup_{os.path.basename(__file__)}")
    shutil.copy(__file__, backup_path)
except: 
    print('cant save _file_ while not running')
    
scipy.io.savemat(name_file, {'data_output': data_output, 'extended_output': extended_output}, do_compression=True)

 
##

from supporting_functions import D_finder

for NR in range(nrepeats):    
    tot_m = 0
    # Initialise new deposit layer. make sure to update call to main below. 
    
    #write a Fs change matrix before the next loop. how many to equilibrate?
    np.set_printoptions(precision=3,suppress=True)
    print(100*(extended_output['Fi_r_ac'][timescale-2] - extended_output['Fi_r_ac'][1]))
    #go and figure out how to save a stratigraphic Fi? for now, just use ac. 
    
     
    Qbi_dep_in2 = np.zeros((reach_data.n_reaches, nlayers_init, n_classes))
    #find last V_dep_sum with data (sometimes its length is ntimes, sometimes less often, sometimes last one is blank)
    nsave_dep_time = data_output['V_dep_sum'].shape[0]-1 #last deposit save index
    
    #find last Qbi_FiLayers with data (sometimes its length is ntimes, sometimes less often, sometimes last one is blank)
    last_tdata = None
    for tf in reversed(range(len(extended_output['Qbi_FiLayers']))):  # Iterate from the last index backwards
        if np.sum(extended_output['Qbi_FiLayers'][tf]) > 0:  # Check if the sum of the layer is greater than zero
            last_tdata = tf
            break  # Exit the loop as soon as the last non-zero index is found
    if last_tdata is None:            
        print('No bed data on restart?')
        raise()
    else:
        FiFinal_Layers = extended_output['Qbi_FiLayers'][last_tdata] # final time index
   
    for n in range(reach_data.n_reaches):
        #Qbi_dep_in[n] = deposit[n] * Fi_r[n,:]    #template from above    
        #reassign new volumes to every reach from final timestep:
            #                this is like original var 'deposit'      * writes as if active Fi everywhere. 
        #Qbi_dep_in2[n,0,:] = data_output['V_dep_sum'][nsave_dep_layer][n] * extended_output['Fi_r_ac'][timescale-2][n]
        for nl in range(nlayers_init):
            FiLayer = FiFinal_Layers[n,nl,:]
            #IMPORTANT choice here - reset bed volume, wiht new Fi?
            #OR, as curerntly written, use old V_dep_sum. would allow bed to degrade to zero potentially
            
            Qbi_dep_in2[n,nl,:] = deposit[n] * FiLayer * (1/nlayers_init)  #RESETS volume on bed (more stable)
            #Qbi_dep_in2[n,nl,:] = data_output['V_dep_sum'][nsave_dep_time][n] * FiLayer * (1/nlayers_init) #retains old volume (crashes? realism?

        # reach_data.D16[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 16, psi)
        # reach_data.D50[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 50, psi)
        # reach_data.D84[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 84, psi)
        reach_data.D16[n] = D_finder(FiFinal_Layers[n,nl,:], 16, psi)
        reach_data.D50[n] = D_finder(FiFinal_Layers[n,nl,:], 50, psi)
        reach_data.D84[n] = D_finder(FiFinal_Layers[n,nl,:], 84, psi)
        
    #update roughness. each vector is length n_reaches
    if vary_roughness:
        Hbar = (data_output['flow_depth'][0:-1]).mean(0) 
        s8f_keul = (1 / 0.41) * np.log((12.14 * Hbar) / (reach_data.C84_fac[n]*reach_data.D84))
        C_keul = s8f_keul * np.sqrt(9.81)
        n_keul = Hbar**(1/6) / C_keul
        reach_data.n = n_keul
    # new el_fn and el_tn. Consider keeping the top reaches, and perhaps our gorge, from moving?
    reach_data.el_fn = extended_output['Node_el'][-1][0:reach_data.n_reaches]
    reach_data.el_tn = extended_output['Node_el'][-1][1:reach_data.n_reaches+1]
    
    reach_data.slope = data_output['slope'][-1]
    slope_length_delta = reach_data.length * (data_output['slope'][-1] - data_output['slope'][1]) #meters of increase this run
    
    
    # Call dcascade main
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, Qbi_input, Qbi_dep_in2, timescale, psi,
                                                 reach_hypsometry, reach_hypsometry_data,
                                                 vary_width,vary_roughness,hypsolayers,
                                                 roundpar, update_slope, eros_max, save_dep_layer, ts_length,
                                                 indx_tr_cap , indx_tr_partition, indx_flo_depth,
                                                 indx_velocity = indx_velocity, 
                                                 indx_vel_partition = indx_vel_partition,
                                                 indx_slope_red = indx_slope_red,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)


    name_file = f"{path_results}/save_all_{NR+1}.mat"
    # Save using scipy.io (MATLAB-style)
    try:
        extended_output["pyfile"] = __file__
        backup_path = os.path.join(path_results, f"backup_{os.path.basename(__file__)}")
        shutil.copy(__file__, backup_path)
    except: 
        print('cant save _file_ while not running')
    scipy.io.savemat(name_file, {'data_output': data_output, 'extended_output': extended_output}, do_compression=True)
