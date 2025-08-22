# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:17:11 2025

@author: diane


Plot basic D-CASCADE outputs: 
one graph per reach, along time

Choose between:
'Volume out [m^3]':         total volume of sediment leaving a reach per time step (= sediment flux x time step)
'Volume in [m^3]':          total volume of sediment entering a reach per time step
'Transport capacity [m^3]': total transport capacity computed in a reach per time step (= volume out if the supply is not limited)
'Sediment budget [m^3]':    total sediment budget per time step (+ deposition, - erosion) (= vol in - vol out)
'D50 active layer [m]':     D50 of the active layer per time step (used to computed the transport capacity)
'D50 volume out [m]' :      D50 of the volume leaving the reach per time step 

"""

# Libraries 
import os
import numpy as np 
from matplotlib import pyplot as plt  
import pandas as pd
import geopandas as gpd


#---------------------Path to the pickle output

path = "..\\cascade_results\\" 
name_simu = 'Vjosa_test'

#---------------------Path to the input river network (.shp) or (.csv)

path_river_network = "..\\inputs\\input_trial\\" #Path to the shp
name_river_network = "River_Network.shp"

#---------------------Folder to store the plots
figure_folder = path+'figures_per_reach\\' # where you will store the figure

if not os.path.exists(figure_folder):       
    os.makedirs(figure_folder)
       
#--------------------Output name you want to plot
output_name = 'Volume out [m^3]'   # Output available in pickle file
#'D50 active layer [m]', 'D50 volume out [m]', 'Sediment budget [m^3]', 'Transport capacity [m^3]', 'Volume in [m^3]', 'Volume out [m^3]'



##############################################################################

##### Usefull function for naming figures

def rename_names(output_name):
    '''For exemple, renames 'Volume out [m^3]' into 'Volume_out' 
    to use for saving csv and plots
    '''
    new_name = ''
    for c in output_name:
        if c == ' ':
            new_name = new_name+'_'
        elif c == '[':
            break
        elif c== '-':
            break
        else:
            new_name = new_name+c
    new_name = new_name[:-1]
    return new_name



##### Plot for each reach, x axis is the time

data_output = pd.read_pickle(open( path + name_simu + '.p' , "rb"))
my_data = data_output[output_name]
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network)

for i in range(my_data.shape[1]):
    reach_FromN = i + 1 #+1 because python starts at 0

    #create figure and graph axes
    fig = plt.figure()
    ax = plt.subplot(111)
        
    ax.plot(np.arange(1, len(my_data[:, i]) + 1, 1), my_data[:, i], linewidth = 2)
    
    if output_name == 'D50 active layer [m]':       
        D50_real = ReachData.loc[ReachData['FromN'] == i+1, 'D50'].unique()[0]
        ax.plot(0, D50_real, '*', label = 'D50meas', markersize = 10)
        
    ax.set_title('FromN: '+str(reach_FromN), fontsize = 16)
    ax.set_xlabel('Time', fontsize = 16)
    ax.set_ylabel(output_name,fontsize = 16)
    ax.tick_params(axis='both', labelsize = 14)
    
    if output_name == 'D50 active layer [m]':
        ax.legend(fontsize = 9)
    
    fig.set_tight_layout(True)
    fig.set_size_inches(900./fig.dpi,600./fig.dpi)
    new_name = rename_names(output_name)
    fig.savefig(figure_folder + str(new_name)+'_'+str(reach_FromN))
    fig.clf()
    plt.close("all")





