# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:53:56 2023

Plot d-cascade results 

@author: Diane Doolaeghe
"""

# libraries 
import pickle
import numpy as np 
from matplotlib import pyplot as plt  
import pandas as pd
import geopandas as gpd
import matplotlib
from matplotlib.pyplot import text
import copy
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#matplotlib.use('Qt5Agg') 



#----------------------directories 
path = "E:\\cascade\\cascade_results\\"
name_output = 'Po_new_0.1.p'

path_Q = "E:\\cascade\\input\\"
name_q = 'Q_latest_reordered.csv' # the order is incresing FromN - ToNode in ReachData 

path_river_network = "E:\\UNIPD\\shp_file_slopes_hydro_and_LR\\"
name_river_network = "Po_rivernet_grainsze_new_d.shp"


figure_folder = "E:\\cascade\\figure1_new\\"


#---------------------loading data

# load network for reference 
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefine from shp format
ReachData = ReachData.sort_values(by = 'FromN', ignore_index = True )
ReachData_Po = ReachData[ReachData['River'] == 'Po'] # select Po
Po_idx = ReachData_Po['FromN'].values
Po_idx = Po_idx.astype(int)

# load outputs 
data_output = pickle.load(open( path + name_output , "rb"))


# load Q file
Q = pd.read_csv(path_Q + name_q, header = 0, sep=',')#, index_col = 'yyyy/mm/dd')  # read from external csv file




#------------------usefull function

def rename_names(output_name):
    '''For exemple, renames 'Mobilized volume [m^3]' into 'Mobilized_volume' 
    to use for saving csv and plots
    '''
    new_name = ''
    for c in output_name:
        if c == ' ':
            new_name = new_name+'_'
        elif c == '[':
            break
        else:
            new_name = new_name+c
    new_name = new_name[:-1]
    return new_name

  
#--------------------plots 

# plot all Qbi as a function of sed class
time_list = [327]
# t = 100

FromN = 6
n_class = len(data_output['Active layer sed in the reach - per class [m^3/s]'])
    
for t in time_list:
    transported_data=[]
    AL_data=[]
    tr_cap_data=[]
    mob_data=[]
    dep_data=[]
    
    for c in range(n_class):    
        output_name = 'Transported sed in the reach - per class [m^3/s]'
        transported_data.append(data_output[output_name][c][t,FromN-1])   
        
        output_name = 'Active layer sed in the reach - per class [m^3/s]'
        AL_data.append(data_output[output_name][c][t,FromN-1])
        
        output_name = 'Tr cap sed in the reach - per class [m^3/s]'
        tr_cap_data.append(data_output[output_name][c][t,FromN-1])
        
        output_name = 'Mobilised sed in the reach - per class [m^3/s]'
        mob_data.append(data_output[output_name][c][t,FromN-1])    
        
        output_name = 'Deposited sed in the reach - per class [m^3/s]' 
        #= deposit layer at the beginning of the time step, before the mobilisation
        dep_data.append(data_output[output_name][c][t,FromN-1])
      
        
    fig = plt.figure()
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    
    ax3.plot(range(n_class), transported_data, '-o', label = 'transported')
    ax2.plot(range(n_class), AL_data, '-o', label = 'AL')
    ax3.plot(range(n_class), tr_cap_data, '-o', label = 'Tr cap')
    ax3.plot(range(n_class), mob_data, '-v', label = 'Mobilised')
    ax1.plot(range(n_class), dep_data, '-o', label = 'Deposited')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    ax3.set_xlabel('Sediment classes', fontsize = 15)
    
    fig.set_tight_layout(True)
    fig.set_size_inches(250./fig.dpi,600./fig.dpi)
    fig.savefig(figure_folder+'per_class_FromN_'+str(FromN)+'_time'+str(t))
