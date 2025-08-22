# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:17:11 2025

@author: diane


Plot D-CASCADE extented outputs --> per initial provenance
one graph per reach, along time

You need to have saved the extended outputs to make these plots. 

Choose between:
'Qbi_mob [m^3]':    total volume of sediment leaving a reach per time step (full matrice) (= sediment flux x time step)
'Qbi_tr [m^3]':     total volume of sediment entering a reach per time step (full matrice)

These are the full storing matrice. They have the shape: [time] x array(prov_reach, reach, sed class)

"""

# Libraries 
import os
import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib.cm as cm 
import pandas as pd
import geopandas as gpd


#---------------------Path to the extended pickle output

path = "..\\cascade_results\\" 
name_simu = 'Vjosa_test'
name_simu_ext = 'Vjosa_test_ext'

#---------------------Folder to store the plots
figure_folder = path+'figures_per_reach\\'          # where you will store the figure

if not os.path.exists(figure_folder):       
    os.makedirs(figure_folder)
       
#--------------------Output name you want to plot
output_name = 'Qbi_mob [m^3]'   # Output available in pickle file
# 'Qbi_mob [m^3]', 'Qbi_tr [m^3]'


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

data_output_ext = pd.read_pickle(open( path + name_simu_ext + '.p' , "rb"))
my_data = data_output_ext[output_name]
n_reach = len(my_data[0][0, :, 0])
n_time = len(my_data)
times = np.arange(1, n_time + 1, 1)


# Reshape my data to remove the per grain size component
my_data_per_prov = np.zeros((n_time, n_reach, n_reach))
for t in range(n_time - 1):
    # Sum over provenances (axe 0)
    my_data_per_prov[t,:,:] = np.sum(my_data[t], axis = (2))

# Choose the colormap (viridis in this case)
cmap = cm.viridis  

for i in range(n_reach): #Loop over each reach
    reach_FromN = i + 1 #+1 because python starts at 0
    
    data_reach = my_data_per_prov[:, :, i]

    #create figure and graph axes
    fig = plt.figure()
    ax = plt.subplot(111)       

    colors = np.array([cmap(i / n_reach) for i in range(n_reach)])
    
    #filter non zero provenance
    non_zero_prov = np.any(data_reach != 0, axis=0)  
    filtered_labels = np.arange(1, n_reach + 1)[non_zero_prov]
    filtered_data = data_reach[:, non_zero_prov].T
    filtered_colors = colors[non_zero_prov]
    
        
    ax.stackplot(times, filtered_data, linewidth = 2.5, labels = filtered_labels, colors = filtered_colors)
                    
    ax.set_title('FromN: '+str(reach_FromN), fontsize = 16)
    ax.set_xlabel('Time', fontsize = 13)
    ax.set_ylabel(output_name,fontsize = 13)
    ax.tick_params(axis='both', labelsize = 12)
    
    ax.legend(fontsize = 6, title='Initial provenance reach')
    
    fig.set_tight_layout(True)
    fig.set_size_inches(900./fig.dpi,600./fig.dpi)
    new_name = rename_names(output_name)
    fig.savefig(figure_folder + str(new_name)+'_'+str(reach_FromN))
    fig.clf()
    plt.close("all")





