# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:17:11 2025

@author: diane


Plot D-CASCADE extented outputs --> per grain sizes
one graph per reach, along time

You need to have saved the extended outputs to make these plots. 

Choose between:
'Volume out per grain sizes [m^3]': total volume of sediment leaving a reach per time step per grain size (= sediment flux x time step)
'Volume in per grain sizes [m^3]': total volume of sediment entering a reach per time step per grain size
'Deposited per grain sizes [m^3]': total volume of sediment depositing in a reach per time step per grain size

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
output_name = 'Volume out per grain sizes [m^3]'   # Output available in pickle file
# 'Volume out per grain sizes [m^3]', 'Volume in per grain sizes [m^3]', 'Deposited per grain sizes [m^3]'


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


data_output = pd.read_pickle(open( path + name_simu + '.p' , "rb"))
psi = data_output['Simulation parameters']['psi']
n_class = len(psi)
dmi = 2**(-psi).reshape(-1,1)
dmi = np.squeeze(dmi)
times = np.arange(1, len(my_data[:, 0, 0]) + 1, 1)

# Choose the colormap (viridis in this case)
cmap = cm.viridis  

for i in range(my_data.shape[1]): #Loop over each reach
    reach_FromN = i + 1 #+1 because python starts at 0

    #create figure and graph axes
    fig = plt.figure()
    ax = plt.subplot(111)       

    colors = [cmap(i / n_class) for i in range(n_class)]    
    labels = [f'd = {d:.3g} mm' for d in dmi]
        
    ax.stackplot(times, my_data[:, i, :].T, linewidth = 2.5, labels = labels, colors = colors)
                    
    ax.set_title('FromN: '+str(reach_FromN), fontsize = 16)
    ax.set_xlabel('Time', fontsize = 13)
    ax.set_ylabel(output_name,fontsize = 13)
    ax.tick_params(axis='both', labelsize = 12)
    
    ax.legend(fontsize = 6)
    
    fig.set_tight_layout(True)
    fig.set_size_inches(900./fig.dpi,600./fig.dpi)
    new_name = rename_names(output_name)
    fig.savefig(figure_folder + str(new_name)+'_'+str(reach_FromN))
    fig.clf()
    plt.close("all")





