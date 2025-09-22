# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:17:11 2025

@author: diane


Plot D-CASCADE extented outputs --> per grain sizes
total sum, plotted along reach indexes, per grain size

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
name_simu = 'Rhone_test'
name_simu_ext = 'Rhone_test_ext'

#---------------------Path to the input river network (.shp) or (.csv)
path_river_network = "..\\inputs\\input_trial\\" #Path to the shp
name_river_network = "Rhone.shp"

#---------------------Folder to store the plots
figure_folder = path+'figures_all_reaches_sum\\'          # where you will store the figure

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



##### Make a stacked plot of the sum, x axis is the reach index

# Load extended outputs
data_output_ext = pd.read_pickle(open( path + name_simu_ext + '.p' , "rb"))
my_data = data_output_ext[output_name]

# Load basic outputs to get size classes info (dmi)
data_output = pd.read_pickle(open( path + name_simu + '.p' , "rb"))
psi = data_output['Simulation parameters']['psi']
n_class = len(psi)
dmi = 2**(-psi).reshape(-1,1)
dmi = np.squeeze(dmi)

# Choose the colormap (viridis in this case)
cmap = cm.viridis 

#create figure and graph axes
fig = plt.figure()
ax = plt.subplot(111) 

# Sum all time steps
my_sum = np.nansum(my_data, axis = 0) 
n_reach = my_sum.shape[0]
reach_FromN = np.arange(1, n_reach + 1, 1)

# Colors and labels
colors = [cmap(i / n_class) for i in range(n_class)]    
labels = [f'd = {d:.3g} mm' for d in dmi]

ax.stackplot(reach_FromN, my_sum.T, linewidth = 2.5, labels = labels, colors = colors)

ax.legend(fontsize = 9)#, bbox_to_anchor=(1, 1))
            
ax.set_xlabel('Reach index (FromN)', fontsize = 18)
ax.set_ylabel(output_name, fontsize = 16)
ax.tick_params(axis='y', which='major', labelsize=15)
ax.tick_params(axis='x', which='major', labelsize=12)

         
fig.set_tight_layout(True)
fig.set_size_inches(2000./fig.dpi, 700./fig.dpi)
new_name = rename_names(output_name)
fig.savefig(figure_folder+str(new_name)+'_sum')







