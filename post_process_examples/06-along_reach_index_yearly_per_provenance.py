# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:17:11 2025

@author: diane


Plot D-CASCADE extented outputs --> per provenance
total sum, plotted along reach indexes, per provenance

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


years = '2004_2006'
eq = 'W&P'
dep_l = 10
layer = 30
trib = '15'

# ---------------------Path to the pickle output
path = f'K:/Labo/20_OSR/Rhone_Fabio_Schneider/Rhone_Slocal_trib{trib}_res{years}_{eq}_layer{layer}cm/'
name_simu = f'Rhone_Martinez_res{years}_{eq}_{dep_l}m_trib{trib}_layer{layer}cm'
name_simu_ext = f'Rhone_Martinez_res{years}_{eq}_{dep_l}m_trib{trib}_layer{layer}cm_ext'

#---------------------Path to the input river network (.shp) or (.csv)
path_river_network = '../inputs/Rhone_river/Network/' #Path to the shp
name_river_network = "rhone_DCASCADE.shp"


#---------------------Folder to store the plots
figure_folder = path+f'figures_all_reaches_sum_{years}_{eq}\\'          # where you will store the figure

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



##### Make a stacked plot of the sum, x axis is the reach index

data_output_ext = pd.read_pickle(open( path + name_simu_ext + '.p' , "rb"))
my_data = data_output_ext[output_name]
n_reach = len(my_data[0][0, :, 0])
reach_FromN = np.arange(1, n_reach + 1, 1)
n_time = len(my_data)
times = np.arange(1, n_time + 1, 1)

# Reshape my data to remove the per grain size component
my_data_per_prov = np.zeros((n_time, n_reach, n_reach))
for t in range(n_time - 1):
    # Sum over provenances (axe 0)
    my_data_per_prov[t,:,:] = np.sum(my_data[t], axis = (2))

# Choose the colormap (viridis in this case)
cmap = cm.viridis 

#create figure and graph axes
fig = plt.figure()
ax = plt.subplot(111) 

# Sum all time steps
my_sum = np.nansum(my_data_per_prov, axis = 0) 
n_reach = my_sum.shape[0]
reach_FromN = np.arange(1, n_reach + 1, 1)

# Colors and labels
colors = np.array([cmap(i / n_reach) for i in range(n_reach)]) 

ax.stackplot(reach_FromN, my_sum, linewidth = 2.5, labels = reach_FromN, colors = colors)

ax.legend(fontsize = 7, ncol = 3)#, bbox_to_anchor=(1, 1))
            
ax.set_xlabel('Reach index (FromN)', fontsize = 18)
ax.set_ylabel(output_name, fontsize = 16)
ax.tick_params(axis='y', which='major', labelsize=15)
ax.tick_params(axis='x', which='major', labelsize=12)

         
fig.set_tight_layout(True)
fig.set_size_inches(2000./fig.dpi, 700./fig.dpi)
new_name = rename_names(output_name)
fig.savefig(figure_folder+str(new_name)+'_sum_per_provenance')







