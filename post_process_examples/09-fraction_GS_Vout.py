# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:52:44 2026

@author: diane
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

import matplotlib.colors as mcolors
import matplotlib.cm as cm

import os

from pathlib import Path

#matplotlib.use('Qt5Agg') 


#%%
#----------------------directories 


# Libraries 
import os
import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib.cm as cm 
import pandas as pd
import geopandas as gpd


#---------------------Path to the extended pickle output
path = "..\\cascade_results\\" 
name_simu = 'Tagliamento_bf'
name_simu_ext = 'Tagliamento_bf_ext'

#---------------------Path to the input river network (.shp) or (.csv)
path_river_network = "..\\inputs\\input_trial\\" #Path to the shp
name_river_network = "Reach_data_tag_bf.csv"

#---------------------Folder to store the plots
figure_folder = path+'figures_all_reaches_sum\\'          # where you will store the figure

if not os.path.exists(figure_folder):       
    os.makedirs(figure_folder)
       

##############################################################################



##### Make a stacked plot of the sum, x axis is the reach index

# Load basic outputs to get size classes info (dmi)
data_output = pd.read_pickle(open( path + name_simu + '.p' , "rb"))
psi = data_output['Simulation parameters']['psi']
n_class = len(psi)
dmi = 2**(-psi).reshape(-1,1)
dmi = np.squeeze(dmi)

# Load extended outputs
data_output_ext = pd.read_pickle(open( path + name_simu_ext + '.p' , "rb"))
Qbi_mob = data_output_ext['Volume out per grain sizes [m^3]']

# Total volume out summed over the years
vout_tot = np.sum(Qbi_mob, axis = 0)
# Fraction
Fir_vout = vout_tot/np.sum(vout_tot, axis = 1, keepdims=True)

n_reach, n_categories = Fir_vout.shape
colors = [plt.cm.jet(i / (n_categories - 1)) for i in range(n_categories)]


# create figure
fig = plt.figure()
ax = plt.subplot2grid(shape=(1, 4), loc=(0, 0), colspan=3)

bottom = np.zeros(n_reach)
reach_FromN = np.arange(1, n_reach + 1, 1)
for i in range(n_categories):
    ax.bar(reach_FromN, Fir_vout[:,i], bottom = bottom, color=colors[i], label=f'd = {dmi[i]} mm')
    bottom += Fir_vout[:,i]
    


ax.legend(fontsize = 9)#, bbox_to_anchor=(1, 1))
            
ax.set_xlabel('Reach index (FromN)', fontsize = 18)
ax.set_ylabel('Volume fraction per grain size', fontsize = 16)
ax.tick_params(axis='y', which='major', labelsize=15)
ax.tick_params(axis='x', which='major', labelsize=12)

         
fig.set_tight_layout(True)
fig.set_size_inches(2000./fig.dpi, 700./fig.dpi)
fig.savefig(figure_folder+'fraction_per_GS_Vout')