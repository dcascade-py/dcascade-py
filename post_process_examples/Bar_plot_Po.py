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

import matplotlib.colors as mcolors
import matplotlib.cm as cm

import os

from pathlib import Path

#matplotlib.use('Qt5Agg') 


#%%
#----------------------directories 



path = "..\\62_test_deposit0_3y\\"
# path = "..\\Cascade_results\\"

name_simu = 'WC_bankfull'
name_output=name_simu+'.p'
name_simu_ext = 'WC_bankfull_ext'
name_ext_output = name_simu_ext+'.p'
 
 
path_river_network = "..\\Inputs\\09-shp_with_smoothed_slopes\\from_AIPO_CS\\"
name_river_network = "Po_river_network.shp"

path_river_network2 = "..\\Inputs\\09-shp_with_smoothed_slopes\\from_DEM2021\\"
name_river_network2 = "Po_river_network.shp"


#---------------------Make figure folder
figure_folder = path+'figures_'+name_simu+'\\'

if not os.path.exists(figure_folder):       
    os.makedirs(figure_folder)
    
#---------------------loading data

# load network for reference 
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefine from shp format
ReachData = ReachData.sort_values(by = 'FromN', ignore_index = True)
ReachData_Po = ReachData[ReachData['River'] == 'Po'] # select Po
Po_idx = ReachData_Po['FromN'].values
Po_idx = Po_idx.astype(int)
Po_names=ReachData_Po['Reach'].str[3:]

# All tributaries
ReachData_trib = ReachData[ReachData['River'] != 'Po'] # select Trib
Trib_idx = ReachData_trib['FromN'].values
Trib_po_idx = ReachData_trib['ToN'].values
Trib_idx = Trib_idx.astype(int)
Trib_names = ReachData_trib['Reach'].str[:-2]
Trib_names = Trib_names.str.lower()
Trib_names.loc[0] = 's. di lanzo'
Trib_names.loc[46] = 'd. baltea'



# load network for reference (slope 2)
ReachData2 = gpd.GeoDataFrame.from_file(path_river_network2 + name_river_network2) #read shapefine from shp format
ReachData2 = ReachData2.sort_values(by = 'FromN', ignore_index = True )
ReachData_Po2 = ReachData2[ReachData2['River'] == 'Po'] # select Po
Po_idx2 = ReachData_Po2['FromN'].values
Po_idx2 = Po_idx2.astype(int)
Po_names2=ReachData_Po2['Reach'].str[3:]

# All tributaries
ReachData_trib2 = ReachData2[ReachData2['River'] != 'Po'] # select Trib
Trib_idx2 = ReachData_trib2['FromN'].values
Trib_po_idx2 = ReachData_trib2['ToN'].values
Trib_idx2 = Trib_idx2.astype(int)
Trib_names2 = ReachData_trib2['Reach'].str[:-2]
Trib_names2 = Trib_names2.str.lower()
Trib_names2.loc[0] = 's. di lanzo'


#Load Q file
path_Q = "..\\Inputs\\"
name_q = 'Po_Qdaily_16y.csv' # the order is incresing FromN - ToNode in ReachData 

Q = pd.read_csv(open( path_Q + name_q , "rb"))









#------------------usefull function

def rename_names(output_name):
    '''For exemple, renames 'Volume out [m^3]' into 'Volume out' 
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

  
#--------------------plots 



############################### bar plot with initial sediment fractions (fi)

sed_range = [-8, 3]  # range of sediment sizes - in krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5). 
n_classes = 12        # number of classes
psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=true).astype(float)
dmi = 2**(-psi).reshape(-1,1)

# fi_r = np.load('fi_r_tuned.npy')

data_output_ext = pd.read_pickle(open( path + name_ext_output, "rb")) 
fi_r_init = data_output_ext['fi_al'][0,:,:]


n_rows, n_categories = fi_r_init.shape

colors = [plt.cm.jet(i / (n_categories - 1)) for i in range(n_categories)]

# axe x
# x = np.arange(n_rows)

fig = plt.figure()

ax1 = plt.subplot2grid(shape=(1, 4), loc=(0, 0), colspan=3)

bottom = np.zeros(len(po_idx))
for i in range(n_categories):
    ax1.bar(po_idx, fi_r_init[po_idx - 1, i], bottom = bottom, color=colors[i], label=f'd = {dmi[i]} mm')
    bottom += fi_r_init[po_idx - 1, i]
    
#add tributaries tonode position on same plot as vertical lines:
#find the value in the middle of y axis
y_min, y_max = plt.ylim()
y_middle = (y_max) / 1.5
  
trib_data = reachdata[reachdata['river'] != 'po']
for _, row in trib_data.iterrows():
    ax1.axvline(x=row['ton'], color = 'grey', linestyle = '--', zorder = 1000)
    plt.text(row['ton'], y_middle, row['river'], rotation=90, verticalalignment='center', color = 'grey', fontsize = 9, zorder = 1000)
    
#add isola serafini as vertical line
ax1.axvline(x=26, color='red', linestyle='-', zorder = 1000)
plt.text(26, y_middle, 'isola serafini', rotation=90, verticalalignment='center', color='red', fontsize = 11, zorder = 1000)




# Add more graduation along x axis
ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.xaxis.set_major_formatter('{x:.0f}')
ax1.xaxis.set_minor_locator(MultipleLocator(1))    

# Update the x-tick labels with the Po names
ax1.set_xticks(Po_idx)
ax1.set_xticklabels(Po_names, fontsize=8, rotation = 45) 

ax1.set_ylabel('Volume fraction per grain size', fontsize = 11)
ax1.tick_params(axis='y', which='major', labelsize=10)
ax1.tick_params(axis='x', which='major', labelsize=9)


ax_trib = plt.subplot2grid(shape=(1, 4), loc=(0, 3), colspan=1)

bottom = np.zeros(len(Trib_idx))
for i in range(n_categories):
    ax_trib.bar(np.arange(len(Trib_idx)), Fi_r_init[Trib_idx - 1, i], bottom = bottom, color=colors[i], label=f'd = {float(dmi[i])} mm')
    bottom += Fi_r_init[Trib_idx - 1, i]

# Update the x-tick labels with the tributary names
ax_trib.set_xticks(np.arange(len(Trib_idx)))
ax_trib.set_xticklabels(Trib_names, fontsize=8, rotation = 75) 
ax_trib.tick_params(axis='y', which='major', labelsize=10)
ax_trib.tick_params(axis='x', which='major', labelsize=9)  

ax_trib.legend(fontsize = 8, bbox_to_anchor=(1, 1))

fig.set_tight_layout(True)
fig.set_size_inches(1700./fig.dpi, 500./fig.dpi)
fig.savefig(figure_folder+'Fi_r_init_t0')



######################## Bar plot with sdiment fraction of the volume out

sed_range = [-8, 3]  # range of sediment sizes - in Krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5). 
n_classes = 12        # number of classes
psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)
dmi = 2**(-psi).reshape(-1,1)


data_output_ext = pd.read_pickle(open( path + name_ext_output, "rb")) 

Qbi_mob = data_output_ext['Volume out per grain sizes [m^3]']

vout_tot = np.sum(Qbi_mob, axis = 0)

Fir_vout = vout_tot/np.sum(vout_tot, axis = 1, keepdims=True)


n_rows, n_categories = Fir_vout.shape

colors = [plt.cm.jet(i / (n_categories - 1)) for i in range(n_categories)]

# Axe X
# x = np.arange(n_rows)

fig = plt.figure()

ax1 = plt.subplot2grid(shape=(1, 4), loc=(0, 0), colspan=3)

bottom = np.zeros(len(Po_idx))
for i in range(n_categories):
    ax1.bar(Po_idx, Fir_vout[Po_idx - 1, i], bottom = bottom, color=colors[i], label=f'd = {dmi[i]} mm')
    bottom += Fir_vout[Po_idx - 1, i]
    
#add tributaries ToNode position on same plot as vertical lines:
#Find the value in the middle of y axis
y_min, y_max = plt.ylim()
y_middle = (y_max) / 1.5
  
trib_data = ReachData[ReachData['River'] != 'Po']
for _, row in trib_data.iterrows():
    ax1.axvline(x=row['ToN'], color = 'grey', linestyle = '--', zorder = 1000)
    plt.text(row['ToN'], y_middle, row['River'], rotation=90, verticalalignment='center', color = 'grey', fontsize = 9, zorder = 1000)
    
#add Isola Serafini as vertical line
ax1.axvline(x=26, color='red', linestyle='-', zorder = 1000)
plt.text(26, y_middle, 'Isola Serafini', rotation=90, verticalalignment='center', color='red', fontsize = 11, zorder = 1000)



# Add more graduation along x axis
ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.xaxis.set_major_formatter('{x:.0f}')
ax1.xaxis.set_minor_locator(MultipleLocator(1))    

# Update the x-tick labels with the Po names
ax1.set_xticks(Po_idx)
ax1.set_xticklabels(Po_names, fontsize=8, rotation = 45) 

ax1.set_ylabel('Volume fraction per grain size', fontsize = 11)
ax1.tick_params(axis='y', which='major', labelsize=10)
ax1.tick_params(axis='x', which='major', labelsize=9)



ax_trib = plt.subplot2grid(shape=(1, 4), loc=(0, 3), colspan=1)

bottom = np.zeros(len(Trib_idx))
for i in range(n_categories):
    ax_trib.bar(np.arange(len(Trib_idx)), Fir_vout[Trib_idx - 1, i], bottom = bottom, color=colors[i], label=f'd = {float(dmi[i])} mm')
    bottom += Fir_vout[Trib_idx - 1, i]

# Update the x-tick labels with the tributary names
ax_trib.set_xticks(np.arange(len(Trib_idx)))
ax_trib.set_xticklabels(Trib_names, fontsize=8, rotation = 75) 
ax_trib.tick_params(axis='y', which='major', labelsize=10)
ax_trib.tick_params(axis='x', which='major', labelsize=9)  

ax_trib.legend(fontsize = 8, bbox_to_anchor=(1, 1))

fig.set_tight_layout(True)
fig.set_size_inches(1700./fig.dpi, 500./fig.dpi)
fig.savefig(figure_folder+'Fi_r_vout')


