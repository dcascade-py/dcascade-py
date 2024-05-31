# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:32:50 2024

@author: Asus
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib
from matplotlib.pyplot import text
import copy
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# matplotlib.use('Qt5Agg')

mobilized_volumes = {}  # List to store mobilized volumes from each file


path = "E:\\cascade\\cascade_results\\"
name_output = "Po_results_H03newslope.p"

data_output = pickle.load(open(path + name_output, "rb"))

mobilized_volume_original = data_output['Mobilized volume [m^3]']

path_river_network = "E:\\cascade\\input\\"
name_river_network = "Po_rivernet_grainsze_new_d.shp"

figure_folder = "E:\\cascade\\figure\\"


# load network for reference
ReachData = gpd.GeoDataFrame.from_file(
    path_river_network + name_river_network)  # read shapefine from shp format
ReachData = ReachData.sort_values(by='FromN', ignore_index=True)
ReachData_Po = ReachData[ReachData['River'] == 'Po']  # select Po
Po_idx = ReachData_Po['FromN'].values
Po_idx = Po_idx.astype(int)


# Loop through each data_output file
mobilized_volumes = {}  # List to store mobilized volumes from each file
##change the name according to the choice of output that needs to be extracted##

percentages = [5, 10, 15, 20]
for percent in percentages:
    # Load the pickled file
    path = "E:\\cascade\\slope_result\\"
    filename = f'output_change_{percent}percent.pkl'
    data_output = pickle.load(open(path + filename, "rb"))

    # # Extract the mobilized volume from the current data_output
    # mobilized_volume = data_output['Mobilized volume [m^3]']   #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    # mobilized_volumes[percent] = mobilized_volume

    # Extract the transported volume from the current data_output
    # 'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    mobilized_volume = data_output['Mobilized volume [m^3]']
    mobilized_volumes[percent] = mobilized_volume

annual_sums = {}
for percent, mobilized_volume in mobilized_volumes.items():
    # Slice the mobilized volume array to consider only the first 365 rows and the columns indicated by Po_idx
    mobilized_volume_year = mobilized_volume[:365, Po_idx - 1]
    # mobilized_volume_year = mobilized_volume[365:731, Po_idx - 1]

    # Calculate the annual sum along the specified columns
    annual_sum = np.nansum(mobilized_volume_year, axis=0)

    # Store the annual sum for the current percentage
    annual_sums[percent] = annual_sum


mobilized_volume_year_original = mobilized_volume_original[:365, Po_idx - 1]
# mobilized_volume_year_original = mobilized_volume_original[365:731, Po_idx - 1]

# Calculate the annual sum along the specified columns
annual_sum_original = np.nansum(mobilized_volume_year_original, axis=0)


# Plot the annual sum from outside the loop
fig = plt.figure()
ax = plt.subplot(111)

plt.plot(Po_idx, annual_sum_original, label="Original",
         color='black', linestyle='--')

# Plot the annual sums from inside the loop
for percent, annual_sum in annual_sums.items():
    plt.plot(Po_idx, annual_sum, label=f"{percent}% Decrease")

plt.xlabel("Po_idx")
plt.ylabel("Annual Sum of Mobilized Volume [m^3]")
plt.title("Annual Sum of Mobilized Volumes 2019")
plt.legend()
plt.grid(True)

# add tributaries ToNode position on same plot as vertical lines:
trib_data = ReachData[ReachData['River'] != 'Po']
for _, row in trib_data.iterrows():
    ax.axvline(x=row['ToN'], color='grey', linestyle='--')
    plt.text(row['ToN'], 0.01, row['River'], rotation=90,
             verticalalignment='center', color='grey')
    
    
    
## plot along time for all the reach    
    
# Loop through each data_output file
mobilized_volumes = {}  # List to store mobilized volumes from each file
##change the name according to the choice of output that needs to be extracted##

percentages = [-5, -10, -15, -20, -50]
for percent in percentages:
    # Load the pickled file
    path = "E:\\cascade\\slope_result_decrease\\"
    filename = f'output_change_{percent}percent.pkl'
    data_output = pickle.load(open(path + filename, "rb"))

    # # Extract the mobilized volume from the current data_output
    # mobilized_volume = data_output['Mobilized volume [m^3]']   #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    # mobilized_volumes[percent] = mobilized_volume

    # Extract the transported volume from the current data_output
    # 'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    mobilized_volume = data_output['Daily trasport capacity [m^3/day]']
    mobilized_volumes[percent] = mobilized_volume

selected_reaches = [7]

plt.figure(figsize=(10, 6))
for percent, mobilized_volume in mobilized_volumes.items():
    # Slice the mobilized volume array to consider only the selected reaches
    mobilized_volume_selected_reaches = mobilized_volume[:, selected_reaches]
    # Plot mobilized volume values along the timestep for selected reaches
    for reach_idx in range(len(selected_reaches)):
        # plt.plot(mobilized_volume_selected_reaches[:, reach_idx], label=f'Percent Increase: {percent}, Reach: {selected_reaches[reach_idx] + 1}')
        plt.scatter(range(len(mobilized_volume_selected_reaches)), mobilized_volume_selected_reaches, label=f'Percent Decrease: {percent}')
# # Create area plot
#plt.fill_between(range(len(mobilized_volume_selected_reaches)), 0, mobilized_volume_selected_reaches[:, reach_idx], alpha=0.3)


mobilized_volume_original_selected_reaches = mobilized_volume_original[:, selected_reaches]

# Plot mobilized volume values along the timestep for selected reaches (original)
for reach_idx in range(len(selected_reaches)):
    plt.plot(mobilized_volume_original_selected_reaches[:, reach_idx], label=f'Original, Reach: {selected_reaches[reach_idx] + 1}', linestyle='--', color='black')

plt.xlabel('Timestep')
plt.ylabel('Daily trasport capacity [m^3/day]')
plt.title('Daily trasport capacity [m^3/day] for Selected Reaches and Percentages')
plt.legend()
plt.grid(True)
plt.show()


## plot along time for all the reach    
    
# Loop through each data_output file
mobilized_volumes = {}  # List to store mobilized volumes from each file
##change the name according to the choice of output that needs to be extracted##

percentages = [-5, -10, -15, -20, -50]
for percent in percentages:
    # Load the pickled file
    path = "E:\\cascade\\slope_result_decrease\\"
    filename = f'output_change_{percent}percent.pkl'
    data_output = pickle.load(open(path + filename, "rb"))

    # # Extract the mobilized volume from the current data_output
    # mobilized_volume = data_output['Mobilized volume [m^3]']   #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    # mobilized_volumes[percent] = mobilized_volume

    # Extract the transported volume from the current data_output
    # 'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    mobilized_volume = data_output['Daily trasport capacity [m^3/day]']
    mobilized_volumes[percent] = mobilized_volume

selected_reaches = [7]

plt.figure(figsize=(10, 6))
for percent, mobilized_volume in mobilized_volumes.items():
    # Slice the mobilized volume array to consider only the selected reaches
    mobilized_volume_selected_reaches = mobilized_volume[selected_reaches, :]
    # Plot mobilized volume values along the timestep for selected reaches
    for reach_idx in range(len(selected_reaches)):
        # plt.plot(mobilized_volume_selected_reaches[:, reach_idx], label=f'Percent Increase: {percent}, Reach: {selected_reaches[reach_idx] + 1}')
        plt.scatter(range(len(mobilized_volume_selected_reaches)), mobilized_volume_selected_reaches, label=f'Percent Decrease: {percent}')
# # Create area plot
#plt.fill_between(range(len(mobilized_volume_selected_reaches)), 0, mobilized_volume_selected_reaches[:, reach_idx], alpha=0.3)


mobilized_volume_original_selected_reaches = mobilized_volume_original[selected_reaches,:]

# Plot mobilized volume values along the timestep for selected reaches (original)
for reach_idx in range(len(selected_reaches)):
    plt.plot(mobilized_volume_original_selected_reaches[reach_idx,:], label=f'Original, Reach: {selected_reaches[reach_idx] + 1}', linestyle='--', color='black')

plt.xlabel('Timestep')
plt.ylabel('Daily trasport capacity [m^3/day]')
plt.title('Daily trasport capacity [m^3/day] for Selected Reaches and Percentages')
plt.legend()
plt.grid(True)
plt.show()






