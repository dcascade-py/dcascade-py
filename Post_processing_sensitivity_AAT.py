# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:32:50 2024

@author: Sahansila
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.pyplot import text
import copy
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# matplotlib.use('Qt5Agg'
import os

# Define the directory where the pickled files are stored
directory = "E:\\cascade\\sensitivity_analysis_results\\combined_random_sampling\\"

# Initialize an empty dictionary to store the data outputs
output_data_dict = {}

# Iterate over each pickled file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        # Split the filename by underscores and remove the extension
        parts = filename.split("_")
        # Extract active_width_change and slope_change from the filename
        awc_percent = int(parts[3])
        sc_percent = int(parts[5].split(".")[0])
        
        # Load the pickled file
        with open(os.path.join(directory, filename), "rb") as file:
            data_output = pickle.load(file)
        
        # Store the data_output in the dictionary
        output_data_dict[(awc_percent, sc_percent)] = data_output

# A dictionary containing the data outputs for each combination
# Data_output for a specific combination can be access like this:
# data_output = output_data_dict[(awc_percent, sc_percent)]

         
# Initialize an empty dictionary to store the mobilized volumes/  for each combination
# mobilized_volumes_dict = {}
# transported_volume_dict = {}
D50_active_layer_dict = {}
# D50_mobilised_layer_dict = {}

# Iterate over each combination in output_data_dict
for combination, data_output in output_data_dict.items():
    # Extract the desired output from the data_output
    output_name = 'D50 active layer [m]'    #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    # mobilized_volume = data_output[output_name]
    # transported_volume = data_output[output_name]
    D50_active_layer = data_output[output_name]
    # D50_mob_layer = data_output[output_name]
    

    # Store the output in the respective dictionary according to thier name
    # mobilized_volumes_dict[combination] = mobilized_volume
    # transported_volume_dict[combination] = transported_volume
    D50_active_layer_dict[combination] = D50_active_layer
    # D50_mobilised_layer_dict[combination] = D50_mob_layer
    
# data_output for a specific combination can be access like this:
# mobilized_volume = mobilized_volumes_dict[(awc_percent, sc_percent)]

# Initialize an empty dictionary to store the total sum/ median/ averages  of chosen data output for each combination
# total_mobilized_volume_dict = {}
# total_transported_volume_dict = {}
median_D50_active_layer_dict = {}
# median_D50_mobilised_layer_dict = {}

# Iterate over each combination in chosen output's dictionary
# for combination, mobilized_volume_array in mobilized_volumes_dict.items():
# for combination, transported_volume_array in transported_volume_dict.items():
for combination, D50_active_layer_array in D50_active_layer_dict.items():    
    # Sum or median of the chosen outputs for the current combination along axis=0
    # total_sum = np.sum(mobilized_volume_array, axis=0)
   median_value = np.median(D50_active_layer_array, axis=0)
    
    
    # Store the total sum/median in the total/median dictionary
    # total_mobilized_volume_dict[combination] = total_sum
   median_D50_active_layer_dict[combination] = median_value

# total/median dict contains the total sum / median of output for each combination
# total sum/ median for a specific combination can be access like this:
# total_sum = total_mobilized_volume_dict[(awc_percent, sc_percent)]



# # Iterate over each combination in total_mobilized_volume_dict
# for combination, total_sum in total_mobilized_volume_dict.items():
#     # Get the active width and slope from the combination tuple
#     awc_percent, sc_percent = combination
    
#     # Create a new figure for each combination
#     plt.figure(figsize=(10, 6))
    
#     # Plot the total mobilized volume for each reach
#     plt.plot(total_sum, marker='o', linestyle='-')
    
#     # Add labels and title
#     plt.xlabel('Reach')
#     plt.ylabel('Total Mobilized Volume')
#     plt.title(f'Total Mobilized Volume for Active Width Change {awc_percent}% and Slope Change {sc_percent}%')
    
#     # Show grid
#     plt.grid(True)
     
#     # Show the plot
#     plt.show()
 

import pandas as pd

# Initialize a dictionary to store data for each river reach
data_dict = {}

# Iterate over each combination in total_mobilized_volume_dict
# for combination, mobilized_volume_sum in total_mobilized_volume_dict.items():
for combination, D50_active_layer_median in median_D50_active_layer_dict.items():
    # Extract active width and slope from the combination tuple
    active_width, slope = combination
    
    # Check if the combination is already in the dictionary, if not, add it
    if combination not in data_dict:
        data_dict[combination] = {}
    
    # Iterate over the river reaches and store the mobilized volume for each
    # for river_reach_index, mobilized_volume in enumerate(mobilized_volume_sum):
    for river_reach_index, D50_active_layer in enumerate(D50_active_layer_median):    
        # Create column names for each river reach
        column_name = f'River Reach {river_reach_index+ 1}'
        
        # Add the mobilized volume for this river reach to the dictionary
        # data_dict[combination][column_name] = mobilized_volume
        data_dict[combination][column_name] = D50_active_layer
        
# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict).T  # Transpose the DataFrame to have combinations as rows and river reaches as columns

# Add column names for active width and slope
df.index.names = ['Active Width Change', 'Slope Change']

# Reset index to have columns for active width and slope
df.reset_index(inplace=True)

# Print the DataFrame
print(df)




# bubble plot showing the intensity of the total mobilized volume

# Determine the number of river reaches
num_reaches = 43

# Determine the number of rows and columns for the subplot grid
num_rows = 3
num_cols = 3  # Adjust as needed to accommodate all 43 river reaches

# Calculate the number of figures needed
num_figures = -(-num_reaches // (num_rows * num_cols))

# Iterate over each figure
for fig_num in range(num_figures):
    # Calculate the start and end index for the current figure
    start_index = fig_num * num_rows * num_cols
    end_index = min((fig_num + 1) * num_rows * num_cols, num_reaches)

    # Create a figure and a subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Iterate over each river reach and create a separate plot
    for i, reach_index in enumerate(range(start_index, end_index)):
        # Extract data for the current river reach
        reach_data = df[df.columns[2 + reach_index]]  # Assuming the first two columns are 'Active Width Change' and 'Slope Change'

        # Define colormap and normalization for this river reach
        cmap = plt.cm.viridis  # You can choose any colormap you like
        normalize = plt.Normalize(vmin=reach_data.min(), vmax=reach_data.max())  # Normalize values for color mapping

        # Create the scatter plot for this river reach
        ax = axes[i]
        ax.scatter(df['Active Width Change'], df['Slope Change'], s=100, c=reach_data, cmap=cmap, norm=normalize, alpha=0.5)

        # Set labels and title for the subplot
        ax.set_xlabel('Active Width Change')
        ax.set_ylabel('Slope Change')
        ax.set_title(f'River Reach {reach_index + 1}')

        # Add colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalize, cmap=cmap), ax=ax)
        cbar.set_label('Value')

    # Hide unused subplots
    for i in range(end_index - start_index, num_rows * num_cols):
        axes[i].axis('off')

    # Adjust layout and spacing
    plt.tight_layout(pad=3.0)

    # Show the plot
    plt.show()

    plt.savefig('E:\\cascade\\combined_random_sampling\\figures\\')
    
    
## for every slope change, one active width chnage line

# Determine the number of river reaches
num_reaches = 43

# Determine the number of rows and columns for the subplot grid
num_rows = 3
num_cols = 3  # Adjust as needed to accommodate all 43 river reaches

# Calculate the number of figures needed
num_figures = -(-num_reaches // (num_rows * num_cols))

folder_path = "E:\\ascade\combined_random_sampling\figures\D50 active layer"

# Iterate over each figure
for fig_num in range(num_figures):
    # Calculate the start and end index for the current figure
    start_index = fig_num * num_rows * num_cols
    end_index = min((fig_num + 1) * num_rows * num_cols, num_reaches)

    # Create a figure and a subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Iterate over each river reach and create a separate plot
    for i, reach_index in enumerate(range(start_index, end_index)):
        # Extract data for the current river reach
        reach_data = df.iloc[:, [0, 1, reach_index + 2]]  # Extracting columns for 'Active Width Change', 'Slope Change', and mobilized volume values

        # Group by 'Active Width Change %' and plot mobilized volume values against 'Slope Change %'
        for width_change, group in reach_data.groupby('Active Width Change'):
            # Sort the group by 'Slope Change %' before plotting
            group_sorted = group.sort_values(by='Slope Change')
            axes[i].plot(group_sorted['Slope Change'], group_sorted.iloc[:, 2], label=f'Width Change: {width_change}%')

        # Set labels and title for the subplot
        axes[i].set_xlabel('Slope Change %')
        axes[i].set_ylabel('D50 active layer [m]') #change the label according to the chosen output
        axes[i].set_title(f'River Reach {reach_index + 1}')
        axes[i].legend()

    # Hide unused subplots
    for i in range(end_index - start_index, num_rows * num_cols):
        axes[i].axis('off')

    # Adjust layout and spacing
    plt.tight_layout(pad=3.0)

    # Show the plot
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a DataFrame named df with columns 'Active Width Change', 'Slope Change', and columns for each river reach starting from the third column

# Choose the river reach index
reach_index = 0  # Index of the river reach to plot

# Extract data for the selected river reach
reach_data = df.iloc[:, [0, 1, reach_index + 2]]  # Extracting columns for 'Active Width Change', 'Slope Change', and mobilized volume values

# Handle missing values
reach_data = reach_data.dropna()  # Or reach_data = reach_data.fillna(0) to fill missing values with 0

# Group data by 'Active Width Change' and collect corresponding 'Slope Change' values
grouped_data = reach_data.groupby('Active Width Change')['Slope Change'].apply(list).reset_index()

# Plot heatmap for each group
for i, group in grouped_data.iterrows():
    active_width_change = group['Active Width Change']
    slope_change_values = group['Slope Change']
    
    # Create a meshgrid for x and y values
    X, Y = np.meshgrid(range(len(slope_change_values)), slope_change_values)
    
    # Plot heatmap for the group
    plt.figure(figsize=(8, 6))
    heatmap = plt.contourf(X, Y, Y, cmap='viridis')
    plt.colorbar(heatmap, label='Slope Change')
    plt.xlabel('Data Point Index')
    plt.ylabel('Slope Change')
    plt.title(f'Heatmap for Active Width Change = {active_width_change}')
    plt.show()




    
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a DataFrame named df with columns 'Active Width Change', 'Slope Change', and columns for each river reach starting from the third column

# Choose the river reach index
reach_index = 0  # Index of the river reach to plot

# Extract data for the selected river reach
reach_data = df.iloc[:, [0, 1, reach_index + 2]]  # Extracting columns for 'Active Width Change', 'Slope Change', and mobilized volume values

# Handle missing values
reach_data = reach_data.dropna()  # Or reach_data = reach_data.fillna(0) to fill missing values with 0

# Reshape data for plotting heatmap
x_values = reach_data['Active Width Change'].values
y_values = reach_data['Slope Change'].values
mobilized_volume = reach_data.iloc[:, 2].values

# Create a meshgrid for x and y values
X, Y = np.meshgrid(x_values, y_values)

# Reshape mobilized volume array to match the shape of the grid
mobilized_volume_grid = mobilized_volume.reshape(X.shape)

# Plot the heatmap for the selected river reach
plt.figure(figsize=(10, 8))
heatmap = plt.contourf(X, Y, mobilized_volume_grid, cmap='viridis')
plt.colorbar(heatmap, label='Mobilized Volume')
plt.xlabel('Active Width Change')
plt.ylabel('Slope Change')
plt.title(f'Heatmap for River Reach {reach_index + 1}')
plt.show()  
 
    



# # Define the active width change for which you want to plot mobilized volume against slope change
# target_awc = -20  # Assuming the target active width change is 0

# # Filter the DataFrame to include only rows where the active width change matches the target value
# filtered_df = df[df['Active Width Change'] == target_awc]

# # Get the number of river reaches
# num_river_reaches = 33 # Assuming you have 10 river reaches

# # Create a new figure for the plot
# plt.figure(figsize=(10, 8))

# # Set the width of each bar
# bar_width = 0.04

# # Set the x locations for the groups
# x = np.arange(len(filtered_df['Slope Change']))

# # Plot mobilized volume against slope change for each river reach using side-by-side bar plots
# for i in range(num_river_reaches):
#     plt.bar(x + i * bar_width, filtered_df[f'River Reach {i}'], label=f'River Reach {i+1}', width=bar_width, alpha=0.7)

# # Add labels and title
# plt.xlabel('Slope Change')
# plt.ylabel('Total Mobilized Volume')
# plt.title(f'Side-by-Side Bar Plot: Mobilized Volume vs. Slope Change for Active Width Change {target_awc}')

# # Set x-axis tick labels
# plt.xticks(x + bar_width * (num_river_reaches - 1) / 2, filtered_df['Slope Change'])

# # Add vertical lines at the intervals
# for i in range(1, len(filtered_df['Slope Change'])):
#     if filtered_df['Slope Change'].iloc[i] != filtered_df['Slope Change'].iloc[i - 1]:
#         plt.axvline(x=i - 0.04, color='gray', linestyle='--', linewidth=0.5)  # Adjust the color, linestyle, and linewidth as needed


# # Add legend
# plt.legend()

# # Show the plot
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# # Define the active width change for which you want to plot mobilized volume against slope change
# target_awc = -20 # Assuming the target active width change is 0

# # Filter the DataFrame to include only rows where the active width change matches the target value
# filtered_df = df[df['Active Width Change'] == target_awc]

# # Get the number of river reaches
# num_river_reaches = 43  # Assuming you have 33 river reaches

# # Create a new figure for the plot
# plt.figure(figsize=(10, 8))

# # Set the width of each bar
# bar_width = 0.9 / num_river_reaches  # Adjusted to accommodate all bars within a width of 0.8

# # Set the x locations for the groups
# x = np.arange(len(filtered_df['Slope Change']))

# # Create a colormap with a sequence of distinct colors
# colors = plt.cm.tab10(np.linspace(0, 1, num_river_reaches))

# # Plot mobilized volume against slope change for each river reach using side-by-side bar plots
# for i in range(num_river_reaches):
#     plt.bar(x + i * bar_width, filtered_df[f'River Reach {i}'], label=f'River Reach {i+1}', width=bar_width, alpha=0.7, color=colors[i])

# # Add labels and title
# plt.xlabel('Slope Change')
# plt.ylabel('Total Mobilized Volume')
# plt.title(f'Side-by-Side Bar Plot: Mobilized Volume vs. Slope Change for Active Width Change {target_awc}')

# # Set x-axis tick labels
# plt.xticks(x + bar_width * (num_river_reaches - 1) / 2, filtered_df['Slope Change'])

# # Add vertical lines at the intervals
# for i in range(1, len(filtered_df['Slope Change'])):
#     if filtered_df['Slope Change'].iloc[i] != filtered_df['Slope Change'].iloc[i - 1]:
#         plt.axvline(x=i - 0.9, color='gray', linestyle='--', linewidth=0.5)  # Adjust the color, linestyle, and linewidth as needed

# # Adjust legend layout to prevent it from blocking the plot
# plt.legend(bbox_to_anchor=(0.96, 1), loc='upper left', fontsize= 8)

# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import ListedColormap

# # Create a new figure and 3D axis
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Define a custom colormap with a different color for each river reach
# num_river_reaches = 20  # Assuming you have 10 river reaches
# colors = plt.cm.viridis(np.linspace(0, 1, num_river_reaches))
# custom_cmap = ListedColormap(colors)

# # Plot each river reach separately
# for i in range(num_river_reaches):
#     ax.scatter(df['Active Width Change'], df['Slope Change'], df[f'River Reach {i}'], label=f'River Reach {i+1}', alpha=0.7, cmap=custom_cmap)

# # Add labels and title
# ax.set_xlabel('Active Width Change')
# ax.set_ylabel('Slope Change')
# ax.set_zlabel('Total Mobilized Volume')
# ax.set_title('3D Scatter Plot: Total Mobilized Volume vs. Active Width Change and Slope Change')

# # Add legend
# ax.legend()

# # Show the plot
# plt.show()



# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Group the DataFrame by active width change
# grouped_df = df.groupby('Active Width Change')

# # Create a new figure and 3D axis
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot each group separately
# for awc, group in grouped_df:
#     for i in range(10):  # Assuming you have 10 river reaches
#         ax.scatter([awc] * len(group), group['Slope Change'], group[f'River Reach {i}'], label=f'AWC {awc}, River Reach {i}', alpha=0.7)

# # Add labels and title
# ax.set_xlabel('Active Width Change')
# ax.set_ylabel('Slope Change')
# ax.set_zlabel('Total Mobilized Volume')
# ax.set_title('3D Scatter Plot: Total Mobilized Volume vs. Active Width Change and Slope Change')

# # Add legend
# ax.legend()

# # Show the plot
# plt.show()


