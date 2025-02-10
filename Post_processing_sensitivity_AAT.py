# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:32:50 2024

@author: Sahansila
"""

#import libraries
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

# Define the directory where the pickled files are stored
directory = "E:\\Sahansila\\combined_random_sampling\\"

# Initialize an empty dictionary to store the data outputs
output_data_dict = {}

# Initialize an empty dictionary to store the chosen output
chosen_output_dict = {}

# Iterate over each pickled file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        # Split the filename by underscores and remove the extension
        parts = filename.split("_")
        # Extract active_width_change and slope_change from the filename
        awc_percent = int(parts[3])
        sc_percent = int(parts[5].split(".")[0])  # this is going to be the key of the dictionary
        
        # Load the pickled file
        with open(os.path.join(directory, filename), "rb") as file:
            data_output = pickle.load(file)
        
        # Store the data_output in the dictionary
        output_data_dict[(awc_percent, sc_percent)] = data_output
        

        # Store the chosen output from the data output in the dictionary
        output_name = 'Mobilized volume [m^3]'  # choose the output here  
        #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
        chosen_output_dict[(awc_percent, sc_percent)] = data_output[output_name]

# A dictionary containing the data outputs for each combination
# Data_output for a specific combination can be access like this:
# data_output = output_data_dict[(awc_percent, sc_percent)]

  

# Initialize an empty dictionary to store the summary of chosen data output for each combination
summary_output_data = {}


# Iterate over each combination in chosen output's dictionary
for key, data in  chosen_output_dict.items():   
    # Summary of the chosen outputs for the sample(key) along axis=0
   summary_output_data[key]= np.sum(data, axis=0)
   #intstead of sum, other statistical measures can be applied
   
# summary for a specific sample for chosen outputs can be access like this:
total_sum = summary_output_data[(-20,-20)]


# Create a DataFrame from the summary output
df = pd.DataFrame( summary_output_data).T  # Transpose to get samples as rows and reaches as columns
#the keys (index) will not be in order

# Set appropriate column names (e.g., Reach 1, Reach 2, ...)
df.columns = [f'River Reach {i+1}' for i in range(df.shape[1])]

# Add column names for active width and slope
df.index.names = ['Active Width Change', 'Slope Change']

# Reset index to have columns for active width and slope
df.reset_index(inplace=True) # change keys from dictionary to columns for plotting


# bubble plot showing the intensity of the total mobilized volume

# Determine the number of river reaches
num_reaches = 64

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

      
#Plot: for every slope change, one active width chnage line

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


# plot of summary output for each combination in a separate layout
# Iterate over each combination in total_mobilized_volume_dict
for key, data in summary_output_data.items():
    # Get the active width and slope from the combination tuple
    awc_percent, sc_percent = key
    
    # Create a new figure for each combination
    plt.figure(figsize=(10, 6))
    
    # Plot the total mobilized volume for each reach
    plt.plot(data, marker='o', linestyle='-')
    
    # Add labels and title
    plt.xlabel('Reach')
    plt.ylabel('Total Mobilized Volume')
    plt.title(f'Total Mobilized Volume for Active Width Change {awc_percent}% and Slope Change {sc_percent}%')
    
    # Show grid
    plt.grid(True)
     
    # Show the plot
    plt.show()
 

