# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:32:50 2024

@author: Sahansila
"""
# import libraries
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  
import os

# Define the directory where the data otput pickled files are stored
directory = "E:\\Sahansila\\SAFE_output\\lhs_unif\\"

# Initialize an empty dictionary to store the data outputs
output_data_dict = {}

# Iterate over each data otputpickled file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        # Split the filename by underscores and remove the extension
        parts = filename.split("_") 
        
       # Extract the integer part from the filename
        part = parts[-1].replace('.pkl', '')  # Remove '.pkl' extension
        index = int(part)  # Convert the part to an integer
        #index refers to the model run with the corresponding reach data
     
        # Load the pickled file
        with open(os.path.join(directory, filename), "rb") as file:
            data_output = pickle.load(file)
        
        # Store the chosen output from the data output in the dictionary
        output_name = 'Mobilized [m^3]'  # choose the output here  
        #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
        output_data_dict[(index)] = data_output[output_name]
        # the integer part is going to the key of the dictionary
       
# Output for a specific sample can be access like this:
# data_output = output_data_dict[(index)]
data_output = output_data_dict [(5)] 

          
# Initialize an empty dictionary to store the summary of chosen data output for each combination
summary_output_data = {}


# Iterate over each combination in chosen output's dictionary
for key, data in output_data_dict.items():   
    # Summary of the chosen outputs for the sample(key) along axis=0
   summary_output_data[key]= np.sum(data, axis=0)
   #intstead of sum, other statistical 
   

# summary for a specific sample for chosen outputs can be access like this:
total_sum = summary_output_data[(5)]


# Create a DataFrame from the summary output
df_mob = pd.DataFrame( summary_output_data).T  # Transpose to get samples as rows and reaches as columns
#the keys (index) will not be in order

# Set appropriate column names (e.g., Reach 1, Reach 2, ...)
df_mob.columns = [f'Reach {i+1}' for i in range(df_mob.shape[1])]

# Sort the DataFrame by its index
df_mob = df_mob.sort_index()


# File location of excel file
file_path = "E:\\Sahansila\\SAFE_output\\ReachData_Xvalues.xlsx"


# Read the Excel file, skip the first row (header), and read the next rows
X_values = pd.read_excel(file_path, sheet_name='Parameter_Values', skiprows=1, nrows=200, header=None)

# Read the header separately
header = pd.read_excel(file_path, sheet_name='Parameter_Values', nrows=0).columns.tolist()

# Assign the header to the DataFrame
X_values.columns = header

# Set the index from 1 to 200
X_values.index = range(1, 201)

# Display the DataFrame
print(X_values)

df = X_values.join(df_mob)


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
        cmap = plt.cm.viridis
        # cmap = "YlOrRd"
        normalize = plt.Normalize(vmin=reach_data.min(), vmax=reach_data.max())  # Normalize values for color mapping

        # Create the scatter plot for this river reach
        ax = axes[i]
        ax.scatter(df['Wac'], df['slope'], s=100, c=reach_data, cmap=cmap, norm=normalize, alpha=0.5)
        # ax.contourf([df['Wac'], df['slope']], reach_data)

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
    
    
    

# contour plot     
    
X = df['Wac']
Y = df['slope']
Z = df['Reach 1']

# Create grid values for X and Y
xi = np.linspace(0.5, 1, 100)  # Generate 200 points between min(X) and max(X)
yi = np.linspace(0.5, 1.5, 100)  # Generate 200 points between min(Y) and max(Y)
xi, yi = np.meshgrid(xi, yi)  # Create a 2D grid from the 1D grid values
    
 
    
from scipy.interpolate import griddata 
# Interpolate Z values onto the grid
zi = griddata((X, Y), Z, (xi, yi), method='linear')    
  
# Create the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(xi, yi, zi, levels=14, cmap='viridis')
plt.colorbar(contour)
plt.scatter(X, Y, c=Z, edgecolors='k')  # Plot original data points
plt.title('Total mobilized volume')
plt.xlabel('Wac')
plt.ylabel('slope')
plt.show()    


# Create the heatmap
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(zi, extent=[min(X), max(X), min(Y), max(Y)], origin='lower', cmap='viridis')
plt.colorbar(heatmap)
plt.scatter(X, Y, c=Z, edgecolors='k')  # Plot original data points
plt.title('Heatmap')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



