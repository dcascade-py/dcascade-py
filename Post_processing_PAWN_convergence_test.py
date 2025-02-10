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

# Import SAFE modules:
from safepython import PAWN
import safepython.plot_functions as pf # module to visualize the results


# # Define the directory where the data otput pickled files are stored
# directory = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\data_output_X_1000"

# # Initialize an empty dictionary to store the data outputs
# output_data_dict = {}

# # Iterate over each data otputpickled file in the directory
# for filename in os.listdir(directory):
#     if filename.endswith(".pkl"):
#         # Split the filename by underscores and remove the extension
#         parts = filename.split("_") 
        
#         # Extract the integer part from the filename
#         part = parts[-1].replace('.pkl', '')  # Remove '.pkl' extension
#         index = int(part)  # Convert the part to an integer
#         #index refers to the model run with the corresponding reach data
     
#         # Load the pickled file
#         with open(os.path.join(directory, filename), "rb") as file:
#             data_output = pickle.load(file)
        
#         # Store the chosen output from the data output in the dictionary
#         output_name = 'Mobilized [m^3]'  # choose the output here  
#         #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
#         output_data_dict[(index)] = data_output[output_name]
#         # the integer part is going to the key of the dictionary
       
# # # Output for a specific sample can be access like this:
# # # data_output = output_data_dict[(index)]
# # data_output = output_data_dict [(5)] 

          
# # Initialize an empty dictionary to store the summary of chosen data output for each combination
# summary_output_data = {}


# # Iterate over each combination in chosen output's dictionary
# for key, data in output_data_dict.items():   
#     # Summary of the chosen outputs for the sample(key) along axis=0
#     summary_output_data[key]= np.sum(data, axis=0)
#     #intstead of sum, other statistical 
   

# # summary for a specific sample for chosen outputs can be access like this:
# # total_sum = summary_output_data[(5)]


# # Create a DataFrame from the summary output
# df_Y = pd.DataFrame( summary_output_data).T  # Transpose to get samples as rows and reaches as columns
# #the keys (index) will not be in order

# # Set appropriate column names (e.g., Reach 1, Reach 2, ...)
# df_Y.columns = [f'Reach {i+1}' for i in range(df_Y.shape[1])]

# # Sort the DataFrame by its index
# df_Y = df_Y.sort_index()


# # Read X from the excel sheets
# # X is the set of input parameters subjected to variability for chosen number of samples

# # File location of excel file
# excel_file = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\ReachData_modified_X_1000\\ReachData_modified_transposed.xlsx"


# # Read the specific sheets into DataFrames
# df1 = pd.read_excel(excel_file, sheet_name='Modified_Wac', skiprows=1, nrows=1000, header= None) 
# df2 = pd.read_excel(excel_file, sheet_name='Modified_Slope', skiprows=1, nrows=1000, header= None)  

# # Drop the first column in each DataFrame, as it contains non-numeric data (samples/model_run_number)
# df1 = df1.iloc[:, 1:]
# df2 = df2.iloc[:, 1:]

# # Number of uncertain parameters subject to SA:
# M = 2
# X_Labels = ['Wac', 'slope']


# # Extract Y from the outlet reach in df_Y
# N = 41
# Y = pd.to_numeric(df_Y.iloc[:, N]).values  # Use corresponding column from df_mob
# #Reach 21 = 22

# n_reach = 64 # Number of reach represented by each column
# n = 10  # Number of conditioing interval intervals for PAWN

# # intialize a dictionary to store the KS (indices) values for each input parameter for every reach 
# KS_max_values_Wac = [] 
# KS_max_values_Slope = []


# for i in range(n_reach):
    
#     # Combine the Wac (df1) and Slope (df2) columns into X
#     X = np.column_stack(((df1.iloc[:, i]).values  , (df2.iloc[:, i]).values ))
    
    
#     # if i == 41:
#     #     print("stop")
   
#     # Compute and plot PAWN sensitivity indices
#     KS_median, KS_mean, KS_max = PAWN.pawn_indices(X, Y, n)
    
#     #store maximum KS values for each input parameter and for all the reaches
#     KS_max_values_Wac.append(KS_max[0])
#     KS_max_values_Slope.append(KS_max[1])
    

# #plotting the consolidated sensitivity indices for all the reaches in one plot

# # Create a Pandas DataFrame with reach column and corresponding  indices (KS) values for AW and slope'
# df = pd.DataFrame({ 
# 'Reach': [f"Reach {i + 1}" for i in range(n_reach)],   # Generate Reach names
# 'Active_width_indices': KS_max_values_Wac,  # Use the list for Wac
# 'Slope_indices': KS_max_values_Slope   # Use the list for slope
#   })   

# #save df into a csv file for plotting purpose
# df.to_csv("E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\Indices\\sample_size_1000.csv", index=False)

N = 26

# Directory containing CSV files
directory = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\KS_values\\Reach26"
# Initialize lists to store data
all_active_width_arrays = []
all_reach_labels = []
legend_labels = []

# Load all CSV files
for file in os.listdir(directory):
    if file.endswith(".csv"):
        # Load the CSV file
        df = pd.read_csv(os.path.join(directory, file))
        
        # Extract data
        active_width_array = df['Active_width_indices'].values
        reach_labels = df['Reach'].tolist()
    
        # Append data for plotting
        all_active_width_arrays.append(active_width_array)
        all_reach_labels.append(reach_labels)
        
        # Use filename as the legend label
        legend_labels.append(file.replace(".csv", ""))

# Define possible markers to use for different datasets
markers = ['o', 's', '*', '^', 'D']  # Circle, Square, Star, Triangle, Diamond

# Plot the data
plt.figure(figsize=(12, 8))
# plot for active width indices
for i, (active_width_array, reach_labels) in enumerate(zip(all_active_width_arrays, all_reach_labels)):
    # Plot each dataset with a different marker
    marker = markers[i % len(markers)]  # Cycle through the markers list if more datasets than markers
    plt.plot(reach_labels, active_width_array, marker=marker, linestyle='None', label=legend_labels[i], markersize = 10)
    for label in reach_labels:
        plt.axvline(x=label, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        

# to plot a vertical line and dynamically place the text in the line
#detemine the y position
y_min, y_max = plt.ylim()  # Get the current y-axis limits
y_position = (y_min + y_max) / 2  # Place the text at the middle of the y-axis

# #plot a vertical line in a reach near Isola serafini dam
# plt.axvline(x= 25, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# # Add text near the vertical line
# plt.text(25, y_position, 'Isola Serafini Dam', color='red', rotation=90, verticalalignment='bottom', fontsize = 14)

#plot a vertical line in an outlet reach
plt.axvline(x= 25, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(25, y_position, 'chosen reach', color='red', rotation=90, verticalalignment='bottom', fontsize = 14)

#plot a vertical line to differentiate tributaries
plt.axvline(x=44, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(44, y_position, 'Tributaries', color='red', rotation=90, verticalalignment='bottom', fontsize =14)

plt.xticks(rotation=90, fontsize=10)  # Adjust font size and rotation
plt.legend(title="AW Inidices for different sample size", loc='best')  # Add legend
plt.title('Sensitivity Indices for Active Width by Reach for different sample size', fontsize=15, pad=15)  # Add padding for title
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.tight_layout()
plt.show()




# Initialize lists to store data
all_slope_arrays = []
all_reach_labels = []
legend_labels = []

# Load all CSV files
for file in os.listdir(directory):
    if file.endswith(".csv"):
        # Load the CSV file
        df = pd.read_csv(os.path.join(directory, file))
        
        # Extract data
        reach_labels = df['Reach'].tolist()
        slope_array = df['Slope_indices'].values
        
        # Append data for plotting
        all_slope_arrays.append(slope_array)
        all_reach_labels.append(reach_labels)
        
        # Use filename as the legend label
        legend_labels.append(file.replace(".csv", ""))

# Define possible markers to use for different datasets
markers = ['o', 's', '*', '^', 'D']  # Circle, Square, Star, Triangle, Diamond

# Plot the data
plt.figure(figsize=(12, 8))

# plot for slope indices
for i, (slope_array, reach_labels) in enumerate(zip(all_slope_arrays, all_reach_labels)):
    # Plot each dataset with a different marker
    marker = markers[i % len(markers)]  # Cycle through the markers list if more datasets than markers
    plt.plot(reach_labels, slope_array, marker=marker, linestyle='None', label=legend_labels[i], markersize = 10)
    for label in reach_labels:
        plt.axvline(x=label, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)


# to plot a vertical line and dynamically place the text in the line
#detemine the y positio
y_min, y_max = plt.ylim()  # Get the current y-axis limits
y_position = (y_min + y_max) / 2  # Place the text at the middle of the y-axis

# #plot a vertical line in a reach near Isola serafini dam
# plt.axvline(x= 25, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# # Add text near the vertical line
# plt.text(25, y_position, 'Isola Serafini Dam', color='red', rotation=90, verticalalignment='bottom', fontsize = 14)

#plot a vertical line in an outlet reach
plt.axvline(x= 25 , color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(25, y_position, 'chosen reach', color='red', rotation=90, verticalalignment='bottom', fontsize = 14)

#plot a vertical line to differentiate tributaries
plt.axvline(x=44, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(44, y_position, 'Tributaries', color='red', rotation=90, verticalalignment='bottom', fontsize =14)

plt.title('Sensitivity Indices for slope by Reach for different sample size', fontsize=15, pad=15)  # Add padding for title
plt.xticks(rotation=90, fontsize=10)  # Adjust font size and rotation
plt.legend(title="Slope Inidices for different sample size", loc='best')  # Add legendplt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.tight_layout()
plt.show()












