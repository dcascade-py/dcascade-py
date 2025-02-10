# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:32:54 2025

@author: User4
"""

# import libraries
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  
import os


# Define the directory where the data otput pickled files are stored
directory = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\data_output\\data_output_X_5000_AW_unif_slope_normal_2020\\"
# Initialize an empty dictionary to store the data outputs
output_data_dict = {}
output_data_dict_1 = {}


# Define the number of data files to read
number_of_files_to_read = 5000  # Specify the desired number of files

selective_indices = np.arange(1,number_of_files_to_read+1)


# Iterate over each data output pickled file in the directory
for i in selective_indices:
    filename = f'data_output_ReachData_modified_{i}.pkl'
    file_path = os.path.join(directory, filename)
    
    # Load the pickled file
    with open(file_path, "rb") as file:
        data_output = pickle.load(file)
    
    

    # # Store the chosen output from the data output in the dictionary
    # output_name = 'Mobilized [m^3]'  # choose the output here
    # # 'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    # output_data_dict[int(i)] = data_output[output_name]
    # # the integer part is going to the key of the dictionary
    
    
     
    # Store the chosen output from the data output in the dictionary
    output_name = 'Budget' 
    output_name_1 = 'Mobilized [m^3]' 
    output_name_2 = 'Transported [m^3]'
    # choose the output here
    # 'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    output_data_dict[int(i)] = data_output[output_name_2] - data_output [output_name_1]
    # the integer part is going to the key of the dictionary
    
       
 
# Initialize an empty dictionary to store the summary of chosen data output for each combination
summary_output_data = {}


# Iterate over each combination in chosen output's dictionary
for key, value in output_data_dict.items():   
    # Summary of the chosen outputs for the sample(key) along axis=0
    summary_output_data[key]= np.sum(value, axis=0)
    #intstead of sum, other statistical 



# Create a DataFrame from the summary output
df_Y = pd.DataFrame( summary_output_data).T  # Transpose to get samples as rows and reaches as columns
#the keys (index) will not be in order

# Set appropriate column names (e.g., Reach 1, Reach 2, ...)
df_Y.columns = [f'Reach {i+1}' for i in range(df_Y.shape[1])]


# Sort the DataFrame by its index
df_Y = df_Y.sort_index()



# File location of excel file


# Define the folder where you want to save the figures

# output_folder = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\figures_slope_trans"

excel_file = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\ReachData_modified\\ReachData_modified_X_10000\\ReachData_modified_transposed.xlsx"

# Read the specific sheets into DataFrames
df1 = pd.read_excel(excel_file, sheet_name='Modified_Wac', skiprows=1, nrows=5000, header= None) 
df2 = pd.read_excel(excel_file, sheet_name='Modified_Slope', skiprows=1, nrows=5000, header= None)  

# Drop the first column in each DataFrame, as it contains non-numeric data (samples/model_run_number)
df1 = df1.iloc[:, 1:]
df2 = df2.iloc[:, 1:]

df2.index = df2.index + 1


# for N in range (1,65):

#     N_output = N
    
#     N_slope = N
    
#     new_df = pd.DataFrame({
#         "slope": df2.iloc[:, N_slope-1],
#         "output": df_Y[f"Reach {N_output}"]
#     })
    
#     new_df_1 = new_df.sort_values(by='slope', ascending=True)
    
#     # new_df_1 = new_df.sort_values(by='AW', ascending=True)
    
#     plt.figure()
    
#     plt.figure(figsize=(9, 6))
    
#     plt.scatter(new_df_1.iloc[:,0], new_df_1.iloc[:,1], label = len(new_df_1), cmap='viridis', alpha=0.7)
    
#     # plt.scatter(new_df_1.iloc[:,0], new_df_1.iloc[:,1], label= i, cmap='viridis', alpha=0.7, facecolor = 'none', edgecolors='r')
    
    
#     # plt.plot(new_df_1.iloc[:,0], new_df_1.iloc[:,1], color= 'blue', label= i)  # Plot column data for each sample

#     # Add labels, title, legend, and grid
#     plt.xlabel( f"slope for Reach {N_slope}", fontsize = 12)
#     plt.ylabel(f"{output_name} for Reach {N_output}", fontsize = 12)
#     plt.title(f"Corelation between slope of Reach {N_slope} and {output_name} for Reach {N_output}", fontsize = 10)
#     plt.legend()
#     plt.grid(True)

        
#     # Save the active width indices plot for all the reach in the figures folder
#     figure_filename = f"figure_Reach_{N}.png"
#     save_path = os.path.join(output_folder, figure_filename)
#     plt.savefig(save_path, format='png', dpi=200)  # Save as JPG with 300 dpi
#     plt.close()  # Close the figure to free up memory



# # scatter plot one over another
# N_output = 17

# N_slope = 41

# new_df = pd.DataFrame({
#     "slope": df2.iloc[:, N_slope-1],
#     "output": df_Y[f"Reach {N_output}"]
# })

# new_df_1 = new_df.sort_values(by='slope', ascending=True)

# plt.scatter(new_df_1.iloc[:,0], new_df_1.iloc[:,1], label = len(new_df_1), cmap='viridis', alpha=0.7)

# # plt.scatter(new_df_1.iloc[:,0], new_df_1.iloc[:,1], label= i, cmap='viridis', alpha=0.7, facecolor = 'none', edgecolors='r')


# # plt.plot(new_df_1.iloc[:,0], new_df_1.iloc[:,1], color= 'blue', label= i)  # Plot column data for each sample

# # Add labels, title, legend, and grid
# plt.xlabel( f"slope for Reach {N_slope}", fontsize = 12)
# plt.ylabel(f"{output_name} for Reach {N_output}", fontsize = 12)
# plt.title(f"Corelation between slope of Reach {N_slope} and {output_name} for Reach {N_output}", fontsize = 10)
# plt.legend()
# plt.grid(True)



# Define multiple N_slope values
N_output = 42
N_slope_values = [ 42]  # Example values

for N_slope in N_slope_values:
    new_df = pd.DataFrame({
        "slope": df2.iloc[:, N_slope - 1],
        "output": df_Y[f"Reach {N_output}"]
    })

    new_df_1 = new_df.sort_values(by="slope", ascending=True)

    plt.scatter(
        new_df_1.iloc[:, 0], new_df_1.iloc[:, 1], 
        label=f"Reach {N_slope}", alpha=0.7
    )

# Add labels, title, legend, and grid
plt.xlabel(f"Slope", fontsize=18)
plt.ylabel(f"{output_name} for Reach {N_output}", fontsize=18)
plt.title(f"Correlation between slope and {output_name} for Reach {N_output}", fontsize=18)
plt.legend(title="N_slope values")
plt.grid(True)

plt.show()



