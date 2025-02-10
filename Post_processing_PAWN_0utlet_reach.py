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


# Define the directory where the data otput pickled files are stored
directory = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\data_output\\data_output_X_5000_AW_unif_slope_normal_2020\\"
# Initialize an empty dictionary to store the data outputs
output_data_dict = {}

# Define the number of data files to read
number_of_files_to_read = 5000 # Specify the desired number of files

selective_indices = np.arange(1,number_of_files_to_read+1)


# Iterate over each data output pickled file in the directory
for i in selective_indices:
    filename = f'data_output_ReachData_modified_{i}.pkl'
    file_path = os.path.join(directory, filename)
    
    # Load the pickled file
    with open(file_path, "rb") as file:
        data_output = pickle.load(file)
    
    

    # Store the chosen output from the data output in the dictionary
    output_name = 'D50 mobilised layer [m]'  # choose the output here
    # 'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    output_data_dict[int(i)] = data_output[output_name]
    # the integer part is going to the key of the dictionary
    
    
# #-----------------calculate eroded/deposited and add it to data_output_t
# data_output['Delta volume [m^3]'] = data_output['Transported [m^3]'] - data_output['Mobilized volume [m^3]']
# data_output['Delta z [m]'] = data_output['Delta volume [m^3]']/(ReachData['Wac'].values*ReachData['Length'].values)

    
# Output for a specific sample can be access like this:
# data_output_check = output_data_dict[(index)]
data_output_check = output_data_dict[25] 

          
# Initialize an empty dictionary to store the summary of chosen data output for each combination
summary_output_data = {}


# Iterate over each combination in chosen output's dictionary
for key, value in output_data_dict.items():   
    # Summary of the chosen outputs for the sample(key) along axis=0
   summary_output_data[key]= np.sum(value, axis=0)
   #intstead of sum, other statistical 
   

# summary for a specific sample for chosen outputs can be access like this:
# total_sum = summary_output_data[(5)]


# Create a DataFrame from the summary output
df_Y = pd.DataFrame( summary_output_data).T  # Transpose to get samples as rows and reaches as columns
#the keys (index) will not be in order

# Set appropriate column names (e.g., Reach 1, Reach 2, ...)
df_Y.columns = [f'Reach {i+1}' for i in range(df_Y.shape[1])]

# Sort the DataFrame by its index
df_Y = df_Y.sort_index()


# Read X from the excel sheets
# X is the set of input parameters subjected to variability for chosen number of samples

# File location of excel file
excel_file = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\ReachData_modified\\ReachData_modified_X_10000\\ReachData_modified_transposed.xlsx"

# Read the specific sheets into DataFrames
df1 = pd.read_excel(excel_file, sheet_name='Modified_Wac', skiprows=1, nrows=5000, header= None) 
df2 = pd.read_excel(excel_file, sheet_name='Modified_Slope', skiprows=1, nrows=5000, header= None)  

# Drop the first column in each DataFrame, as it contains non-numeric data (samples/model_run_number)
df1 = df1.iloc[:, 1:]
df2 = df2.iloc[:, 1:]

# Number of uncertain parameters subject to SA:
M = 2
X_Labels = ['Wac', 'slope']


# Extract Y from the outlet reach in df_Y
N = 41
Y = pd.to_numeric(df_Y.iloc[:, N]).values  # Use corresponding column from df_mob
#Reach 21 = 22


# Define the folder where you want to save the figures
output_folder_cdf = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\figures\\new\\figures_sample_size_5000_trans\\CDF_reach36"
output_folder_indices = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\figures\\new\\figures_sample_size_5000_trans\\Indices_reach36"
output_folder = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\figures"
# Ensure the directory exists, and if it doesn't, create it
if not os.path.exists(output_folder_cdf):
    os.makedirs(output_folder_cdf)
    
if not os.path.exists(output_folder_indices):
    os.makedirs(output_folder_indices)   
  
    

n_reach = 64 # Number of reach represented by each column
n = 10  # Number of conditioing interval intervals for PAWN

# intialize a dictionary to store the KS (indices) values for each input parameter for every reach 
KS_max_values_Wac = [] 
KS_max_values_Slope = []


for i in range(n_reach):
    
    # Combine the Wac (df1) and Slope (df2) columns into X
    X = np.column_stack(((df1.iloc[:, i]).values  , (df2.iloc[:, i]).values ))
    
    
    # if i == 41:
    #     print("stop")
  
    # Plot the CDF using the current subplot
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n, cbar=True, n_col=3, labelinput=['Wac', 'Slope'])
    
    # Set title for each subplot
    plt.title(f'Reach {i+1}')  # Title indicating which column (Reach) is plotted
    
    # Save the CDF plot in the CDF folder
    figure_filename_cdf = f"cdf_figure_{i+1}.png"
    save_path_cdf = os.path.join(output_folder_cdf, figure_filename_cdf)
    plt.savefig(save_path_cdf, format='png', dpi=200)  # Save as JPG with 300 dpi
    plt.close()  # Close the figure to free up memory

    # print(f"CDF figure saved at: {save_path_cdf}")
    
    # Compute and plot PAWN sensitivity indices
    KS_median, KS_mean, KS_max = PAWN.pawn_indices(X, Y, n)
    
    #store maximum KS values for each input parameter and for all the reaches
    KS_max_values_Wac.append(KS_max[0])
    KS_max_values_Slope.append(KS_max[1])
    

    # Set title for each subplot
    plt.title(f'Reach {i+1}')  # Title indicating which column is plotted
    
    # Plot results for KS_max 
    plt.figure()
    pf.boxplot1(KS_max, X_Labels=['Wac', 'Slope'], Y_Label='Ks (max)')
    
    # Save the PAWN indices plot in the figures
    figure_filename_indices = f"indices_figure_{i+1}.png"
    save_path_indices = os.path.join(output_folder_indices, figure_filename_indices)
    plt.savefig(save_path_indices, format='png', dpi=200)  # Save as JPG with 300 dpi
    plt.close()  # Close the figure to free up memory

    print(f"PAWN indices figure saved at: {save_path_indices}")
    


#plotting the consolidated sensitivity indices for all the reaches in one plot

# Create a Pandas DataFrame with reach column and corresponding  indices (KS) values for AW and slope'
df = pd.DataFrame({ 
'Reach': [f"Reach {i + 1}" for i in range(n_reach)],   # Generate Reach names
'Active_width_indices': KS_max_values_Wac,  # Use the list for Wac
'Slope_indices': KS_max_values_Slope   # Use the list for slope
 })   

# save the ks values in a csv format
df.to_csv("E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\KS_values_trans\Reach36\\sample_size_5000.csv", index=False)


# Convert Active_width and Slope from dataframe to 1D numpy arrays
active_width_array = df['Active_width_indices'].values  # Extract Active_width as a numpy array
slope_array = df['Slope_indices'].values  # Extract Slope as a numpy array
reach_labels = df['Reach'].tolist()  # Extract reach labels as a list


# Plot using boxplot1 for KS values for Active_width
plt.figure()
plt.figure(figsize=(9, 6))  # Increase figure size for better readability
plt.xticks(rotation=90, fontsize= 5)  # Reduce font size and adjust rotation
pf.boxplot1(active_width_array, X_Labels=reach_labels, Y_Label='Active Width indices')

# to plot a vertical line and dynamically place the text in the line
#detemine the y position
y_min, y_max = plt.ylim()  # Get the current y-axis limits
y_position = (y_min + y_max) / 2  # Place the text at the middle of the y-axis

#plot a vertical line in a reach near Isola serafini dam
plt.axvline(x= 26, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(26, y_position, 'Isola Serafini Dam', color='red', rotation=90, verticalalignment='bottom', fontsize = 14)



#plot a vertical line in an outlet reach
plt.axvline(x= N +1, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(N+1, y_position, 'chosen reach', color='red', rotation=90, verticalalignment='bottom', fontsize = 14)

#plot a vertical line to differentiate tributaries
plt.axvline(x=44, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(44, y_position, 'Tributaries', color='red', rotation=90, verticalalignment='bottom', fontsize =14)

plt.title('Boxplot of Active Width by Reach', fontsize=15, pad=15)  # Add padding for title
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.show()
 
# # Save the active width indices plot for all the reach in the figures folder
# figure_filename = f"figure_AW_{N+1}.png"
# save_path = os.path.join(output_folder, figure_filename)
# plt.savefig(save_path, format='png', dpi=200)  # Save as JPG with 300 dpi
# plt.close()  # Close the figure to free up memory


# Plot using boxplot1 for ks values for Slope
plt.figure()
plt.figure(figsize=(9, 6))  # Increase figure size for better readability
plt.xticks(rotation=90, fontsize= 5)  # Reduce font size and adjust rotation
pf.boxplot1(slope_array, X_Labels=reach_labels, Y_Label='Slope_indices')
# to plot a vertical line and dynamically place the text in the line
#detemine the y position
y_min, y_max = plt.ylim()  # Get the current y-axis limits
y_position = (y_min + y_max) / 2  # Place the text at the middle of the y-axis

#plot a vertical line in a reach near Isola serafini dam
plt.axvline(x= 26, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(26, y_position, 'Isola Serafini Dam', color='red', rotation=90, verticalalignment='bottom', fontsize = 14)


#plot a vertical line in an outlet reach
plt.axvline(x= N+1, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(N+1, y_position, 'chosen reach', color='red', rotation=90, verticalalignment='bottom', fontsize = 16)

#plot a vertical line to differentiate tributaries
plt.axvline(x=44, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(44, y_position, 'Tributaries', color='red', rotation=90, verticalalignment='bottom', fontsize =16)

plt.title('Boxplot of Slope by Reach', fontsize=15, pad=15) 
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.show()

# # Save the active width indices plot for all the reach in the figures folder
# figure_filename = f"figure_Slope_{N+1}.png"
# save_path = os.path.join(output_folder, figure_filename)
# plt.savefig(save_path, format='png', dpi=200)  # Save as JPG with 300 dpi
# plt.close()  # Close the figure to free up memory


# add the indices values to the Reachdata shapefile
import geopandas as gpd

#Loading River shape files 
path_river_network = "E:\\Sahansila\\input\\shp_file_slopes_hydro_and_LR\\02-shp_trib_GSD_updated\\"
name_river_network = 'Po_river_network.shp'
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefile from shp formatfrom safepython.sampling import AAT_sampling # module to perform the input sampling


# Extract the numeric part from 'Reach' (e.g., extract '1' from 'Reach 1')
df['Reach'] = df['Reach'].astype(str)  # Convert Reach column to string
df['Reach'] = df['Reach'].str.extract(r'(\d+)').astype(int)  # Extract numeric partdf

# Ensure the corresponding column 'FromN' in the shapefile is an integer
ReachData['FromN'] = ReachData['FromN'].astype(int)


# Loop through the GeoDataFrame (gdf) and assign values from df 
for idx, row in ReachData.iterrows():
    # Find the matching row in df where 'Reach ID' matches 'FromN' to ensure the indices values
    #corresponds to right reach
    matching_row = df[df['Reach'] == row['FromN']]
    
    # Ensure matching_row is not empty
    if not matching_row.empty:
          # Assign values to ReachData columns, ensuring column names are correct
          ReachData.at[idx, 'Active_width_indices'] = matching_row['Active_width_indices'].values[0]
          ReachData.at[idx, 'Slope_indices'] = matching_row['Slope_indices'].values[0]
    else:
          print(f"No matching row found in df for Reach ID {row['FromN']}")

# Specify the new directory path
new_directory = "E:\\Sahansila\\input\\new\\updated_ReachData_indices_reach36_trans_sample_size_5000_X\\"

# Ensure the directory exists, and if it doesn't, create it
if not os.path.exists(new_directory):
    os.makedirs(new_directory)
    
# Specify the filename
filename = 'updated_shapefile_reach_36.shp'  # Name of the shapefile

# Concatenate the path and filename
output_file = os.path.join(new_directory, filename)
# Save the updated GeoDataFrame to the new shapefile
ReachData.to_file(output_file, driver='ESRI Shapefile')


















