# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:43:12 2024

@author: Sahansila

"""

import numpy as np
import geopandas as gpd
import scipy.stats as st
import pandas as pd

from safepython.sampling import AAT_sampling, AAT_sampling_extend # module to perform the input sampling
from safepython.util import aggregate_boot  # function to aggregate the bootstrap results




#-------River shape files 
path_river_network = "C:\\Sahansila\\cascade\\input\\shp_file_slopes_hydro_and_LR\\01-shp_corrected_names\\"
name_river_network = 'Po_river_network.shp'
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefile from shp formatfrom safepython.sampling import AAT_sampling # module to perform the input sampling

    
# Number of uncertain parameters subject to SA
M = 2


# # Parameter ranges 
# xmin = [0.5, 0.5] #uniform distribution
# xmax = [1, 1.5] #uniform distrbution

mean = [0, 0] # normal distribution
sd = [0.25, 0.25] # normal distribution 

# Name of parameters (will be used to customize plots):
X_Labels = ['Wac', 'slope']


# Parameter distributions:
distr_fun = st.uniform # uniform distribution
# The shape parameters of the uniform distribution are the lower limit and the
#difference between lower and upper limits:

# distr_fun = st.norm # normal distribution    
# The shape parameters of the normal distribution are the mean and standard deviation
   
distr_par = [np.nan] * M
for i in range(M):
    # distr_par[i] = [xmin[i], xmax[i] - xmin[i]]
    distr_par[i] = [mean[i], sd[i]]


#sampling inputs space
samp_strat = 'lhs' # Latin Hypercube  
#Options: 'rsu': random uniform
         #'lhs': latin hypercube
N = 200 #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)
# X contains the array of data with random nubers generated with which the parameters in reachdata will be multiplied, replaced or added

#Give the location to save the reachdata modified
path_ReachData = "C:\\sahansila\\cascade\\SAFE_output\\test\\"

# # Run the iterative loop for the range of number of samples and change the parameters using random numbers in X 
# #and save the modified reach data for every loop
# for i in range(N):
#     ReachData_modified = ReachData.copy() 
#     ReachData_modified['Wac'] = ReachData_modified['Wac'] * X[i, 0]
#     ReachData_modified['Slope'] = ReachData_modified['Slope'] * X[i, 1]
    
#     # Save the modified Reachdata GeoDataFrame with a useful name
#     output_filename = f"ReachData_modified_{i+1}.shp"
#     ReachData_modified.to_file(path_ReachData + output_filename)

#     print(f"Saved {output_filename}")
    
# # for check if the numbers in the X corresponds to the values in the modified reach data 
# # read the specific shape file and check the values  
# name_ReachData = "ReachData_modified_5.shp"
# ReachData_5 = gpd.GeoDataFrame.from_file(path_ReachData + name_ReachData) #read shapefile from shp format


# #Save the X values in excel file
# path_reachdata = "E:\\cascade\\ReachData_Xvalues\\"

# # Create an Excel writer object
# output_excel_path = path_reachdata + "ReachData_Xvalues_test.xlsx"
# with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
#     # Save the X array to a separate sheet
#     X_df = pd.DataFrame(X, columns=X_Labels)
#     # Add an index column starting from 1 to N
#     X_df.insert(0, 'Index', range(1, len(X_df) + 1))
#     X_df.to_excel(writer, sheet_name='Parameter_Values', index=False)

# DataFrames to store the random values for each model run
random_values_Wac = pd.DataFrame()  # For storing multipliers for 'Wac'
random_values_Slope = pd.DataFrame()  # For storing multipliers for 'Slope'

# DataFrames to store the modified values for each model run
Wac_values_df = pd.DataFrame()  # For Wac column
Slope_values_df = pd.DataFrame()  # For Slope column

# Run the iterative loop for N model simulations
for i in range(N):
    ReachData_modified = ReachData.copy()
    
    # Generate unique random multipliers for each row in 'Wac' and 'Slope'
    Wac_random_values = np.random.choice(X[:, 0], len(ReachData_modified), replace=False)
    Slope_random_values = np.random.choice(X[:, 1], len(ReachData_modified), replace=False)
    
    # Apply the unique multipliers to 'Wac' and 'Slope'
    ReachData_modified['Wac'] = ReachData_modified['Wac'] * Wac_random_values
    ReachData_modified['Slope'] = ReachData_modified['Slope'] * Slope_random_values
    
    # Save the random multipliers in their respective DataFrames
    random_values_Wac[f'Model_Run_{i+1}'] = Wac_random_values
    random_values_Slope[f'Model_Run_{i+1}'] = Slope_random_values
    
    # Save the modified values in their respective DataFrames
    Wac_values_df[f'Model_Run_{i+1}'] = ReachData_modified['Wac'].values
    Slope_values_df[f'Model_Run_{i+1}'] = ReachData_modified['Slope'].values
    
    # Save the modified ReachData as shapefiles or CSV
    output_filename = f"ReachData_modified_{i+1}.shp"
    ReachData_modified.to_file(path_ReachData + output_filename)

# Save the random multipliers and modified data in Excel sheets
with pd.ExcelWriter(path_ReachData + 'ReachData_modified.xlsx') as writer:
    random_values_Wac.to_excel(writer, sheet_name='Random_Wac_Multipliers', index=False)
    random_values_Slope.to_excel(writer, sheet_name='Random_Slope_Multipliers', index=False)
    Wac_values_df.to_excel(writer, sheet_name='Modified_Wac', index=False)
    Slope_values_df.to_excel(writer, sheet_name='Modified_Slope', index=False)


