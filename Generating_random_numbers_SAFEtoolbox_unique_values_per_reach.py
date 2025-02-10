# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:43:12 2024

@author: Sahansila

"""

import numpy as np
import geopandas as gpd
import scipy.stats as st
import pandas as pd

from safepython.sampling import AAT_sampling# module to perform the input sampling


#-------River shape files 
path_river_network = "E:\\Sahansila\\shp_file_slopes_hydro_and_LR\\"
name_river_network = 'Po_river_network.shp'
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefile from shp format
# Sort ReachData according to the FromN
ReachData = ReachData.sort_values(by = 'FromN', ignore_index = True)

    
# Number of uncertain parameters subject to SA
M = 2


# Define distributions for each variable
distr_fun = [st.uniform, st.norm]  # First variable is uniform, second is normal
distr_par = [[0.5, 0.5], [1, 0.25]]  # Uniform(loc=0.5, scale=1), Normal(mean=0, std=1)


# # Parameter ranges 
# xmin = [0.5, 0.5] #uniform distribution
# xmax = [1, 1.5] #uniform distrbution

# # Distribution parameters for normal distribution
# distr_fun = st.norm  # Normal distribution

# distr_par = [np.nan] * M
# for i in range(M):
#     distr_par[i] = [mean[i], sd[i]]
    
    
# Sampling strategy and number of samples
samp_strat = 'lhs'  # Latin Hypercube Sampling
N = 10000 # Number of samples

# Sampling inputs
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)

# Display first few samples
print("Generated Samples:\n", X [:5])

# Parameter labels
X_Labels = ['Wac (reduced by 50%)', 'Slope (±50%)']

# Plot histograms for each parameter
import matplotlib.pyplot as plt

for i in range(M):
    plt.figure()
    plt.hist(X[:, i], bins=30, density=True, alpha=0.6, color='blue')

    # Add labels and title
    plt.title(f'Distribution of {X_Labels[i]}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()


#Give the location to save the reachdata modified
path_ReachData = "E:\\Sahansila\\SAFE_output\\test\\"


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
    
    Wac_values_df[f'Model_Run_{i+1}'] = ReachData_modified['Wac'].values
    # Save the modified values in their respective DataFrames
    Slope_values_df[f'Model_Run_{i+1}'] = ReachData_modified['Slope'].values
    
    # Save the modified ReachData as shapefiles or CSV
    output_filename = f"ReachData_modified_{i+1}.shp"
    ReachData_modified.to_file(path_ReachData + output_filename)


with pd.ExcelWriter(path_ReachData + 'ReachData_modified_transposed.xlsx', engine='openpyxl') as writer:
    if not writer.book.worksheets:  # Ensure there is at least one sheet
        writer.book.create_sheet("Sheet1")
    random_values_Wac.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Random_Wac_Multipliers', index=True)
    random_values_Slope.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Random_Slope_Multipliers', index=True)
    Wac_values_df.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Modified_Wac', index=True)
    Slope_values_df.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Modified_Slope', index=True)
    
plt.figure()
plt.hist(Wac_random_values, bins=10, density=True, alpha=0.6, color='blue')




    
    