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
=======

#-------River shape files 
>>>>>>> Stashed changes
name_river_network = 'Po_river_network.shp'
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefile from shp format
# Sort ReachData according to the FromN
ReachData = ReachData.sort_values(by = 'FromN', ignore_index = True)

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

<<<<<<< Updated upstream
# # Distribution parameters for normal distribution
# distr_fun = st.norm  # Normal distribution

# distr_par = [np.nan] * M
# for i in range(M):
#     distr_par[i] = [mean[i], sd[i]]
    
    
# Sampling strategy and number of samples
samp_strat = 'lhs'  # Latin Hypercube Sampling
N = 10000 # Number of samples

=======
# Parameter means and standard deviations
mean = [1, 1]  # Normal distribution means
sd = [0.25, 0.25]  # Normal distribution standard deviations

# Adjusted ranges for each parameter
ranges = [
    [0.5 * mean[0], mean[0]],      # First parameter: reduced by 50%
    [0.5 * mean[1], 1.5 * mean[1]]  # Second parameter: ±50%
]

# Distribution parameters for normal distribution
distr_fun = st.norm  # Normal distribution

distr_par = [np.nan] * M
for i in range(M):
    distr_par[i] = [mean[i], sd[i]]
    
    
# Sampling strategy and number of samples
samp_strat = 'lhs'  # Latin Hypercube Sampling
N = 2000  # Number of samples

>>>>>>> Stashed changes
# Sampling inputs
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)

# Display first few samples
<<<<<<< Updated upstream
print("Generated Samples:\n", X [:5])
=======
print("Generated Samples:\n", X[:5])


# Adjust values to fall within the range by re-sampling for out-of-bounds values
for i in range(M):
    lower, upper = ranges[i]
    # Check for values outside the range and replace them with random normal samples within the range
    out_of_bounds_indices = (X[:, i] < lower) | (X[:, i] > upper)

   # For first parameter, re-sample out-of-bounds values
    while np.any(out_of_bounds_indices):
            # Re-sample the out-of-bounds values from a normal distribution
            X[out_of_bounds_indices, i] = np.random.normal(loc=mean[i], scale=sd[i], size=np.sum(out_of_bounds_indices))
            # Ensure that resampled values are within bounds
            out_of_bounds_indices = (X[:, i] < lower) | (X[:, i] > upper)

    # else:  # For the second parameter, just clip the values
    #     X[:, i] = np.clip(X[:, i], lower, upper)
        
# Display first few samples
print("Generated Samples:\n", X[:5])
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
path_ReachData = "E:\\Sahansila\\SAFE_output\\test\\"
=======
path_ReachData = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\ReachData_modified_X_2000\\"
>>>>>>> Stashed changes


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

<<<<<<< Updated upstream

with pd.ExcelWriter(path_ReachData + 'ReachData_modified_transposed.xlsx', engine='openpyxl') as writer:
    if not writer.book.worksheets:  # Ensure there is at least one sheet
        writer.book.create_sheet("Sheet1")
    random_values_Wac.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Random_Wac_Multipliers', index=True)
    random_values_Slope.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Random_Slope_Multipliers', index=True)
    Wac_values_df.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Modified_Wac', index=True)
    Slope_values_df.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Modified_Slope', index=True)
    
plt.figure()
plt.hist(Wac_random_values, bins=10, density=True, alpha=0.6, color='blue')


=======
# Save the random multipliers and modified data in Excel sheets
# with pd.ExcelWriter(path_ReachData + 'ReachData_modified.xlsx') as writer:
#     random_values_Wac.to_excel(writer, sheet_name='Random_Wac_Multipliers', index=False)
#     random_values_Slope.to_excel(writer, sheet_name='Random_Slope_Multipliers', index=False)
#     Wac_values_df.to_excel(writer, sheet_name='Modified_Wac', index=False)
#     Slope_values_df.to_excel(writer, sheet_name='Modified_Slope', index=False)


with pd.ExcelWriter(path_ReachData + 'ReachData_modified_transposed.xlsx', engine='openpyxl') as writer:
    if not writer.book.worksheets:  # Ensure there is at least one sheet
        writer.book.create_sheet("Sheet1")
    random_values_Wac.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Random_Wac_Multipliers', index=True)
    random_values_Slope.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Random_Slope_Multipliers', index=True)
    Wac_values_df.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Modified_Wac', index=True)
    Slope_values_df.T.rename(columns=lambda x: f'Reach {x+1}').to_excel(writer, sheet_name='Modified_Slope', index=True)
    
plt.figure()
plt.hist(Slope_random_values, bins=30, density=True, alpha=0.6, color='blue')


>>>>>>> Stashed changes


    
    