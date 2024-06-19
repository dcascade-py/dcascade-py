# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:43:12 2024

@author: Sahansila

"""

import numpy as np
import geopandas as gpd
import scipy.stats as st
import pandas as pd
from safepython.sampling import AAT_sampling # module to perform the input sampling

# Read the shape file of the river network
path_river_network = "E:\\UNIPD\\shp_file_slopes_hydro_and_LR\\"
name_river_network = "Po_river_network.shp"
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefile from shp format
    
# Number of uncertain parameters subject to SA
M = 2


# Parameter ranges 
xmin = [0.5, 0.5] #uniform distribution
xmax = [1, 1.5] #uniform distrbution

# mean = [1, 1] # normal distribution
# sd = [0.1, 0.5] # normal distribution 

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
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]
    # distr_par[i] = [mean[i], sd[i]]


#sampling inputs space
samp_strat = 'lhs' # Latin Hypercube  
#Options: 'rsu': random uniform
         #'lhs': latin hypercube
N = 10 #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)
# X contains the array of data with random nubers generated with which the parameters in reachdata will be multiplied, replaced or added

#Give the location to save the reachdata modified
path_ReachData = "E:\\cascade\\ReachData_for_sensitivity\\ReachData_test\\"

# Run the iterative loop for the range of number of samples and change the parameters using random numbers in X 
#and save the modified reach data for every loop
for i in range(N):
    ReachData_modified = ReachData.copy() 
    ReachData_modified['Wac'] = ReachData_modified['Wac'] * X[i, 0]
    ReachData_modified['Slope'] = ReachData_modified['Slope'] * X[i, 1]
    
    # Save the modified Reachdata GeoDataFrame with a useful name
    output_filename = f"ReachData_modified_{i+1}.shp"
    ReachData_modified.to_file(path_ReachData + output_filename)

    print(f"Saved {output_filename}")
    
# for check if the numbers in the X corresponds to the values in the modified reach data 
# read the specific shape file and check the values  
name_ReachData = "ReachData_modified_5.shp"
ReachData_5 = gpd.GeoDataFrame.from_file(path_ReachData + name_ReachData) #read shapefile from shp format


#Save the X values in excel file
path_reachdata = "E:\\cascade\\ReachData_Xvalues\\"

# Create an Excel writer object
output_excel_path = path_reachdata + "ReachData_Xvalues_test.xlsx"
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    # Save the X array to a separate sheet
    X_df = pd.DataFrame(X, columns=X_Labels)
    # Add an index column starting from 1 to N
    X_df.insert(0, 'Index', range(1, len(X_df) + 1))
    X_df.to_excel(writer, sheet_name='Parameter_Values', index=False)






