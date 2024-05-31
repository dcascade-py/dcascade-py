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


path_river_network = "E:\\UNIPD\\shp_file_slopes_hydro_and_LR\\"
name_river_network = "Po_rivernet_grainsze_new_d.shp"


# read the network 
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefine from shp format
    
# Number of uncertain parameters subject to SA:
M = 2


# Parameter ranges (from Kollat et al.(2012)):
# xmin = [1, 1]
# xmax = [0.2, 0.2]

mean = [0, 0]
sd = [0.2, 0.1]

# Name of parameters (will be used to customize plots):
X_Labels = ['Wac', 'slope']


# Parameter distributions:
# distr_fun = st.uniform # uniform distribution
distr_fun = st.norm # uniform distribution
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M):
    # distr_par[i] = [xmin[i], xmax[i] - xmin[i]]
    # distr_par[i] = [xmin[i], xmax[i]]
    distr_par[i] = [mean[i], sd[i]]


#%% Step 3 (sample inputs space)
samp_strat = 'lhs' # Latin Hypercube
N = 10 #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)

path_ReachData = "E:\\cascade\\ReachData_for_sensitivity\\ReachData_rsu_unif_dist\\"

# Perform operations and save results
for i in range(N):
    ReachData_modified = ReachData.copy() 
    ReachData_modified['Wac'] = ReachData_modified['Wac'] * X[i, 0]
    ReachData_modified['Slope'] = ReachData_modified['Slope'] * X[i, 1]
    
    # Save the modified GeoDataFrame with a useful name
    output_filename = f"ReachData_modified_{i+1}.shp"
    ReachData_modified.to_file(path_ReachData + output_filename)

    print(f"Saved {output_filename}")
    
    
name_ReachData = "ReachData_modified_2.shp"


# read the network 
ReachData_2 = gpd.GeoDataFrame.from_file(path_ReachData + name_ReachData) #read shapefine from shp format

path_reachdata = "E:\\cascade\\ReachData_Xvalues\\"

# Create an Excel writer object
output_excel_path = path_reachdata + "ReachData_Xvalues_rsu_unif.xlsx"
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    # Save the X array to a separate sheet
    X_df = pd.DataFrame(X, columns=X_Labels)
    X_df.to_excel(writer, sheet_name='Parameter_Values', index=False)






