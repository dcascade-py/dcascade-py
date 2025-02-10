# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:32:50 2024

@author: Sahansila
"""
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.pyplot import text
import copy
import os

# Define the directory where the pickled files are stored
directory = "E:\\cascade\\ReachData_for_sensitivity\\XA\\XA_data_output\\"

# Initialize an empty dictionary to store the data outputs
output_data_dict = {}

# Iterate over each pickled file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        # Split the filename by underscores and remove the extension
        parts = filename.split("_")
        
       # Extract the integer part from the filename
        reach_data_part = parts[-1].replace('.pkl', '')  # Remove '.pkl' extension
        ReachData = int(reach_data_part)  # Convert the part to an integer
     
        # Load the pickled file
        with open(os.path.join(directory, filename), "rb") as file:
            data_output = pickle.load(file)
        
        # Store the data_output in the dictionary
        output_data_dict[(ReachData)] = data_output

# A dictionary containing the data outputs for each ReachData
# Data_output for a specific ReachData can be access like this:
# data_output = output_data_dict[(ReachDatnumber)]
data_output = output_data_dict [(5)]


         
# Initialize an empty dictionary to store the mobilized volumes/  for each combination
mobilized_volumes_dict = {}

# Iterate over each combination in output_data_dict
for combination, data_output in output_data_dict.items():
    # Extract the desired output from the data_output
    output_name = 'Mobilized volume [m^3]'    #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    mobilized_volume = data_output[output_name]
 

    # Store the output in the respective dictionary according to thier name
    mobilized_volumes_dict[combination] = mobilized_volume
 
    


# Initialize an empty dictionary to store the total sum/ median/ averages  of chosen data output for each combination
total_mobilized_volume_dict = {}


# Iterate over each combination in chosen output's dictionary
for combination, mobilized_volume_array in mobilized_volumes_dict.items():   
    # Sum or median of the chosen outputs for the current combination along axis=0
    total_sum = np.sum(mobilized_volume_array, axis=0)
   # median_value = np.median(D50_active_layer_array, axis=0)
    
    
    # Store the total sum/median in the total/median dictionary
    total_mobilized_volume_dict[combination] = total_sum
   # median_D50_active_layer_dict[combination] = median_value

# total/median dict contains the total sum / median of output for each combination
# total sum/ median for a specific combination can be access like this:
total_sum = total_mobilized_volume_dict[(5)]



# Create a DataFrame from the total_mobilized_volume_dict
df_mob = pd.DataFrame(total_mobilized_volume_dict).T  # Transpose to get combinations as rows and reaches as columns

# Set appropriate column names (e.g., Reach 1, Reach 2, ...)
df_mob.columns = [f'Reach {i+1}' for i in range(df_mob.shape[1])]

# Sort the DataFrame by its index
df_mob = df_mob.sort_index()

YA = df_mob.iloc [:, 4]


# Define the directory where the pickled files are stored
directory = "E:\\cascade\\ReachData_for_sensitivity\\XB\\XB_data_output\\"

# Initialize an empty dictionary to store the data outputs
output_data_dict = {}

# Iterate over each pickled file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        # Split the filename by underscores and remove the extension
        parts = filename.split("_")
        
       # Extract the integer part from the filename
        reach_data_part = parts[-1].replace('.pkl', '')  # Remove '.pkl' extension
        ReachData = int(reach_data_part)  # Convert the part to an integer
     
        # Load the pickled file
        with open(os.path.join(directory, filename), "rb") as file:
            data_output = pickle.load(file)
        
        # Store the data_output in the dictionary
        output_data_dict[(ReachData)] = data_output

# A dictionary containing the data outputs for each ReachData
# Data_output for a specific ReachData can be access like this:
# data_output = output_data_dict[(ReachDatnumber)]
data_output = output_data_dict [(5)]


         
# Initialize an empty dictionary to store the mobilized volumes/  for each combination
mobilized_volumes_dict = {}

# Iterate over each combination in output_data_dict
for combination, data_output in output_data_dict.items():
    # Extract the desired output from the data_output
    output_name = 'Mobilized volume [m^3]'    #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    mobilized_volume = data_output[output_name]
 

    # Store the output in the respective dictionary according to thier name
    mobilized_volumes_dict[combination] = mobilized_volume
 
    


# Initialize an empty dictionary to store the total sum/ median/ averages  of chosen data output for each combination
total_mobilized_volume_dict = {}


# Iterate over each combination in chosen output's dictionary
for combination, mobilized_volume_array in mobilized_volumes_dict.items():   
    # Sum or median of the chosen outputs for the current combination along axis=0
    total_sum = np.sum(mobilized_volume_array, axis=0)
   # median_value = np.median(D50_active_layer_array, axis=0)
    
    
    # Store the total sum/median in the total/median dictionary
    total_mobilized_volume_dict[combination] = total_sum
   # median_D50_active_layer_dict[combination] = median_value

# total/median dict contains the total sum / median of output for each combination
# total sum/ median for a specific combination can be access like this:
total_sum = total_mobilized_volume_dict[(5)]


import pandas as pd
import numpy as np


# Create a DataFrame from the total_mobilized_volume_dict
df_mob = pd.DataFrame(total_mobilized_volume_dict).T  # Transpose to get combinations as rows and reaches as columns

# Set appropriate column names (e.g., Reach 1, Reach 2, ...)
df_mob.columns = [f'Reach {i+1}' for i in range(df_mob.shape[1])]

# Sort the DataFrame by its index
df_mob = df_mob.sort_index()

YB = df_mob.iloc [:, 4]


# Define the directory where the pickled files are stored
directory = "E:\\cascade\\ReachData_for_sensitivity\\XC\\XC_data_output\\"

# Initialize an empty dictionary to store the data outputs
output_data_dict = {}

# Iterate over each pickled file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        # Split the filename by underscores and remove the extension
        parts = filename.split("_")
        
       # Extract the integer part from the filename
        reach_data_part = parts[-1].replace('.pkl', '')  # Remove '.pkl' extension
        ReachData = int(reach_data_part)  # Convert the part to an integer
     
        # Load the pickled file
        with open(os.path.join(directory, filename), "rb") as file:
            data_output = pickle.load(file)
        
        # Store the data_output in the dictionary
        output_data_dict[(ReachData)] = data_output

# A dictionary containing the data outputs for each ReachData
# Data_output for a specific ReachData can be access like this:
# data_output = output_data_dict[(ReachDatnumber)]
data_output = output_data_dict [(5)]


         
# Initialize an empty dictionary to store the mobilized volumes/  for each combination
mobilized_volumes_dict = {}

# Iterate over each combination in output_data_dict
for combination, data_output in output_data_dict.items():
    # Extract the desired output from the data_output
    output_name = 'Mobilized volume [m^3]'    #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    mobilized_volume = data_output[output_name]
 

    # Store the output in the respective dictionary according to thier name
    mobilized_volumes_dict[combination] = mobilized_volume
 
    


# Initialize an empty dictionary to store the total sum/ median/ averages  of chosen data output for each combination
total_mobilized_volume_dict = {}


# Iterate over each combination in chosen output's dictionary
for combination, mobilized_volume_array in mobilized_volumes_dict.items():   
    # Sum or median of the chosen outputs for the current combination along axis=0
    total_sum = np.sum(mobilized_volume_array, axis=0)
   # median_value = np.median(D50_active_layer_array, axis=0)
    
    
    # Store the total sum/median in the total/median dictionary
    total_mobilized_volume_dict[combination] = total_sum
   # median_D50_active_layer_dict[combination] = median_value

# total/median dict contains the total sum / median of output for each combination
# total sum/ median for a specific combination can be access like this:
total_sum = total_mobilized_volume_dict[(5)]


import pandas as pd
import numpy as np


# Create a DataFrame from the total_mobilized_volume_dict
df_mob = pd.DataFrame(total_mobilized_volume_dict).T  # Transpose to get combinations as rows and reaches as columns

# Set appropriate column names (e.g., Reach 1, Reach 2, ...)
df_mob.columns = [f'Reach {i+1}' for i in range(df_mob.shape[1])]

# Sort the DataFrame by its index
df_mob = df_mob.sort_index()

YC = df_mob.iloc [:, 4]



import safepython.VBSA as VB # module to perform VBSA
import safepython.plot_functions as pf # module to visualize the results
from safepython.model_execution import model_execution # module to execute the model
from safepython.sampling import AAT_sampling, AAT_sampling_extend  # module to
# perform the input sampling
from safepython.util import aggregate_boot # function to aggregate results across bootstrap
# resamples

# Number of uncertain parameters subject to SA:
M = 2
X_Labels = ['Wac', 'slope']



# Compute main (first-order) and total effects:
Si, STi = VB.vbsa_indices(YA.to_numpy(), YB.to_numpy(), YC.to_numpy(), M)

# Plot results:
plt.figure()
plt.subplot(131)
pf.boxplot1(Si, X_Labels=X_Labels, Y_Label='main effects')
plt.subplot(132)
pf.boxplot1(STi, X_Labels=X_Labels, Y_Label='total effects')
plt.subplot(133)
pf.boxplot1(STi-Si, X_Labels=X_Labels, Y_Label='interactions')
plt.show()

# Plot main and total effects in one plot:
plt.figure()
pf.boxplot2(np.stack((Si, STi)), X_Labels=X_Labels,
            legend=['main effects', 'total effects'])
plt.show()

# Check the model output distribution (if multi-modal or highly skewed, the
# variance-based approach may not be adequate):
Y = np.concatenate((YA, YC))
plt.figure()
pf.plot_cdf(Y, Y_Label='NSE')
plt.show()
plt.figure()
fi, yi = pf.plot_pdf(Y, Y_Label='NSE')
plt.show()


