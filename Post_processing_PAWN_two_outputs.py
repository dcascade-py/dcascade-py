

# import libraries
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  
import os




# Define the directory where the data otput pickled files are stored
directory = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\data_output\\data_output_X_5000_AW_unif_slope_normal\\"
# Initialize an empty dictionary to store the data outputs
output_data_dict = {}
output_data_dict_1 = {}


# Define the number of data files to read
number_of_files_to_read = 5000  # Specify the desired number of files

selective_indices = np.arange(1,number_of_files_to_read+1)


# Two different outputs


output_data_dict_1 = {}
output_data_dict_2 = {}

# Define the directory where the data otput pickled files are stored
directory = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\data_output\\data_output_X_5000_AW_unif_slope_normal\\"
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
    



    # Store the chosen output from the data output in the dictionary
    output_name_1 =  'D50 active layer [m]'  # choose the output here
    # 'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    output_data_dict_1[int(i)] = data_output[output_name_1]
    # the integer part is going to the key of the dictionary
 
     
    # # Store the chosen output from the data output in the dictionary
    # output_name_2 = 'Transported [m^3]'  # choose the output here
    # # 'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    # output_data_dict_2[int(i)] = data_output[output_name_2]
    # # the integer part is going to the key of the dictionary
    

    
    # Store the chosen output from the data output in the dictionary
    output_name = 'Budget' 
    output_name_2 = 'Mobilized [m^3]' 
    output_name_3 = 'Transported [m^3]'
    # choose the output here
    # 'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
    output_data_dict_2[int(i)] = data_output[output_name_3] - data_output [output_name_2]
    # the integer part is going to the key of the dictionary
    
 
 

# Initialize an empty dictionary to store the summary of chosen data output for each combination
summary_output_data_1 = {}


# Iterate over each combination in chosen output's dictionary
for key, value in output_data_dict_1.items():   
    # Summary of the chosen outputs for the sample(key) along axis=0
    summary_output_data_1[key]= np.median(value, axis=0)
    #intstead of sum, other statistical 



# Create a DataFrame from the summary output
df_Y1 = pd.DataFrame( summary_output_data_1).T  # Transpose to get samples as rows and reaches as columns
#the keys (index) will not be in order

# Set appropriate column names (e.g., Reach 1, Reach 2, ...)
df_Y1.columns = [f'Reach {i+1}' for i in range(df_Y1.shape[1])]


# Sort the DataFrame by its index
df_Y1 = df_Y1.sort_index() 
 

# Initialize an empty dictionary to store the summary of chosen data output for each combination
summary_output_data_2 = {}


# Iterate over each combination in chosen output's dictionary
for key, value in output_data_dict_2.items():   
    # Summary of the chosen outputs for the sample(key) along axis=0
    summary_output_data_2[key]= np.sum(value, axis=0)
    #intstead of sum, other statistical 



# Create a DataFrame from the summary output
df_Y2 = pd.DataFrame( summary_output_data_2).T  # Transpose to get samples as rows and reaches as columns
#the keys (index) will not be in order

# Set appropriate column names (e.g., Reach 1, Reach 2, ...)
df_Y2.columns = [f'Reach {i+1}' for i in range(df_Y2.shape[1])]


# Sort the DataFrame by its index
df_Y2 = df_Y2.sort_index() 


# Define the folder where you want to save the figures

output_folder = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\figures_D50AL_budget"


# for N in range (1,45):


#     # new_df = pd.DataFrame({
#     #     f"{output_name_1}": df_Y1[f"Reach {N}"],
#     #     f"{output_name_2}": df_Y2[f"Reach {N}"]
#     # })
    
#     new_df = pd.DataFrame({
#         f"{output_name_1}": df_Y1[f"Reach {N}"],
#         f"{output_name}": df_Y2[f"Reach {N}"]
#     })
    
#     new_df_1 = new_df.sort_values(by=f"{output_name_1}", ascending=True)
    
#     plt.figure()
    
#     plt.figure(figsize=(9, 6))
    
    
#     plt.scatter(new_df_1.iloc[:,0], new_df_1.iloc[:,1], label = len(new_df_1), cmap='viridis', alpha=0.7)
     
    
#     # Add labels, title, legend, and grid
#     plt.xlabel( f"{output_name_1} for Reach {N}", fontsize = 12)
#     plt.ylabel(f"{output_name} for Reach {N}", fontsize = 12)
#     plt.title(f"Corelation between {output_name_1} and {output_name} for Reach {N}", fontsize = 10)
#     plt.legend()
#     plt.grid(True)
    
    
    
#     # Save the active width indices plot for all the reach in the figures folder
#     figure_filename = f"figure_Reach_{N}.png"
#     save_path = os.path.join(output_folder, figure_filename)
#     plt.savefig(save_path, format='png', dpi=200)  # Save as JPG with 300 dpi
#     plt.close()  # Close the figure to free up memory
    


# scatter plot one over another
N_output = 24

N_D50 = 23

new_df = pd.DataFrame({
    f"{output_name_1}": df_Y1[f"Reach {N_D50}"],
    f"{output_name}": df_Y2[f"Reach {N_output}"]
})

new_df_1 = new_df.sort_values(by=f"{output_name_1}", ascending=True)

plt.scatter(new_df_1.iloc[:,0], new_df_1.iloc[:,1], label = len(new_df_1), cmap='viridis', alpha=0.7)

# plt.scatter(new_df_1.iloc[:,0], new_df_1.iloc[:,1], label= i, cmap='viridis', alpha=0.7, facecolor = 'none', edgecolors='r')


# plt.plot(new_df_1.iloc[:,0], new_df_1.iloc[:,1], color= 'blue', label= i)  # Plot column data for each sample

# Add labels, title, legend, and grid
plt.xlabel( f"{output_name_1} for Reach {N_D50}", fontsize = 18)
plt.ylabel(f"{output_name} for Reach {N_output}", fontsize = 18)
plt.title(f"Corelation between {output_name_1} of Reach {N_D50} and {output_name} of of Reach {N_output}", fontsize = 15)
plt.legend()
plt.grid(True)




