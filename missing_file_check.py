# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:15:12 2024

@author: Sahansila
"""


import os
import re

# Directory where the model run files are stored
directory = "E:\\Sahansila\\SAFE_output\\EH_2parameter_AW_Slope_vel0.1H\\data_output_batch_test\\"

# Pattern to match filenames and extract the number
pattern = re.compile(r"^data_output_ReachData_modified_(\d+)\.pkl$")

# Expected range of model runs
expected_runs = set(range(1, 5000))  # Numbers from 1 to 2000

try:
    # List all files in the directory
    files_in_directory = os.listdir(directory)
    
    # Extract numbers from filenames that strictly match the pattern
    actual_runs = set(
        int(pattern.search(file).group(1)) 
        for file in files_in_directory 
        if pattern.match(file)
    )
    
    # Find missing runs
    missing_runs = sorted(expected_runs - actual_runs)

    # Create missing files in the directory
    for missing in missing_runs:
        missing_file_path = os.path.join(directory, f"data_output_ReachData_modified_{missing}.pkl")
        # Create an empty file
        with open(missing_file_path, "w") as f:
            pass  # Empty file creation
        print(f"Created: {missing_file_path}")

    print(f"Missing files created successfully. Total created: {len(missing_runs)}")
except FileNotFoundError:
    print(f"Directory '{directory}' does not exist. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")