import numpy as np
import pandas as pd
import os
import pickle
import copy

from WooDCASCADE_loop_v01 import DCASCADE_main
from preprocessing import graph_preprocessing

# ---- USER SETTINGS ---- #
path_river_network = r'C:\path\to\ReachFile\folder\\'
name_river_network = r'ReachFile.csv' # Located at the folder stated at path_river_network
path_results = r"C:\path\to\Output\folder\\"

export_pickle = True
pickle_file = os.path.join(path_results, 'Name_of_output_file.p')

n_classes = 1
timescale = 4  # Initial state + 3 modeled events <-- timescale = 4
n_iterations = 10

# Function options:
#   "Global" and "HiLo" use n_classes = 1
#   "WoodClass" and "HiLoWoodClass" use n_classes = 2
function_family = "Global"

recruitment_schedule = {
    # 0: [(5, 5.0)],          # Add 5.0 m³ of total wood to reach 5
    # 0: [(5, [3.0, 2.0])],   # Two classes: add 3.0 m³ Large and 2.0 m³ Medium
}

barrier_schedule = {
    # 0: [(5, 0.50)],
}

# Used only by "HiLo" and "HiLoWoodClass"
hilo_event_schedule = {
    # 0: "hi",
    # 1: "lo",
    # 2: "hi",
}

update_slope = False
save_dep_layer = 'always'
roundpar = 0


# ---- STORAGE ---- #
all_runs_output = {}  # {iteration_number: data_output}

# ---- RUN LOOP ---- #
for iteration in range(1, n_iterations + 1):
    print(f"Running iteration {iteration}...")

    # Re-initialize everything
    ReachData = pd.read_csv(os.path.join(path_river_network, name_river_network))
    ReachData = ReachData.sort_values(by='FromN', ignore_index=True)
    Network = graph_preprocessing(ReachData)

    n_reaches = len(ReachData)
    
    # Load the initial wood deposit according to the number of classes.
    if n_classes == 1:
        # One-class models: total deposited wood
        deposit = ReachData[['deposit']].to_numpy(dtype=float)
    
    elif n_classes == 2:
        # Two-class models: class order is [Large, Medium]
        deposit = ReachData[['deposit_L', 'deposit_M']].to_numpy(dtype=float)
    
    else:
        raise ValueError(
            f"n_classes must be either 1 or 2; received {n_classes}."
        )
    
    # Required model shape: [reach, deposit layer, wood class]
    Qbi_dep_in = deposit[:, np.newaxis, :]

    # Run the model
    data_output, extended_output = DCASCADE_main(
        ReachData, Network, Qbi_dep_in, timescale,
        roundpar, update_slope, save_dep_layer, n_classes,
        function_family=function_family,
        recruitment_schedule=recruitment_schedule,
        barrier_schedule=barrier_schedule,
        event_schedule=hilo_event_schedule)

    # Optional: clean output of unused keys
    data_output_clean = copy.deepcopy(data_output)
    drop_keys = [k for k in data_output_clean if k.endswith('per class [m^3/s]')]
    for k in drop_keys:
        del data_output_clean[k]

    all_runs_output[iteration] = data_output_clean

# ---- EXPORT TO PICKLE ---- #
if export_pickle:
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_runs_output, f)
    print(f"All {n_iterations} runs saved to {pickle_file}")
