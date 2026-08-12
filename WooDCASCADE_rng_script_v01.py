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

n_classes = 1 ### Change to 2 if applying Wood Class; select appropriate f(x) in loop
timescale = 4
update_slope = False
save_dep_layer = 'always'
roundpar = 0

n_iterations = 10
export_pickle = True
pickle_file = os.path.join(path_results, 'Name_of_output_file.p')

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
    deposit = ReachData.deposit ### One class of wood
    # deposit = np.array([ReachData.deposit_L,ReachData.deposit_S]).T  # Two classes of wood
    Qbi_dep_in = np.zeros((n_reaches, 1, n_classes))
    for n in range(len(ReachData)):
        Qbi_dep_in[n] = deposit[n]

    # Run the model
    data_output, extended_output = DCASCADE_main(
        ReachData, Network, Qbi_dep_in, timescale,
        roundpar, update_slope, save_dep_layer, n_classes
    )

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
