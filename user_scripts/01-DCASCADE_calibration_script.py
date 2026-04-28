import os
import sys
import numpy as np
import pandas as pd
import spotpy
import seaborn as sns
import matplotlib.pyplot as plt

# Add source folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from spotpy_setup import spotpy_setup

# ------------------------ 1. Setup SPOTPY ------------------------
# Initialize the model for calibration
spot_setup = spotpy_setup()

# Define the sampler: Shuffled Complex Evolution (SCE-UA)
sampler = spotpy.algorithms.sceua(
    spot_setup, 
    dbname='../cascade_results/Spotpy_calibration_SCEUA_dcascade',  # results file name
    dbformat='csv'
    )

# ------------------------ 2. Run Calibration ------------------------
# Number of repetitions (iterations)
rep = 1000
# Sampling with SCE-UA algorithm
sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1)

# ------------------------ 3. Load Results ------------------------
results = spotpy.analyser.load_csv_results('../cascade_results/Spotpy_calibration_SCEUA_dcascade')

# Inspect results
print("Results type:", type(results))
print("Structured array fields:", results.dtype.names)

# Convert structured array to pandas DataFrame for easier handling
results_df = pd.DataFrame(results)
print(results_df.head())

# Display parameter field names
param_names = spotpy.analyser.get_parameternames(results)
print("Parameter names:", param_names)

# Display best parameter set
best_params = spotpy.analyser.get_best_parameterset(results, maximize=False)
print("Best parameter set:", best_params)

# Translate back into D16, D50, D84 for interpretation
D16_0 = best_params['parbase_0'][0]
D50_0 = best_params['parbase_0'][0] + best_params['pardelta1_0'][0]
D84_0 = best_params['parbase_0'][0] + best_params['pardelta1_0'][0] + best_params['pardelta2_0'][0]

# D16_1 = best_params['parbase_1'][0]
# D50_1 = best_params['parbase_1'][0] + best_params['pardelta1_1'][0]
# D84_1 = best_params['parbase_1'][0] + best_params['pardelta1_1'][0] + best_params['pardelta2_1'][0]

print(f"First Reach, Best D16: {D16_0:.4f}, D50: {D50_0:.4f}, D84: {D84_0:.4f}")
# print(f"Second Reach, Best D16: {D16_1:.4f}, D50: {D50_1:.4f}, D84: {D84_1:.4f}")

# ------------------------ 4. Plot Objective Function Trace ------------------------
fig= plt.figure(1,figsize=(9,5))
plt.plot(results['like1'])
plt.ylabel('RMSE')
plt.xlabel('Iteration')
plt.title('Objective Function Trace')
fig.savefig('../cascade_results/Spotpy_calibration_SCEUA_objectivefunctiontrace.png',dpi=300)
plt.show()

# ------------------------ 5. Load Model Output for Evaluation ------------------------
# Path to pickle output of a simulation
path = "../cascade_results/"
name_simu = 'Vjosa_test'
data_output = pd.read_pickle(open( path + name_simu + '.p' , "rb"))
evaluation = data_output['Volume out [m^3]'][:,1] 

# Plot the best model run vs evaluation
spotpy.analyser.plot_bestmodelrun(results, evaluation, fig_name="../cascade_results/Spotpy_calibration_Best_model_run.png")

# ------------------------ 6. Parameter Pairplot ------------------------
# Extract only parameter columns for plotting
param_cols = spotpy.analyser.get_parameter_fields(results)
param_cols.append('like1')  # include objective function

results_df['D16'] = results_df['parbase_0']
results_df['D50'] = results_df['parbase_0'] + results_df['pardelta1_0']
results_df['D84'] = results_df['parbase_0'] + results_df['pardelta1_0'] + results_df['pardelta2_0']

param_cols.append('D16')
param_cols.append('D50')
param_cols.append('D84')

results_df_best_10 = results_df.nsmallest(10, 'like1')


sns.pairplot(results_df_best_10[param_cols])
plt.show()