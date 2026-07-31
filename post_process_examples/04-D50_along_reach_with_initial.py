"""
Plot D-CASCADE basic outputs

yearly sum or median (for D50), plotted along reach indexes

Choose between:
'Volume out [m^3]':         total volume of sediment leaving a reach per time step (= sediment flux x time step)
'Volume in [m^3]':          total volume of sediment entering a reach per time step
'Transport capacity [m^3]': total transport capacity computed in a reach per time step (= volume out if the supply is not limited)
'Sediment budget [m^3]':    total sediment budget per time step (+ deposition, - erosion) (= vol in - vol out)
'D50 active layer [m]':     D50 of the active layer per time step (used to computed the transport capacity)
'D50 volume out [m]' :      D50 of the volume leaving the reach per time step
"""

# Libraries
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import geopandas as gpd


# ---------------------Path to the pickle output
path = "..\\cascade_results\\"
name_simu = 'Tagliamento_bf_al005'

# ---------------------Path to the input river network (.shp) or (.csv)
path_river_network = "..\\inputs\\Tagliamento_river\\"
name_river_network = "Reach_data_tag_bf.csv"

# ---------------------Folder to store the plots
figure_folder = os.path.join(path, 'figures_all_reaches_sum')
csv_folder = os.path.join(path, 'csv_all_reaches_sum')

if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# --------------------Output name you want to plot
output_name = 'Volume out [m^3]'
# 'D50 active layer [m]', 'D50 volume out [m]', 'Sediment budget [m^3]',
# 'Transport capacity [m^3]', 'Volume in [m^3]', 'Volume out [m^3]'

# --------------------First year simulated (for legend)
year_0 = 2019


##############################################################################
# Useful function for naming figures
##############################################################################

def rename_names(output_name):
    """
    For example, renames 'Volume out [m^3]' into 'Volume_out'
    to use for saving csv and plots
    """
    new_name = ''
    for c in output_name:
        if c == ' ':
            new_name = new_name + '_'
        elif c == '[':
            break
        elif c == '-':
            break
        else:
            new_name = new_name + c
    new_name = new_name[:-1]
    return new_name


##############################################################################
# Read initial river network data and extract initial D50
##############################################################################

river_path = os.path.join(path_river_network, name_river_network)

if river_path.lower().endswith(".csv"):
    river_df = pd.read_csv(river_path)
elif river_path.lower().endswith(".shp"):
    river_df = gpd.read_file(river_path)
else:
    raise ValueError("River network file must be .csv or .shp")

if "D50" not in river_df.columns:
    raise ValueError("Column 'D50' not found in river network file.")

initial_d50 = river_df["D50"].to_numpy(dtype=float)


##############################################################################
# Read pickle output
##############################################################################

with open(os.path.join(path, name_simu + '.p'), "rb") as f:
    data_output = pickle.load(f)

my_data = data_output[output_name]
n_reach = my_data.shape[1]
FromN_idx = np.arange(1, n_reach + 1, 1)
n_time = my_data.shape[0]

full_years_number = n_time // 365
rest_days = n_time % 365

if rest_days != 0:
    year_number = full_years_number + 1
else:
    year_number = full_years_number


##############################################################################
# Plot yearly sum/median along reaches
##############################################################################

fig = plt.figure()
ax = plt.subplot(111)

color = iter(plt.cm.viridis(np.linspace(0, 1, year_number)))
sum_all = np.zeros(n_reach)

t_0 = 0
t_end = 364
year = year_0

# For CSV output
yearly_results = []

for idx_year in range(year_number):

    if t_0 == n_time:
        continue

    if (n_time - t_0) < 365:
        t_end = n_time - 1

    time_list = list(range(t_0, t_end + 1))
    c = next(color)

    row_dict = {
        "year": year,
        "start_timestep": t_0 + 1,
        "end_timestep": t_end + 1,
        "n_days": len(time_list)
    }

    if output_name in ['Volume out [m^3]', 'Volume in [m^3]', 'Transport capacity [m^3]', 'Sediment budget [m^3]']:
        my_sum = np.sum(my_data[time_list, :], axis=0)
        ax.plot(FromN_idx, my_sum, label=str(year), color=c)

        for i, val in enumerate(my_sum):
            row_dict[f"reach_{i+1}"] = val

        if len(time_list) == 365:
            sum_all += my_sum

        yearly_results.append(row_dict)

    if output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
        my_median = np.median(my_data[time_list, :], axis=0)
        ax.plot(FromN_idx, my_median, label=str(year), color=c)

        for i, val in enumerate(my_median):
            row_dict[f"reach_{i+1}"] = val

        yearly_results.append(row_dict)

    t_0 = t_end + 1
    t_end = t_end + 365
    year += 1

# Add average line for volume-related outputs
if output_name in ['Volume out [m^3]', 'Volume in [m^3]', 'Transport capacity [m^3]', 'Sediment budget [m^3]']:
    if full_years_number > 1:
        avg_full_years = sum_all / full_years_number
        ax.plot(FromN_idx, avg_full_years, linewidth=2.5, color='black', label='Average')

        avg_row = {
            "year": "Average_full_years",
            "start_timestep": "",
            "end_timestep": "",
            "n_days": 365
        }
        for i, val in enumerate(avg_full_years):
            avg_row[f"reach_{i+1}"] = val
        yearly_results.append(avg_row)

# Add initial D50 line for D50 outputs
if output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
    ax.plot(FromN_idx, initial_d50, color='black', linewidth=2.5, linestyle='--', label='Initial D50')

    init_row = {
        "year": "Initial_D50",
        "start_timestep": "",
        "end_timestep": "",
        "n_days": ""
    }
    for i, val in enumerate(initial_d50):
        init_row[f"reach_{i+1}"] = val
    yearly_results.append(init_row)

# Add horizontal zero line
if output_name == 'Delta z [m]' or output_name == 'Sediment budget [m^3]':
    ax.hlines(0, xmin=3, xmax=42, linestyle='--', color='gray')

ax.legend(fontsize=12)
ax.set_xlabel('Reach index (FromN)', fontsize=18)
ax.set_ylabel(output_name, fontsize=16)
ax.tick_params(axis='y', which='major', labelsize=15)
ax.tick_params(axis='x', which='major', labelsize=12)

fig.set_tight_layout(True)
fig.set_size_inches(2000. / fig.dpi, 700. / fig.dpi)

new_name = rename_names(output_name)

# Save yearly plot
fig.savefig(os.path.join(figure_folder, f'{new_name}_yearly.png'), dpi=300)
plt.close(fig)

# Save yearly CSV
yearly_df = pd.DataFrame(yearly_results)
yearly_df.to_csv(os.path.join(csv_folder, f'{new_name}_yearly.csv'), index=False)


##############################################################################
# Additional plot: all D50 active layer lines, x axis is the reach index
# Red = timestep 1, Blue = timestep n_time
##############################################################################

output_name_d50 = 'D50 active layer [m]'
my_data_d50 = data_output[output_name_d50]
n_reach_d50 = my_data_d50.shape[1]
FromN_idx_d50 = np.arange(1, n_reach_d50 + 1, 1)
n_time_d50 = my_data_d50.shape[0]

fig = plt.figure()
ax = plt.subplot(111)

cmap = plt.cm.coolwarm_r

for t in range(n_time_d50):
    color = cmap(t / (n_time_d50 - 1))
    ax.plot(FromN_idx_d50, my_data_d50[t, :], color=color, linewidth=1)

# Add initial D50 as black dashed line
ax.plot(FromN_idx_d50, initial_d50, color='black', linewidth=2.5, linestyle='--', label='Initial D50')

ax.set_xlabel('Reach index (FromN)', fontsize=18)
ax.set_ylabel(output_name_d50, fontsize=16)
ax.tick_params(axis='y', which='major', labelsize=15)
ax.tick_params(axis='x', which='major', labelsize=12)

norm = mpl.colors.Normalize(vmin=1, vmax=n_time_d50)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Timestep', fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.legend(fontsize=12)

fig.set_tight_layout(True)
fig.set_size_inches(2000. / fig.dpi, 700. / fig.dpi)

new_name_d50 = rename_names(output_name_d50)

# Save D50 all timestep plot
fig.savefig(os.path.join(figure_folder, f'{new_name_d50}_all_timesteps_colorbar.png'), dpi=300)
plt.close(fig)


##############################################################################
# Save all timesteps D50 active layer to CSV
##############################################################################

d50_all_df = pd.DataFrame(my_data_d50, columns=[f"reach_{i+1}" for i in range(n_reach_d50)])
d50_all_df.insert(0, "timestep", np.arange(1, n_time_d50 + 1))
d50_all_df.to_csv(os.path.join(csv_folder, f'{new_name_d50}_all_timesteps.csv'), index=False)

print("Plots saved in:", figure_folder)
print("CSV files saved in:", csv_folder)