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
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import geopandas as gpd


# ---------------------Path to the pickle output
path = "..\\cascade_results\\"
name_simu = 'Tagliamento_bf'

# ---------------------Path to the input river network (.shp) or (.csv)
path_river_network = "..\\inputs\\Tagliamento_river\\"
name_river_network = "Reach_data_tag_bf.csv"

# ---------------------Folder to store the plots
figure_folder = path + 'figures_all_reaches_sum\\'

if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

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
# Plot average or median, x axis is the reach index
##############################################################################

data_output = pd.read_pickle(open(path + name_simu + '.p', "rb"))
my_data = data_output[output_name]
n_reach = my_data.shape[1]
FromN_idx = np.arange(1, n_reach + 1, 1)
n_time = my_data.shape[0]

full_years_number = n_time // 365
rest_days = n_time % 365

# create figure and graph axes
fig = plt.figure()
ax = plt.subplot(111)

if rest_days != 0:
    year_number = full_years_number + 1
else:
    year_number = full_years_number

color = iter(plt.cm.viridis(np.linspace(0, 1, year_number)))

sum_all = np.zeros(n_reach)

t_0 = 0
t_end = 364   # (365 - 1) since time 0 is the first day
year = year_0

for idx_year in range(year_number):

    if t_0 == n_time:
        continue

    if (n_time - t_0) < 365:
        t_end = n_time - (t_0 + 1)

    time_list = [i for i in range(t_0, t_end + 1, 1)]
    c = next(color)

    if output_name in ['Volume out [m^3]', 'Volume in [m^3]', 'Transport capacity [m^3]', 'Sediment budget [m^3]']:
        my_sum = np.sum(my_data[time_list, :], axis=0)
        ax.plot(FromN_idx, my_sum, label=str(year), color=c)

        if ((t_end + 1) - t_0) == 365:
            sum_all += my_sum

    if output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
        my_median = np.median(my_data[time_list, :], axis=0)
        ax.plot(FromN_idx, my_median, label=str(year), color=c)

    t_0 = t_end + 1
    t_end = t_end + 365
    year += 1

# Plot the average for certain output types
if output_name in ['Volume out [m^3]', 'Volume in [m^3]', 'Transport capacity [m^3]', 'Sediment budget [m^3]']:
    if full_years_number > 1:
        sum_all /= full_years_number
        ax.plot(FromN_idx, sum_all, linewidth=2.5, color='black', label='Average')

# Add initial D50 line for D50 plots
if output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
    ax.plot(FromN_idx, initial_d50, color='black', linewidth=2.5, linestyle='--', label='Initial D50')

# Add a horizontal line at 0
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
fig.savefig(figure_folder + str(new_name) + '_yearly.png', dpi=300)
plt.close(fig)


##############################################################################
# Additional plot: all D50 active layer lines, x axis is the reach index
# Red = timestep 1, Blue = timestep n_time
##############################################################################

output_name = 'D50 active layer [m]'
my_data = data_output[output_name]
n_reach = my_data.shape[1]
FromN_idx = np.arange(1, n_reach + 1, 1)
n_time = my_data.shape[0]

# create figure and graph axes
fig = plt.figure()
ax = plt.subplot(111)

# reversed coolwarm: red at start, blue at end
cmap = plt.cm.coolwarm_r

for t in range(n_time):
    color = cmap(t / (n_time - 1))
    ax.plot(FromN_idx, my_data[t, :], color=color, linewidth=1)

# Add initial D50 as black line
ax.plot(FromN_idx, initial_d50, color='black', linewidth=2.5, linestyle='--', label='Initial D50')

ax.set_xlabel('Reach index (FromN)', fontsize=18)
ax.set_ylabel(output_name, fontsize=16)
ax.tick_params(axis='y', which='major', labelsize=15)
ax.tick_params(axis='x', which='major', labelsize=12)

# colorbar showing timestep range
norm = mpl.colors.Normalize(vmin=1, vmax=n_time)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Timestep', fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.legend(fontsize=12)

fig.set_tight_layout(True)
fig.set_size_inches(2000. / fig.dpi, 700. / fig.dpi)
new_name = rename_names(output_name)
fig.savefig(figure_folder + str(new_name) + '_all_timesteps_colorbar.png', dpi=300)
plt.close(fig)