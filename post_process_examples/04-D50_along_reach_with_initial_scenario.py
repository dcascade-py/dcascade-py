"""
Plot D-CASCADE outputs for multiple scenarios

This script:
1. creates yearly plots for each scenario
2. saves yearly CSV output for each scenario
3. creates D50 all-timestep plot for each scenario
4. saves D50 all-timestep CSV for each scenario
5. creates one combined plot with all scenarios together

Outputs supported:
'Volume out [m^3]'
'Volume in [m^3]'
'Transport capacity [m^3]'
'Sediment budget [m^3]'
'D50 active layer [m]'
'D50 volume out [m]'
"""

# Libraries
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib as mpl


# =============================================================================
# USER INPUTS
# =============================================================================

# Path to D-CASCADE result files
path = "..\\cascade_results\\"

# Path to river network input
path_river_network = "..\\inputs\\Tagliamento_river\\"

# Folder to store outputs
figure_folder = os.path.join(path, "figures_multi_scenario")
csv_folder = os.path.join(path, "csv_multi_scenario")

os.makedirs(figure_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

# First year simulated
year_0 = 2019

# Output to analyse
output_name = 'D50 active layer [m]'
# Try also:
# 'Volume in [m^3]'
# 'Transport capacity [m^3]'
# 'Sediment budget [m^3]'
# 'D50 active layer [m]'
# 'D50 volume out [m]'

# Scenario definitions
scenarios = [
    {"name_simu": "Tagliamento_50bf_al005", "river_file": "Reach_data_tag_50bf.csv", "label": "50bf_al005"},
    {"name_simu": "Tagliamento_50bf_al002", "river_file": "Reach_data_tag_50bf.csv", "label": "50bf_al002"},
    {"name_simu": "Tagliamento_bf_al005",   "river_file": "Reach_data_tag_bf.csv",   "label": "bf_al005"},
    {"name_simu": "Tagliamento_bf_al002",   "river_file": "Reach_data_tag_bf.csv",   "label": "bf_al002"},
]


# =============================================================================
# FUNCTIONS
# =============================================================================

def rename_names(output_name):
    """
    Renames 'Volume out [m^3]' into 'Volume_out'
    """
    new_name = ''
    for c in output_name:
        if c == ' ':
            new_name += '_'
        elif c == '[':
            break
        elif c == '-':
            break
        else:
            new_name += c
    new_name = new_name[:-1]
    return new_name


def read_river_network(river_path):
    if river_path.lower().endswith(".csv"):
        river_df = pd.read_csv(river_path)
    elif river_path.lower().endswith(".shp"):
        river_df = gpd.read_file(river_path)
    else:
        raise ValueError("River network file must be .csv or .shp")

    if "D50" not in river_df.columns:
        raise ValueError(f"Column 'D50' not found in {river_path}")

    initial_d50 = river_df["D50"].to_numpy(dtype=float)
    return river_df, initial_d50


def compute_yearly_profiles(my_data, output_name, year_0):
    """
    Returns:
    - yearly_results: list of dictionaries for CSV
    - lines_to_plot: list of tuples (year_label, values)
    - avg_full_years: average of complete years for volume outputs, else None
    """
    n_time = my_data.shape[0]
    n_reach = my_data.shape[1]

    full_years_number = n_time // 365
    rest_days = n_time % 365
    year_number = full_years_number + 1 if rest_days != 0 else full_years_number

    sum_all = np.zeros(n_reach)
    yearly_results = []
    lines_to_plot = []

    t_0 = 0
    t_end = 364
    year = year_0

    for idx_year in range(year_number):
        if t_0 == n_time:
            continue

        if (n_time - t_0) < 365:
            t_end = n_time - 1

        time_list = list(range(t_0, t_end + 1))

        row_dict = {
            "year": year,
            "start_timestep": t_0 + 1,
            "end_timestep": t_end + 1,
            "n_days": len(time_list)
        }

        if output_name in ['Volume out [m^3]', 'Volume in [m^3]', 'Transport capacity [m^3]', 'Sediment budget [m^3]']:
            vals = np.sum(my_data[time_list, :], axis=0)
            lines_to_plot.append((str(year), vals))

            for i, val in enumerate(vals):
                row_dict[f"reach_{i+1}"] = val

            if len(time_list) == 365:
                sum_all += vals

            yearly_results.append(row_dict)

        elif output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
            vals = np.median(my_data[time_list, :], axis=0)
            lines_to_plot.append((str(year), vals))

            for i, val in enumerate(vals):
                row_dict[f"reach_{i+1}"] = val

            yearly_results.append(row_dict)

        t_0 = t_end + 1
        t_end = t_end + 365
        year += 1

    avg_full_years = None
    if output_name in ['Volume out [m^3]', 'Volume in [m^3]', 'Transport capacity [m^3]', 'Sediment budget [m^3]']:
        if full_years_number > 0:
            avg_full_years = sum_all / full_years_number

            avg_row = {
                "year": "Average_full_years",
                "start_timestep": "",
                "end_timestep": "",
                "n_days": 365
            }
            for i, val in enumerate(avg_full_years):
                avg_row[f"reach_{i+1}"] = val
            yearly_results.append(avg_row)

    return yearly_results, lines_to_plot, avg_full_years


def set_all_reach_ticks(ax, n_reach):
    """
    Show all reach labels: 1, 2, 3, ..., n_reach
    """
    ax.set_xticks(np.arange(1, n_reach + 1, 1))
    ax.set_xticklabels(np.arange(1, n_reach + 1, 1), rotation=90, fontsize=8)


# =============================================================================
# MAIN LOOP OVER SCENARIOS
# =============================================================================

combined_profiles = []

for sc in scenarios:
    name_simu = sc["name_simu"]
    river_file = sc["river_file"]
    scenario_label = sc["label"]

    print(f"Processing scenario: {scenario_label}")

    river_path = os.path.join(path_river_network, river_file)
    _, initial_d50 = read_river_network(river_path)

    result_path = os.path.join(path, name_simu + ".p")
    with open(result_path, "rb") as f:
        data_output = pickle.load(f)

    if output_name not in data_output:
        raise ValueError(f"Output '{output_name}' not found in {name_simu}.p")

    my_data = data_output[output_name]
    n_time = my_data.shape[0]
    n_reach = my_data.shape[1]
    FromN_idx = np.arange(1, n_reach + 1, 1)

    # -------------------------------------------------------------------------
    # 1. Yearly plot and CSV
    # -------------------------------------------------------------------------
    yearly_results, lines_to_plot, avg_full_years = compute_yearly_profiles(my_data, output_name, year_0)

    fig = plt.figure(figsize=(20, 7))
    ax = plt.subplot(111)

    colors = iter(plt.cm.viridis(np.linspace(0, 1, len(lines_to_plot))))

    for year_label, vals in lines_to_plot:
        c = next(colors)
        ax.plot(FromN_idx, vals, label=year_label, color=c)

    if output_name in ['Volume out [m^3]', 'Volume in [m^3]', 'Transport capacity [m^3]', 'Sediment budget [m^3]']:
        if avg_full_years is not None:
            ax.plot(FromN_idx, avg_full_years, linewidth=2.5, color='black', label='Average')
            combined_profiles.append((scenario_label, avg_full_years))

    if output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
        ax.plot(FromN_idx, initial_d50, color='black', linewidth=2.5, linestyle='--', label='Initial D50')

        # for combined plot, use last yearly line or mean over all timesteps
        scenario_profile = np.median(my_data, axis=0)
        combined_profiles.append((scenario_label, scenario_profile))

    if output_name == 'Delta z [m]' or output_name == 'Sediment budget [m^3]':
        ax.axhline(0, linestyle='--', color='gray')

    ax.legend(fontsize=10)
    ax.set_xlabel('Reach index (FromN)', fontsize=14)
    ax.set_ylabel(output_name, fontsize=14)
    ax.set_title(f"{scenario_label} | {output_name}", fontsize=16)
    ax.tick_params(axis='y', which='major', labelsize=11)

    # Show all reach labels
    set_all_reach_ticks(ax, n_reach)

    fig.tight_layout()

    new_name = rename_names(output_name)
    fig.savefig(os.path.join(figure_folder, f"{scenario_label}_{new_name}_yearly.png"), dpi=300)
    plt.close(fig)

    yearly_df = pd.DataFrame(yearly_results)
    yearly_df.to_csv(os.path.join(csv_folder, f"{scenario_label}_{new_name}_yearly.csv"), index=False)

    # -------------------------------------------------------------------------
    # 2. D50 active layer all timesteps plot and CSV
    # -------------------------------------------------------------------------
    output_name_d50 = 'D50 active layer [m]'

    if output_name_d50 in data_output:
        my_data_d50 = data_output[output_name_d50]
        n_reach_d50 = my_data_d50.shape[1]
        n_time_d50 = my_data_d50.shape[0]
        FromN_idx_d50 = np.arange(1, n_reach_d50 + 1, 1)

        fig = plt.figure(figsize=(20, 7))
        ax = plt.subplot(111)

        cmap = plt.cm.coolwarm_r

        for t in range(n_time_d50):
            color = cmap(t / (n_time_d50 - 1)) if n_time_d50 > 1 else cmap(0.5)
            ax.plot(FromN_idx_d50, my_data_d50[t, :], color=color, linewidth=1)

        ax.plot(FromN_idx_d50, initial_d50, color='black', linewidth=2.5, linestyle='--', label='Initial D50')

        ax.set_xlabel('Reach index (FromN)', fontsize=14)
        ax.set_ylabel(output_name_d50, fontsize=14)
        ax.set_title(f"{scenario_label} | {output_name_d50} | All timesteps", fontsize=16)
        ax.tick_params(axis='y', which='major', labelsize=11)

        # Show all reach labels
        set_all_reach_ticks(ax, n_reach_d50)

        norm = mpl.colors.Normalize(vmin=1, vmax=n_time_d50)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Timestep', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        ax.legend(fontsize=10)

        fig.tight_layout()
        new_name_d50 = rename_names(output_name_d50)
        fig.savefig(os.path.join(figure_folder, f"{scenario_label}_{new_name_d50}_all_timesteps.png"), dpi=300)
        plt.close(fig)

        d50_all_df = pd.DataFrame(
            my_data_d50,
            columns=[f"reach_{i+1}" for i in range(n_reach_d50)]
        )
        d50_all_df.insert(0, "timestep", np.arange(1, n_time_d50 + 1))
        d50_all_df.to_csv(os.path.join(csv_folder, f"{scenario_label}_{new_name_d50}_all_timesteps.csv"), index=False)


# =============================================================================
# 3. COMBINED PLOT OF ALL SCENARIOS
# =============================================================================

if len(combined_profiles) > 0:
    fig = plt.figure(figsize=(20, 7))
    ax = plt.subplot(111)

    # assume all scenarios have same number of reaches
    n_reach_comb = len(combined_profiles[0][1])
    FromN_idx_comb = np.arange(1, n_reach_comb + 1, 1)

    for scenario_label, vals in combined_profiles:
        ax.plot(FromN_idx_comb, vals, linewidth=2, label=scenario_label)

    if output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
        # add one initial D50 line from first matching river network
        first_river_path = os.path.join(path_river_network, scenarios[0]["river_file"])
        _, initial_d50_first = read_river_network(first_river_path)
        ax.plot(FromN_idx_comb, initial_d50_first, color='black', linestyle='--', linewidth=2.5, label='Initial D50')

    if output_name == 'Delta z [m]' or output_name == 'Sediment budget [m^3]':
        ax.axhline(0, linestyle='--', color='gray')

    ax.set_xlabel('Reach index (FromN)', fontsize=14)
    ax.set_ylabel(output_name, fontsize=14)
    ax.set_title(f"Combined scenarios | {output_name}", fontsize=16)
    ax.legend(fontsize=10)
    ax.tick_params(axis='y', which='major', labelsize=11)

    # Show all reach labels
    set_all_reach_ticks(ax, n_reach_comb)

    fig.tight_layout()

    new_name = rename_names(output_name)
    fig.savefig(os.path.join(figure_folder, f"ALL_SCENARIOS_{new_name}_combined.png"), dpi=300)
    plt.close(fig)

    # Save combined CSV too
    combined_df = pd.DataFrame({"reach": np.arange(1, n_reach_comb + 1)})
    for scenario_label, vals in combined_profiles:
        combined_df[scenario_label] = vals

    if output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
        combined_df["Initial_D50"] = initial_d50_first

    combined_df.to_csv(os.path.join(csv_folder, f"ALL_SCENARIOS_{new_name}_combined.csv"), index=False)


print("Done.")
print("Plots saved in:", figure_folder)
print("CSV files saved in:", csv_folder)