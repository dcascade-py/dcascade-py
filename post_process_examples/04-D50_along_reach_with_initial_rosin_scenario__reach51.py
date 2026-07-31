"""
Plot D-CASCADE outputs for multiple scenarios

This version:
1. Plots only reaches 1 to 51
2. Adds vertical lines at reaches 8, 9, 12
3. Creates yearly plots for each scenario
4. Saves yearly CSV output for each scenario
5. Creates D50 all-timestep plot for each scenario
6. Saves D50 all-timestep CSV for each scenario
7. Creates one combined plot with all scenarios together
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib as mpl

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))

if src_path not in sys.path:
    sys.path.append(src_path)

from GSD_curvefit import GSDcurvefit


# =============================================================================
# USER INPUTS
# =============================================================================

path = "..\\cascade_results\\"
path_river_network = "..\\inputs\\Tagliamento_river\\"

figure_folder = os.path.join(path, "figures_multi_scenario")
csv_folder = os.path.join(path, "csv_multi_scenario")

os.makedirs(figure_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

year_0 = 2019

max_reach = 51
vlines = [8, 9, 12]

output_name = 'D50 active layer [m]'

sed_range = (-8, -1)
n_classes = 10

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
    return new_name[:-1]


def read_river_network_sorted_for_plotting(river_path):
    if river_path.lower().endswith(".csv"):
        river_df = pd.read_csv(river_path)
    elif river_path.lower().endswith(".shp"):
        river_df = gpd.read_file(river_path)
    else:
        raise ValueError("River network file must be .csv or .shp")

    for col in ["D50", "FromN"]:
        if col not in river_df.columns:
            raise ValueError(f"Column '{col}' not found in {river_path}")

    river_df_sorted = river_df.sort_values("FromN").reset_index(drop=True)

    initial_d50_sorted = river_df_sorted["D50"].to_numpy(dtype=float)
    fromn_values = river_df_sorted["FromN"].to_numpy()

    if "reach_id" in river_df_sorted.columns:
        reach_id_sorted = river_df_sorted["reach_id"].to_numpy()
    else:
        reach_id_sorted = np.arange(1, len(river_df_sorted) + 1)

    return river_df_sorted, initial_d50_sorted, fromn_values, reach_id_sorted


def compute_dxx_from_fractions(fi_row, dmi_mm, x=50):
    fi_row = np.asarray(fi_row, dtype=float).flatten()
    dmi_mm = np.asarray(dmi_mm, dtype=float).flatten()

    total = np.nansum(fi_row)
    if total <= 0:
        return np.nan

    fi_row = fi_row / total

    sort_idx = np.argsort(dmi_mm)
    d_sorted = dmi_mm[sort_idx]
    f_sorted = fi_row[sort_idx]

    cum_finer = np.cumsum(f_sorted)
    target = x / 100.0

    if target <= cum_finer[0]:
        return d_sorted[0]

    if target >= cum_finer[-1]:
        return d_sorted[-1]

    idx = np.searchsorted(cum_finer, target)

    d1 = d_sorted[idx - 1]
    d2 = d_sorted[idx]
    f1 = cum_finer[idx - 1]
    f2 = cum_finer[idx]

    if np.isclose(f2, f1):
        return d2

    return d1 + (target - f1) / (f2 - f1) * (d2 - d1)


def compute_rosin_d50_t0(river_df_sorted, sed_range=(-8, -1), n_classes=10):
    for col in ["D16", "D50", "D84"]:
        if col not in river_df_sorted.columns:
            raise ValueError(f"Column '{col}' not found in river network file")

    psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)
    dmi_mm = np.squeeze(2 ** (-psi))

    Fi_r, _, _ = GSDcurvefit(
        river_df_sorted["D16"].values,
        river_df_sorted["D50"].values,
        river_df_sorted["D84"].values,
        psi
    )

    d50_rosin_mm = np.array([
        compute_dxx_from_fractions(Fi_r[i, :], dmi_mm, x=50)
        for i in range(Fi_r.shape[0])
    ])

    return d50_rosin_mm / 1000.0


def compute_yearly_profiles(my_data, output_name, year_0):
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

            if len(time_list) == 365:
                sum_all += vals

        elif output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
            vals = np.median(my_data[time_list, :], axis=0)
            lines_to_plot.append((str(year), vals))

        else:
            raise ValueError(f"Unsupported output name: {output_name}")

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
    ax.set_xticks(np.arange(1, n_reach + 1, 1))
    ax.set_xticklabels(np.arange(1, n_reach + 1, 1), rotation=90, fontsize=8)


def add_vertical_lines(ax):
    for x in vlines:
        ax.axvline(x=x, color="red", linestyle="--", linewidth=1.2, alpha=0.8)


# =============================================================================
# MAIN LOOP OVER SCENARIOS
# =============================================================================

combined_profiles = []
combined_reference_info = None

for sc in scenarios:
    name_simu = sc["name_simu"]
    river_file = sc["river_file"]
    scenario_label = sc["label"]

    print(f"Processing scenario: {scenario_label}")

    river_path = os.path.join(path_river_network, river_file)

    river_df_sorted, initial_d50, fromn_values, reach_id_sorted = read_river_network_sorted_for_plotting(river_path)
    rosin_d50_t0 = compute_rosin_d50_t0(river_df_sorted, sed_range=sed_range, n_classes=n_classes)

    result_path = os.path.join(path, name_simu + ".p")

    with open(result_path, "rb") as f:
        data_output = pickle.load(f)

    if output_name not in data_output:
        raise ValueError(f"Output '{output_name}' not found in {name_simu}.p")

    my_data = np.asarray(data_output[output_name])

    n_time = my_data.shape[0]
    n_reach = my_data.shape[1]

    if len(initial_d50) != n_reach:
        raise ValueError(
            f"Length mismatch in {scenario_label}: "
            f"river data has {len(initial_d50)} reaches after FromN sorting, "
            f"but model output has {n_reach} reaches."
        )

    keep = np.arange(min(max_reach, n_reach))

    my_data = my_data[:, keep]
    initial_d50_trim = initial_d50[keep]
    rosin_d50_t0_trim = rosin_d50_t0[keep]
    fromn_values_trim = fromn_values[keep]
    reach_id_sorted_trim = reach_id_sorted[keep]

    n_reach_plot = len(keep)
    reach_idx = np.arange(1, n_reach_plot + 1)

    if combined_reference_info is None:
        combined_reference_info = {
            "initial_d50": initial_d50_trim,
            "rosin_d50_t0": rosin_d50_t0_trim,
            "fromn_values": fromn_values_trim,
            "reach_id_sorted": reach_id_sorted_trim
        }

    # -------------------------------------------------------------------------
    # 1. Yearly plot and CSV
    # -------------------------------------------------------------------------
    yearly_results, lines_to_plot, avg_full_years = compute_yearly_profiles(my_data, output_name, year_0)

    fig = plt.figure(figsize=(20, 7))
    ax = plt.subplot(111)

    colors = iter(plt.cm.viridis(np.linspace(0, 1, len(lines_to_plot))))

    for year_label, vals in lines_to_plot:
        c = next(colors)
        ax.plot(reach_idx, vals, label=year_label, color=c)

    if output_name in ['Volume out [m^3]', 'Volume in [m^3]', 'Transport capacity [m^3]', 'Sediment budget [m^3]']:
        if avg_full_years is not None:
            ax.plot(reach_idx, avg_full_years, linewidth=2.5, color='black', label='Average')
            combined_profiles.append((scenario_label, avg_full_years))

    if output_name in ['D50 active layer [m]', 'D50 volume out [m]']:
        ax.plot(reach_idx, initial_d50_trim, color='black', linewidth=2.5, linestyle='--', label='Input D50 sorted by FromN')
        ax.plot(reach_idx, rosin_d50_t0_trim, color='orange', linewidth=2.0, linestyle='-.', label='Rosin D50 at t=0')

        scenario_profile = np.median(my_data, axis=0)
        combined_profiles.append((scenario_label, scenario_profile))

    if output_name == 'Delta z [m]' or output_name == 'Sediment budget [m^3]':
        ax.axhline(0, linestyle='--', color='gray')

    add_vertical_lines(ax)

    ax.legend(fontsize=10)
    ax.set_xlabel('Reach order in model output FromN order', fontsize=14)
    ax.set_ylabel(output_name, fontsize=14)
    ax.set_title(f"{scenario_label} | {output_name} | Reaches 1 to {max_reach}", fontsize=16)
    ax.tick_params(axis='y', which='major', labelsize=11)

    ax.set_xlim(1, n_reach_plot)
    set_all_reach_ticks(ax, n_reach_plot)

    fig.tight_layout()

    new_name = rename_names(output_name)
    fig.savefig(os.path.join(figure_folder, f"{scenario_label}_{new_name}_yearly_1_51.png"), dpi=300)
    plt.close(fig)

    yearly_df = pd.DataFrame(yearly_results)
    yearly_df.to_csv(os.path.join(csv_folder, f"{scenario_label}_{new_name}_yearly_1_51.csv"), index=False)

    # -------------------------------------------------------------------------
    # 2. D50 active layer all timesteps plot and CSV
    # -------------------------------------------------------------------------
    output_name_d50 = 'D50 active layer [m]'

    if output_name_d50 in data_output:
        my_data_d50 = np.asarray(data_output[output_name_d50])

        n_reach_d50 = my_data_d50.shape[1]
        n_time_d50 = my_data_d50.shape[0]

        if len(initial_d50) != n_reach_d50:
            raise ValueError(
                f"Length mismatch in D50 plot for {scenario_label}: "
                f"river data has {len(initial_d50)} reaches after FromN sorting, "
                f"but D50 output has {n_reach_d50} reaches."
            )

        keep_d50 = np.arange(min(max_reach, n_reach_d50))

        my_data_d50 = my_data_d50[:, keep_d50]
        initial_d50_d50_trim = initial_d50[keep_d50]
        rosin_d50_t0_d50_trim = rosin_d50_t0[keep_d50]
        fromn_values_d50_trim = fromn_values[keep_d50]
        reach_id_sorted_d50_trim = reach_id_sorted[keep_d50]

        n_reach_d50_plot = len(keep_d50)
        reach_idx_d50 = np.arange(1, n_reach_d50_plot + 1)

        fig = plt.figure(figsize=(20, 7))
        ax = plt.subplot(111)

        cmap = plt.cm.coolwarm_r

        for t in range(n_time_d50):
            color = cmap(t / (n_time_d50 - 1)) if n_time_d50 > 1 else cmap(0.5)
            ax.plot(reach_idx_d50, my_data_d50[t, :], color=color, linewidth=1)

        ax.plot(reach_idx_d50, initial_d50_d50_trim, color='black', linewidth=2.5, linestyle='--', label='Input D50 sorted by FromN')
        ax.plot(reach_idx_d50, rosin_d50_t0_d50_trim, color='orange', linewidth=2.0, linestyle='-.', label='Rosin D50 at t=0')

        add_vertical_lines(ax)

        ax.set_xlabel('Reach order in model output FromN order', fontsize=14)
        ax.set_ylabel(output_name_d50, fontsize=14)
        ax.set_title(f"{scenario_label} | {output_name_d50} | All timesteps | Reaches 1 to {max_reach}", fontsize=16)
        ax.tick_params(axis='y', which='major', labelsize=11)

        ax.set_xlim(1, n_reach_d50_plot)
        set_all_reach_ticks(ax, n_reach_d50_plot)

        norm = mpl.colors.Normalize(vmin=1, vmax=n_time_d50)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Timestep', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        ax.legend(fontsize=10)

        fig.tight_layout()

        new_name_d50 = rename_names(output_name_d50)
        fig.savefig(os.path.join(figure_folder, f"{scenario_label}_{new_name_d50}_all_timesteps_1_51.png"), dpi=300)
        plt.close(fig)

        d50_all_df = pd.DataFrame(
            my_data_d50,
            columns=[f"reach_{i+1}" for i in range(n_reach_d50_plot)]
        )
        d50_all_df.insert(0, "timestep", np.arange(1, n_time_d50 + 1))
        d50_all_df.to_csv(
            os.path.join(csv_folder, f"{scenario_label}_{new_name_d50}_all_timesteps_1_51.csv"),
            index=False
        )

        d50_compare_df = pd.DataFrame({
            "plot_reach_index": np.arange(1, n_reach_d50_plot + 1),
            "FromN": fromn_values_d50_trim,
            "reach_id_sorted_by_FromN": reach_id_sorted_d50_trim,
            "Input_D50_m": initial_d50_d50_trim,
            "Rosin_D50_t0_m": rosin_d50_t0_d50_trim,
            "Model_D50AL_t0_m": my_data_d50[0, :]
        })

        d50_compare_df.to_csv(
            os.path.join(csv_folder, f"{scenario_label}_{new_name_d50}_t0_comparison_1_51.csv"),
            index=False
        )


# =============================================================================
# 3. COMBINED PLOT OF ALL SCENARIOS
# =============================================================================

if len(combined_profiles) > 0:
    fig = plt.figure(figsize=(20, 7))
    ax = plt.subplot(111)

    n_reach_comb = len(combined_profiles[0][1])
    reach_idx_comb = np.arange(1, n_reach_comb + 1)

    for scenario_label, vals in combined_profiles:
        ax.plot(reach_idx_comb, vals, linewidth=2, label=scenario_label)

    if output_name in ['D50 active layer [m]', 'D50 volume out [m]'] and combined_reference_info is not None:
        ax.plot(
            reach_idx_comb,
            combined_reference_info["initial_d50"],
            color='black',
            linestyle='--',
            linewidth=2.5,
            label='Input D50 sorted by FromN'
        )

        ax.plot(
            reach_idx_comb,
            combined_reference_info["rosin_d50_t0"],
            color='orange',
            linestyle='-.',
            linewidth=2.0,
            label='Rosin D50 at t=0'
        )

    if output_name == 'Delta z [m]' or output_name == 'Sediment budget [m^3]':
        ax.axhline(0, linestyle='--', color='gray')

    add_vertical_lines(ax)

    ax.set_xlabel('Reach order in model output FromN order', fontsize=14)
    ax.set_ylabel(output_name, fontsize=14)
    ax.set_title(f"Combined scenarios | {output_name} | Reaches 1 to {max_reach}", fontsize=16)
    ax.legend(fontsize=10)
    ax.tick_params(axis='y', which='major', labelsize=11)

    ax.set_xlim(1, n_reach_comb)
    set_all_reach_ticks(ax, n_reach_comb)

    fig.tight_layout()

    new_name = rename_names(output_name)
    fig.savefig(os.path.join(figure_folder, f"ALL_SCENARIOS_{new_name}_combined_1_51.png"), dpi=300)
    plt.close(fig)

    combined_df = pd.DataFrame({"plot_reach_index": np.arange(1, n_reach_comb + 1)})

    for scenario_label, vals in combined_profiles:
        combined_df[scenario_label] = vals

    if output_name in ['D50 active layer [m]', 'D50 volume out [m]'] and combined_reference_info is not None:
        combined_df["FromN"] = combined_reference_info["fromn_values"]
        combined_df["reach_id_sorted_by_FromN"] = combined_reference_info["reach_id_sorted"]
        combined_df["Input_D50"] = combined_reference_info["initial_d50"]
        combined_df["Rosin_D50_t0"] = combined_reference_info["rosin_d50_t0"]

    combined_df.to_csv(
        os.path.join(csv_folder, f"ALL_SCENARIOS_{new_name}_combined_1_51.csv"),
        index=False
    )


print("Done.")
print("Plots saved in:", figure_folder)
print("CSV files saved in:", csv_folder)