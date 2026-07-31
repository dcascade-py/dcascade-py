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

figure_folder = os.path.join(path, "figures_d50_volume_out")
csv_folder = os.path.join(path, "csv_d50_volume_out")

os.makedirs(figure_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

max_reach = 51
vlines = [8, 9, 12]

output_d50 = "D50 volume out [m]"

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

def safe_name(output_name):
    return (
        output_name.replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("^", "")
        .replace("/", "_")
    )


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
    initial_d50 = river_df_sorted["D50"].to_numpy(dtype=float)

    return river_df_sorted, initial_d50


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


def compute_rosin_d50_t0(river_df_sorted):
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


def set_all_reach_ticks(ax, n_reach):
    ax.set_xticks(np.arange(1, n_reach + 1, 1))
    ax.set_xticklabels(np.arange(1, n_reach + 1, 1), rotation=90, fontsize=8)


def add_vertical_lines(ax):
    for x in vlines:
        ax.axvline(x=x, color="red", linestyle="--", linewidth=1.2, alpha=0.8)


def keep_previous_when_zero(data):
    df = pd.DataFrame(data)

    # Treat 0 as no new generated value
    # Keep the previous valid value until a new non-zero value appears
    df = df.replace(0, np.nan).ffill()

    return df.to_numpy()


# =============================================================================
# MAIN LOOP: D50 VOLUME OUT ALL TIMESTEPS ONLY
# =============================================================================

for sc in scenarios:
    name_simu = sc["name_simu"]
    river_file = sc["river_file"]
    scenario_label = sc["label"]

    print(f"Processing scenario: {scenario_label}")

    river_path = os.path.join(path_river_network, river_file)

    river_df_sorted, initial_d50 = read_river_network_sorted_for_plotting(river_path)
    rosin_d50_t0 = compute_rosin_d50_t0(river_df_sorted)

    result_path = os.path.join(path, name_simu + ".p")

    with open(result_path, "rb") as f:
        data_output = pickle.load(f)

    if output_d50 not in data_output:
        raise ValueError(f"Output '{output_d50}' not found in {name_simu}.p")

    d50_data = np.asarray(data_output[output_d50], dtype=float)

    # Replace 0 values with the previous valid value for plotting and CSV
    d50_data = keep_previous_when_zero(d50_data)

    n_time, n_reach = d50_data.shape

    keep = np.arange(min(max_reach, n_reach))

    d50_data = d50_data[:, keep]
    initial_d50 = initial_d50[keep]
    rosin_d50_t0 = rosin_d50_t0[keep]

    n_reach_plot = len(keep)
    reach_idx = np.arange(1, n_reach_plot + 1)

    fig = plt.figure(figsize=(20, 7))
    ax = plt.subplot(111)

    cmap = plt.cm.coolwarm_r

    for t in range(n_time):
        color = cmap(t / (n_time - 1)) if n_time > 1 else cmap(0.5)
        ax.plot(reach_idx, d50_data[t, :], color=color, linewidth=1)

    ax.plot(
        reach_idx,
        initial_d50,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="Input D50 sorted by FromN"
    )

    ax.plot(
        reach_idx,
        rosin_d50_t0,
        color="orange",
        linewidth=2.0,
        linestyle="-.",
        label="Rosin D50 at t=0"
    )

    add_vertical_lines(ax)

    ax.set_xlabel("Reach order in model output FromN order", fontsize=14)
    ax.set_ylabel(output_d50, fontsize=14)
    ax.set_title(
        f"{scenario_label} | {output_d50} | All timesteps | Reaches 1 to {max_reach}",
        fontsize=16
    )

    ax.set_xlim(1, n_reach_plot)
    set_all_reach_ticks(ax, n_reach_plot)

    norm = mpl.colors.Normalize(vmin=1, vmax=n_time)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Timestep", fontsize=12)

    ax.legend(fontsize=10)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            figure_folder,
            f"{scenario_label}_{safe_name(output_d50)}_all_timesteps_keep_previous_1_51.png"
        ),
        dpi=300
    )

    plt.close(fig)

    d50_all_df = pd.DataFrame(
        d50_data,
        columns=[f"reach_{i+1}" for i in range(n_reach_plot)]
    )

    d50_all_df.insert(0, "timestep", np.arange(1, n_time + 1))

    d50_all_df.to_csv(
        os.path.join(
            csv_folder,
            f"{scenario_label}_{safe_name(output_d50)}_all_timesteps_keep_previous_1_51.csv"
        ),
        index=False
    )


print("Done.")
print("D50 volume out timestep plots saved in:", figure_folder)
print("D50 volume out timestep CSV files saved in:", csv_folder)