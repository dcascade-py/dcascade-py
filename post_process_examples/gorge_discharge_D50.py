"""
Plot D50 active layer through time for gorge reaches FromN 8, 9, and 12
with discharge shown as bar graph in the same composite plot.

Important:
    D50 is extracted using FromN:
        FromN = 8, 9, 12

    Discharge is extracted using the corresponding reach_id:
        FromN 8  -> reachID 10
        FromN 9  -> reachID 11
        FromN 12 -> reachID 14

Outputs:
1. Composite plot with 3 subplots:
   - D50 active layer as line
   - Initial D50 as dashed line
   - Discharge as bar graph on secondary y-axis

2. CSV with:
   - timestep
   - D50 for FromN 8, 9, 12
   - initial D50 for FromN 8, 9, 12
   - discharge using reachID 10, 11, 14
"""

# =========================================================
# LIBRARIES
# =========================================================
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd


# =========================================================
# PATHS
# =========================================================
path = "..\\cascade_results\\"
name_simu = "Tagliamento_bf_al005"

path_river_network = "..\\inputs\\Tagliamento_river\\"
name_river_network = "Reach_data_tag_bf.csv"

path_q = "..\\inputs\\Tagliamento_river\\"
name_q = "Tagliamento_Qdaily_4y_2021_2024.csv"

figure_folder = os.path.join(path, "figures_gorges_D50")
csv_folder = os.path.join(path, "csv_gorges_D50")

if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)


# =========================================================
# SETTINGS
# =========================================================
output_name_d50 = "D50 active layer [m]"

# Gorges are selected using FromN
gorge_fromn = [8, 9, 12]

# Discharge must be extracted using reachID, not FromN
gorge_q_reach_id = {
    8: 10,
    9: 11,
    12: 14
}


# =========================================================
# FUNCTIONS
# =========================================================
def find_column(df, possible_names):
    """
    Finds a column ignoring spaces and capitalization.
    """
    clean_cols = {col.strip().lower(): col for col in df.columns}

    for name in possible_names:
        key = name.strip().lower()
        if key in clean_cols:
            return clean_cols[key]

    raise ValueError(f"None of these columns were found: {possible_names}")


def find_q_column(q_df, reach_number):
    """
    Finds the discharge column for a reachID.

    Expected examples:
    - reachID=10
    - ReachID=10
    - reach_id=10
    - Reach=10
    - 10
    """
    q_df.columns = [c.strip() for c in q_df.columns]

    possible_cols = [
        f"reachID={reach_number}",
        f"ReachID={reach_number}",
        f"reach_id={reach_number}",
        f"Reach={reach_number}",
        str(reach_number)
    ]

    for col in possible_cols:
        if col in q_df.columns:
            return col

    print("Available discharge columns:")
    for col in q_df.columns:
        print(col)

    raise ValueError(f"Discharge column not found for reachID {reach_number}")


# =========================================================
# READ RIVER NETWORK
# =========================================================
river_path = os.path.join(path_river_network, name_river_network)

if river_path.lower().endswith(".csv"):
    river_df = pd.read_csv(river_path)
elif river_path.lower().endswith(".shp"):
    river_df = gpd.read_file(river_path)
else:
    raise ValueError("River network file must be .csv or .shp")

river_df.columns = [c.strip() for c in river_df.columns]

fromn_col = find_column(
    river_df,
    ["FromN", "from_n", "From_N", "fromn"]
)

d50_col = find_column(
    river_df,
    ["D50", "d50"]
)

# Sort by FromN so it matches DCASCADE output ordering
river_df_sorted = river_df.sort_values(fromn_col).reset_index(drop=True)

fromn_sorted = river_df_sorted[fromn_col].astype(int).to_numpy()
initial_d50_sorted = river_df_sorted[d50_col].to_numpy(dtype=float)

# Find indices for gorge FromN 8, 9, and 12
gorge_indices = []

for fn in gorge_fromn:
    idx = np.where(fromn_sorted == fn)[0]

    if len(idx) == 0:
        raise ValueError(f"FromN {fn} not found in river network.")

    gorge_indices.append(idx[0])

gorge_indices = np.array(gorge_indices)
initial_d50_gorges = initial_d50_sorted[gorge_indices]


# =========================================================
# READ PICKLE OUTPUT
# =========================================================
with open(os.path.join(path, name_simu + ".p"), "rb") as f:
    data_output = pickle.load(f)

if output_name_d50 not in data_output:
    print("Available keys in pickle:")
    for key in data_output.keys():
        print(key)

    raise KeyError(f"{output_name_d50} not found in pickle output.")

my_data_d50 = np.array(data_output[output_name_d50])

# Extract only gorge columns using FromN indices
d50_gorges = my_data_d50[:, gorge_indices]

n_time = d50_gorges.shape[0]
time_steps = np.arange(1, n_time + 1)


# =========================================================
# READ DISCHARGE CSV
# =========================================================
q_path = os.path.join(path_q, name_q)

q_df = pd.read_csv(q_path)
q_df.columns = [c.strip() for c in q_df.columns]

# Extract discharge for the correct reachID values:
# FromN 8  -> reachID 10
# FromN 9  -> reachID 11
# FromN 12 -> reachID 14
q_gorges = np.zeros((n_time, len(gorge_fromn)))

for i, fn in enumerate(gorge_fromn):

    q_reach_id = gorge_q_reach_id[fn]

    q_col = find_q_column(q_df, q_reach_id)
    q_values = q_df[q_col].to_numpy(dtype=float)

    # Match discharge length with D50 output length
    if len(q_values) > n_time:
        print(
            f"Warning: Discharge for FromN {fn}, reachID {q_reach_id}, "
            f"has {len(q_values)} rows, but D50 has {n_time}. "
            f"Trimming discharge to first {n_time} rows."
        )
        q_values = q_values[:n_time]

    elif len(q_values) < n_time:
        raise ValueError(
            f"Discharge length for FromN {fn}, reachID {q_reach_id}, "
            f"is {len(q_values)}, but D50 length is {n_time}. "
            f"Discharge file is too short."
        )

    q_gorges[:, i] = q_values

    print(
        f"Discharge for FromN {fn} extracted from reachID {q_reach_id}, "
        f"column: {q_col}"
    )


# =========================================================
# SAVE CSV
# =========================================================
d50_q_csv = pd.DataFrame({
    "timestep_day": time_steps
})

for i, fn in enumerate(gorge_fromn):

    q_reach_id = gorge_q_reach_id[fn]

    d50_q_csv[f"D50_FromN_{fn}_m"] = d50_gorges[:, i]
    d50_q_csv[f"Initial_D50_FromN_{fn}_m"] = initial_d50_gorges[i]
    d50_q_csv[f"Q_FromN_{fn}_reachID_{q_reach_id}_m3s"] = q_gorges[:, i]

csv_path = os.path.join(
    csv_folder,
    "D50_and_discharge_timeseries_gorges_FromN_8_9_12_QreachID_10_11_14.csv"
)

d50_q_csv.to_csv(csv_path, index=False)


# =========================================================
# COMPOSITE PLOT
# D50 LINE + DISCHARGE BAR GRAPH
# =========================================================
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(15, 11),
    sharex=True
)

for ax, fn, i in zip(axes, gorge_fromn, range(len(gorge_fromn))):

    q_reach_id = gorge_q_reach_id[fn]

    # Secondary axis for discharge bars
    ax_q = ax.twinx()

    # Discharge bar graph
    ax_q.bar(
        time_steps,
        q_gorges[:, i],
        width=1.0,
        alpha=0.25,
        label=f"Q reachID {q_reach_id}"
    )

    ax_q.set_ylabel("Q [m³/s]", fontsize=11)
    ax_q.tick_params(axis="y", labelsize=10)

    # D50 active layer line
    ax.plot(
        time_steps,
        d50_gorges[:, i],
        linewidth=1.3,
        label=f"D50 FromN {fn}"
    )

    # Initial D50 line
    ax.hlines(
        initial_d50_gorges[i],
        xmin=1,
        xmax=n_time,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Initial D50"
    )

    ax.set_title(
        f"Gorge FromN {fn}, discharge from reachID {q_reach_id}",
        fontsize=14
    )

    ax.set_ylabel("D50 [m]", fontsize=12)
    ax.grid(True, alpha=0.5)

    # Combine legends from both axes
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_q.get_legend_handles_labels()

    ax.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        fontsize=9,
        loc="upper right"
    )

axes[-1].set_xlabel("Timestep [day]", fontsize=12)

fig.suptitle(
    "D50 Active Layer and Discharge Through Time, Gorge Reaches FromN 8, 9, and 12",
    fontsize=18
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

figure_path = os.path.join(
    figure_folder,
    "D50_active_layer_with_discharge_bar_FromN_8_9_12_QreachID_10_11_14.png"
)

fig.savefig(figure_path, dpi=300)
plt.close(fig)


# =========================================================
# DONE
# =========================================================
print("Done.")
print("Composite plot saved in:", figure_path)
print("CSV saved in:", csv_path)