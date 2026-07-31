"""
Plot separately for gorge reaches FromN 8, 9, and 12.

Important:
    D50 active layer and D50 volume out are extracted using FromN:
        FromN = 8, 9, 12

    Discharge is extracted using the corresponding reach_id:
        FromN 8  -> reachID 10
        FromN 9  -> reachID 11
        FromN 12 -> reachID 14

For each gorge reach:
    1 PNG with 8 subplots:
        4 rows = 4 scenarios
        2 columns:
            left  = D50 active layer vs Discharge
            right = D50 volume out vs Discharge

Outputs:
    1. 3 figures, one for each gorge reach
    2. CSV files for each scenario
"""

# =========================================================
# LIBRARIES
# =========================================================
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt


# =========================================================
# PATHS
# =========================================================
path = "..\\cascade_results\\"

path_river_network = "..\\inputs\\Tagliamento_river\\"
name_river_network_bf = "Reach_data_tag_bf.csv"
name_river_network_50bf = "Reach_data_tag_50bf.csv"

path_q = "..\\inputs\\Tagliamento_river\\"
name_q_bf = "Tagliamento_Qdaily_4y_2021_2024.csv"

# If you do not have a separate discharge file for 0.5BF,
# keep the same file name as the BF case.
name_q_50bf = "Tagliamento_Qdaily_4y_2021_2024.csv"

figure_folder = os.path.join(path, "figures_D50_vs_Q_gorge_reaches")
csv_folder = os.path.join(path, "csv_D50_vs_Q_gorge_reaches")

os.makedirs(figure_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

# D50 gorges are selected using FromN
gorge_fromn = [8, 9, 12]

# But discharge columns must use reachID, not FromN
gorge_q_reach_id = {
    8: 10,
    9: 11,
    12: 14
}

key_d50_active = "D50 active layer [m]"

possible_keys_d50_volume_out = [
    "D50 volume out [m]",
    "D50 volume out",
    "D50 out [m]",
    "D50 out",
    "D50 transported [m]",
    "D50 leaving reach [m]"
]


# =========================================================
# CASES
# =========================================================
cases = {
    "BF-0.2": {
        "pickle": os.path.join(path, "Tagliamento_bf_al002.p"),
        "river_csv": os.path.join(path_river_network, name_river_network_bf),
        "q_csv": os.path.join(path_q, name_q_bf)
    },
    "BF-0.5": {
        "pickle": os.path.join(path, "Tagliamento_bf_al005.p"),
        "river_csv": os.path.join(path_river_network, name_river_network_bf),
        "q_csv": os.path.join(path_q, name_q_bf)
    },
    "0.5BF-0.2": {
        "pickle": os.path.join(path, "Tagliamento_50bf_al002.p"),
        "river_csv": os.path.join(path_river_network, name_river_network_50bf),
        "q_csv": os.path.join(path_q, name_q_50bf)
    },
    "0.5BF-0.5": {
        "pickle": os.path.join(path, "Tagliamento_50bf_al005.p"),
        "river_csv": os.path.join(path_river_network, name_river_network_50bf),
        "q_csv": os.path.join(path_q, name_q_50bf)
    }
}


# =========================================================
# FUNCTIONS
# =========================================================
def find_column(df, possible_names):
    """
    Find a column name ignoring spaces and capitalization.
    """
    clean_cols = {col.strip().lower(): col for col in df.columns}

    for name in possible_names:
        key = name.strip().lower()
        if key in clean_cols:
            return clean_cols[key]

    raise ValueError(f"None of these columns were found: {possible_names}")


def find_q_column(q_df, reach_number):
    """
    Find the discharge column for a given reachID.

    Expected examples:
        reachID=10
        ReachID=10
        reach_id=10
        Reach=10
        10
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


def find_pickle_key(data_output, possible_keys):
    """
    Find a pickle key ignoring capitalization.
    """
    clean_keys = {k.strip().lower(): k for k in data_output.keys()}

    for key in possible_keys:
        clean_key = key.strip().lower()
        if clean_key in clean_keys:
            return clean_keys[clean_key]

    print("Available keys in pickle:")
    for k in data_output.keys():
        print(k)

    raise KeyError(f"None of these keys were found: {possible_keys}")


def get_gorge_indices(river_file, gorge_fromn):
    """
    Get indices of FromN 8, 9, and 12 after sorting the river file by FromN.
    These indices are used for extracting D50 data from the DCASCADE output.
    """
    if river_file.lower().endswith(".csv"):
        river_df = pd.read_csv(river_file)
    elif river_file.lower().endswith(".shp"):
        river_df = gpd.read_file(river_file)
    else:
        raise ValueError("River network file must be .csv or .shp")

    river_df.columns = [c.strip() for c in river_df.columns]

    fromn_col = find_column(
        river_df,
        ["FromN", "from_n", "From_N", "fromn"]
    )

    river_df_sorted = river_df.sort_values(fromn_col).reset_index(drop=True)
    fromn_sorted = river_df_sorted[fromn_col].astype(int).to_numpy()

    gorge_indices = []

    for fn in gorge_fromn:
        idx = np.where(fromn_sorted == fn)[0]

        if len(idx) == 0:
            raise ValueError(f"FromN {fn} not found in river network.")

        gorge_indices.append(idx[0])

    return np.array(gorge_indices)


def extract_reach_series(arr, gorge_indices):
    """
    Extract gorge reach columns from a 2D output array.

    Expected:
        arr shape = [time, reach]

    If the array is instead:
        arr shape = [reach, time]

    the function transposes it automatically.
    """
    arr = np.array(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    max_gorge_idx = int(np.max(gorge_indices))

    # likely [time, reach]
    if arr.shape[1] > max_gorge_idx:
        return arr[:, gorge_indices]

    # maybe [reach, time]
    elif arr.shape[0] > max_gorge_idx:
        return arr[gorge_indices, :].T

    else:
        raise ValueError(f"Could not detect reach axis in array shape {arr.shape}")


def get_discharge(q_file, gorge_fromn, gorge_q_reach_id, n_time):
    """
    Read discharge time series for the gorge reaches.

    D50 is extracted using FromN:
        8, 9, 12

    Discharge is extracted using reachID:
        FromN 8  -> reachID 10
        FromN 9  -> reachID 11
        FromN 12 -> reachID 14
    """
    q_df = pd.read_csv(q_file)
    q_df.columns = [c.strip() for c in q_df.columns]

    q_gorges = np.zeros((n_time, len(gorge_fromn)))

    for i, fn in enumerate(gorge_fromn):

        q_reach_id = gorge_q_reach_id[fn]

        q_col = find_q_column(q_df, q_reach_id)
        q_values = q_df[q_col].to_numpy(dtype=float)

        if len(q_values) > n_time:
            print(
                f"Warning: Discharge for FromN {fn}, reachID {q_reach_id}, "
                f"has {len(q_values)} rows, but model output has {n_time}. "
                f"Trimming discharge."
            )
            q_values = q_values[:n_time]

        elif len(q_values) < n_time:
            raise ValueError(
                f"Discharge length for FromN {fn}, reachID {q_reach_id}, "
                f"is {len(q_values)}, but model output length is {n_time}."
            )

        q_gorges[:, i] = q_values

        print(
            f"Discharge for FromN {fn} extracted from reachID {q_reach_id}, "
            f"column: {q_col}"
        )

    return q_gorges


# =========================================================
# READ ALL CASES
# =========================================================
all_data = {}

for case_name, files in cases.items():

    print("--------------------------------------------------")
    print(f"Reading case: {case_name}")

    with open(files["pickle"], "rb") as f:
        data_output = pickle.load(f)

    if key_d50_active not in data_output:
        print("Available keys in pickle:")
        for key in data_output.keys():
            print(key)

        raise KeyError(f"{key_d50_active} not found in {case_name}")

    key_d50_volume_out = find_pickle_key(
        data_output,
        possible_keys_d50_volume_out
    )

    print(f"D50 active key used: {key_d50_active}")
    print(f"D50 volume out key used: {key_d50_volume_out}")

    d50_active_all = np.array(data_output[key_d50_active])
    d50_volume_out_all = np.array(data_output[key_d50_volume_out])

    gorge_indices = get_gorge_indices(
        files["river_csv"],
        gorge_fromn
    )

    d50_active_gorges = extract_reach_series(
        d50_active_all,
        gorge_indices
    )

    d50_volume_out_gorges = extract_reach_series(
        d50_volume_out_all,
        gorge_indices
    )

    n_time = d50_active_gorges.shape[0]
    time_steps = np.arange(1, n_time + 1)

    q_gorges = get_discharge(
        files["q_csv"],
        gorge_fromn,
        gorge_q_reach_id,
        n_time
    )

    all_data[case_name] = {
        "time_steps": time_steps,
        "d50_active": d50_active_gorges,
        "d50_volume_out": d50_volume_out_gorges,
        "q": q_gorges
    }

    # -----------------------------------------------------
    # SAVE CSV FOR EACH CASE
    # -----------------------------------------------------
    df_case = pd.DataFrame({
        "timestep_day": time_steps
    })

    for i, fn in enumerate(gorge_fromn):

        q_reach_id = gorge_q_reach_id[fn]

        df_case[f"D50_active_FromN_{fn}_m"] = d50_active_gorges[:, i]
        df_case[f"D50_volume_out_FromN_{fn}_m"] = d50_volume_out_gorges[:, i]
        df_case[f"Q_FromN_{fn}_reachID_{q_reach_id}_m3s"] = q_gorges[:, i]

    safe_case_name = case_name.replace(".", "p")
    csv_path = os.path.join(
        csv_folder,
        f"D50_vs_Q_{safe_case_name}.csv"
    )

    df_case.to_csv(csv_path, index=False)

    print(f"CSV saved: {csv_path}")


# =========================================================
# PLOT SEPARATELY FOR EACH REACH
# 1 PNG per reach
# 8 subplots per PNG = 4 scenarios x 2 plot types
# =========================================================
for reach_idx, fn in enumerate(gorge_fromn):

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(14, 18),
        sharey=False
    )

    case_names = list(cases.keys())

    for row_idx, case_name in enumerate(case_names):

        data_case = all_data[case_name]
        q_reach_id = gorge_q_reach_id[fn]

        # -------------------------------------------------
        # LEFT COLUMN:
        # D50 active layer vs discharge
        # -------------------------------------------------
        ax_left = axes[row_idx, 0]

        ax_left.scatter(
            data_case["d50_active"][:, reach_idx],
            data_case["q"][:, reach_idx],
            s=14,
            alpha=0.65
        )

        ax_left.set_title(
            f"{case_name} | FromN {fn}, Q reachID {q_reach_id} | D50 active vs Q",
            fontsize=10
        )

        ax_left.set_xlabel("D50 active layer [m]", fontsize=10)
        ax_left.set_ylabel("Q [m³/s]", fontsize=10)
        ax_left.grid(True, alpha=0.5)

        # -------------------------------------------------
        # RIGHT COLUMN:
        # D50 volume out vs discharge
        # -------------------------------------------------
        ax_right = axes[row_idx, 1]

        ax_right.scatter(
            data_case["d50_volume_out"][:, reach_idx],
            data_case["q"][:, reach_idx],
            s=14,
            alpha=0.65
        )

        ax_right.set_title(
            f"{case_name} | FromN {fn}, Q reachID {q_reach_id} | D50 volume out vs Q",
            fontsize=10
        )

        ax_right.set_xlabel("D50 volume out [m]", fontsize=10)
        ax_right.set_ylabel("Q [m³/s]", fontsize=10)
        ax_right.grid(True, alpha=0.5)

    fig.suptitle(
        f"Gorge FromN {fn}: D50 vs Discharge, Q from reachID {gorge_q_reach_id[fn]}",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    figure_path = os.path.join(
        figure_folder,
        f"FromN_{fn}_QreachID_{gorge_q_reach_id[fn]}_8plots_4scenarios_D50_vs_Q.png"
    )

    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    print(f"Figure saved: {figure_path}")


# =========================================================
# DONE
# =========================================================
print("--------------------------------------------------")
print("Done.")
print("All figures saved in:", figure_folder)
print("All CSV files saved in:", csv_folder)