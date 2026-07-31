# -*- coding: utf-8 -*-
"""
Plot D-CASCADE results for the Tagliamento River.

This version:
1. Reads the Tagliamento reach network from CSV or Excel.
2. Separates the main Tagliamento channel and tributaries using the "Trib" column.
3. Reads D-CASCADE standard and extended output pickle files.
4. Plots initial active-layer fractions from:
       data_output_ext["fi_al"]
5. Also plots outgoing sediment volume fractions by grain size if available.
6. Creates fallback plots from the normal output file.
7. Prints available keys and array shapes for checking.

Important:
The fi_al key must exist in the extended output file.

Expected fi_al shape:
    time x reach x grain-size class

For the initial active-layer plot, this script uses:
    data_output_ext["fi_al"][0, :, :]
"""

# ------------------------------------------------------------
# 1. LIBRARIES
# ------------------------------------------------------------

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ------------------------------------------------------------
# 2. USER SETTINGS
# ------------------------------------------------------------

# Main project folder
BASE_DIR = Path(r"D:\0Padova Study\sem 4\GitHub\dcascade-py")

# Input folder for Tagliamento
INPUT_DIR = BASE_DIR / "inputs" / "Tagliamento_river"

# Results folder
RESULTS_DIR = BASE_DIR / "cascade_results"

# Reach network file
REACH_FILE = INPUT_DIR / "Reach_data_tag_bf.csv"

# Discharge file
Q_FILE = INPUT_DIR / "Tagliamento_Qdaily_4y_2021_2024.csv"

# Simulation name.
# This must match the name_output used in your simulation script.
SIMULATION_NAME = "Tagliamento_bf"

# D-CASCADE standard output file
OUTPUT_FILE = RESULTS_DIR / f"{SIMULATION_NAME}.p"

# D-CASCADE extended output file
EXT_OUTPUT_FILE = RESULTS_DIR / f"{SIMULATION_NAME}_ext.p"

# Figures will be saved here
FIGURE_FOLDER = RESULTS_DIR / f"figures_{SIMULATION_NAME}"
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

# Last main-channel FromN to plot.
# FromN 52 to 58 are ignored because those reaches do not have slope data.
MAX_MAIN_FROMN = 51


# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def read_reach_table(file_path):
    """
    Read the reach data table from CSV or Excel.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Reach file not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported reach file type: {file_path.suffix}")

    df.columns = df.columns.str.strip()

    return df


def read_q_table(file_path):
    """
    Read discharge table from CSV or Excel.
    This is not required for the plots, but useful to check that the file loads.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Warning: Q file not found: {file_path}")
        return None

    if file_path.suffix.lower() == ".csv":
        q = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        q = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported Q file type: {file_path.suffix}")

    q.columns = q.columns.str.strip()

    return q


def load_pickle(file_path):
    """
    Load a D-CASCADE pickle output file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def get_psi_from_output(data_output, default_sed_range=(-8, -1), default_n_classes=10):
    """
    Get sediment classes in phi scale from the D-CASCADE output if available.
    Otherwise, use Tagliamento default values.
    """

    try:
        psi = data_output["Simulation parameters"]["psi"]
        psi = np.asarray(psi, dtype=float)
    except Exception:
        psi = np.linspace(
            default_sed_range[0],
            default_sed_range[1],
            num=default_n_classes,
            endpoint=True
        ).astype(float)

    return psi


def phi_to_diameter_mm(psi):
    """
    Convert Krumbein phi values to sediment diameter in mm.

    D-CASCADE convention:
        d = 2^(-phi)
    """

    return 2 ** (-psi)


def safe_fraction(values, axis=1):
    """
    Convert volumes to fractions safely.
    If the total is zero, the fraction is set to zero instead of NaN or infinity.
    """

    values = np.asarray(values, dtype=float)
    totals = np.sum(values, axis=axis, keepdims=True)

    return np.divide(
        values,
        totals,
        out=np.zeros_like(values, dtype=float),
        where=totals != 0
    )


def prepare_reach_groups(reach_df):
    """
    Prepare main-channel and tributary reach tables.

    Expected columns:
    - reach_id
    - FromN
    - ToN
    - Trib
    - Name, optional
    """

    required_cols = ["reach_id", "FromN", "ToN", "Trib"]
    missing = [c for c in required_cols if c not in reach_df.columns]

    if missing:
        raise ValueError(
            "Missing required columns in reach table: "
            + ", ".join(missing)
        )

    reach_df = reach_df.copy()

    for col in ["reach_id", "FromN", "ToN"]:
        reach_df[col] = pd.to_numeric(reach_df[col], errors="coerce")

    if reach_df[["reach_id", "FromN", "ToN"]].isna().any().any():
        raise ValueError("reach_id, FromN, or ToN contains non-numeric values.")

    # Main river: in your Tagliamento file, Trib = 'n'
    main_df = reach_df[
        reach_df["Trib"].astype(str).str.lower().str.strip() == "n"
    ].copy()

    # Ignore main-channel reaches without slope data.
    main_df = main_df[main_df["FromN"] <= MAX_MAIN_FROMN].copy()

    # Tributaries: everything else
    trib_df = reach_df[
        reach_df["Trib"].astype(str).str.lower().str.strip() != "n"
    ].copy()

    # Sort main river from upstream to downstream
    main_df = main_df.sort_values("FromN").reset_index(drop=True)

    # Sort tributaries by where they join the main river
    trib_df = trib_df.sort_values("ToN").reset_index(drop=True)

    # Array indices in Python are zero-based, while reach_id is one-based
    main_df["array_idx"] = main_df["reach_id"].astype(int) - 1
    trib_df["array_idx"] = trib_df["reach_id"].astype(int) - 1

    # X-axis position for main river bars
    main_df["x_plot"] = main_df["FromN"].astype(int)

    # Tributary names
    if "Name" in trib_df.columns:
        trib_df["name_clean"] = trib_df["Name"].fillna("tributary").astype(str)
    else:
        trib_df["name_clean"] = (
            "tributary_" + trib_df["reach_id"].astype(int).astype(str)
        )

    trib_df["trib_number"] = np.arange(1, len(trib_df) + 1)

    trib_df["plot_name"] = (
        trib_df["trib_number"].astype(str)
        + " ("
        + trib_df["name_clean"]
        + ")"
    )

    return main_df, trib_df


def validate_fraction_matrix(fraction_matrix, main_df, trib_df, label):
    """
    Check that the fraction matrix can be plotted with the reach table indices.
    """

    fraction_matrix = np.asarray(fraction_matrix, dtype=float)

    if fraction_matrix.ndim != 2:
        raise ValueError(
            f"{label} must be 2D with shape reach x grain-size class. "
            f"Current shape: {fraction_matrix.shape}"
        )

    max_needed_idx = max(
        int(main_df["array_idx"].max()),
        int(trib_df["array_idx"].max())
    )

    if max_needed_idx >= fraction_matrix.shape[0]:
        raise ValueError(
            f"{label} has only {fraction_matrix.shape[0]} reaches, "
            f"but the reach table needs index {max_needed_idx}."
        )

    row_sums = np.sum(fraction_matrix, axis=1)

    print(f"\nChecking {label}:")
    print(f"  Shape: {fraction_matrix.shape}")
    print(f"  Minimum row sum: {np.nanmin(row_sums):.6f}")
    print(f"  Maximum row sum: {np.nanmax(row_sums):.6f}")

    return fraction_matrix


def add_tributary_lines(ax, trib_df):
    """
    Add vertical lines showing where tributaries enter the main river.
    """

    ymin, ymax = ax.get_ylim()
    y_text = ymax * 0.55 if ymax != 0 else 0.8

    x_min, x_max = ax.get_xlim()

    for _, row in trib_df.iterrows():
        x = row["ToN"]

        if x < x_min or x > x_max:
            continue

        name = row["plot_name"]

        ax.axvline(
            x=x,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=1000
        )

        ax.text(
            x,
            y_text,
            name,
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            color="dimgray",
            fontsize=9,
            fontweight="bold",
            zorder=1001,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=1.5
            )
        )


def style_main_axis(ax, main_df, ylabel):
    """
    Format the main Tagliamento axis.
    """

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Main Tagliamento reaches, plotted by FromN", fontsize=10)

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    x_values = main_df["x_plot"].values
    labels = main_df["FromN"].astype(int).astype(str).values

    ax.set_xticks(x_values)

    ax.set_xticklabels(
        labels,
        fontsize=8,
        rotation=90,
        horizontalalignment="center",
        verticalalignment="top"
    )

    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.tick_params(axis="x", which="major", labelsize=8)

    ax.margins(x=0.01)


def plot_grain_fraction_two_panels(
    fraction_matrix,
    main_df,
    trib_df,
    dmi_mm,
    title,
    ylabel,
    output_path
):
    """
    Make a stacked bar plot with:
    - left panel: main Tagliamento channel
    - right panel: tributaries

    fraction_matrix must be:
        n_reaches x n_classes
    """

    fraction_matrix = np.asarray(fraction_matrix, dtype=float)

    n_reaches, n_classes = fraction_matrix.shape

    if len(dmi_mm) != n_classes:
        raise ValueError(
            f"dmi_mm has {len(dmi_mm)} classes, "
            f"but fraction matrix has {n_classes} classes."
        )

    colors = [plt.cm.jet(i / (n_classes - 1)) for i in range(n_classes)]

    fig = plt.figure()
    ax_main = plt.subplot2grid(shape=(1, 4), loc=(0, 0), colspan=3)
    ax_trib = plt.subplot2grid(shape=(1, 4), loc=(0, 3), colspan=1)

    # -----------------------------
    # Main Tagliamento plot
    # -----------------------------
    main_idx = main_df["array_idx"].values
    main_x = main_df["x_plot"].values

    bottom = np.zeros(len(main_df))

    for i in range(n_classes):
        values = fraction_matrix[main_idx, i]

        ax_main.bar(
            main_x,
            values,
            bottom=bottom,
            color=colors[i],
            label=f"d = {dmi_mm[i]:.3g} mm"
        )

        bottom += values

    add_tributary_lines(ax_main, trib_df)
    style_main_axis(ax_main, main_df, ylabel)
    ax_main.set_title(title, fontsize=12)
    ax_main.set_ylim(0, 1.05)

    # -----------------------------
    # Tributary plot
    # -----------------------------
    trib_idx = trib_df["array_idx"].values
    trib_x = np.arange(len(trib_df))

    bottom = np.zeros(len(trib_df))

    for i in range(n_classes):
        values = fraction_matrix[trib_idx, i]

        ax_trib.bar(
            trib_x,
            values,
            bottom=bottom,
            color=colors[i],
            label=f"d = {dmi_mm[i]:.3g} mm"
        )

        bottom += values

    ax_trib.set_xticks(trib_x)

    ax_trib.set_xticklabels(
        trib_df["plot_name"],
        fontsize=9,
        rotation=90,
        horizontalalignment="center",
        verticalalignment="top"
    )

    ax_trib.tick_params(axis="y", which="major", labelsize=10)
    ax_trib.tick_params(axis="x", which="major", labelsize=8)
    ax_trib.set_title("Tributaries", fontsize=12)
    ax_trib.set_ylim(0, 1.05)

    ax_trib.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.set_size_inches(1700 / fig.dpi, 650 / fig.dpi)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_reach_series_two_panels(
    values,
    main_df,
    trib_df,
    title,
    ylabel,
    output_path
):
    """
    Plot one value per reach for the main channel and tributaries.
    """

    values = np.asarray(values, dtype=float)

    fig = plt.figure()
    ax_main = plt.subplot2grid(shape=(1, 4), loc=(0, 0), colspan=3)
    ax_trib = plt.subplot2grid(shape=(1, 4), loc=(0, 3), colspan=1)

    main_idx = main_df["array_idx"].values
    main_x = main_df["x_plot"].values

    ax_main.bar(main_x, values[main_idx])
    add_tributary_lines(ax_main, trib_df)
    style_main_axis(ax_main, main_df, ylabel)
    ax_main.set_title(title, fontsize=12)

    trib_idx = trib_df["array_idx"].values
    trib_x = np.arange(len(trib_df))

    ax_trib.bar(trib_x, values[trib_idx])
    ax_trib.set_xticks(trib_x)

    ax_trib.set_xticklabels(
        trib_df["plot_name"],
        fontsize=9,
        rotation=90,
        horizontalalignment="center",
        verticalalignment="top"
    )

    ax_trib.tick_params(axis="y", which="major", labelsize=10)
    ax_trib.tick_params(axis="x", which="major", labelsize=8)
    ax_trib.set_title("Tributaries", fontsize=12)

    fig.set_size_inches(1700 / fig.dpi, 650 / fig.dpi)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 4. LOAD DATA
# ------------------------------------------------------------

print("Loading input and output files...")

reach_df = read_reach_table(REACH_FILE)
q_df = read_q_table(Q_FILE)

data_output = load_pickle(OUTPUT_FILE)

if EXT_OUTPUT_FILE.exists():
    data_output_ext = load_pickle(EXT_OUTPUT_FILE)
    print(f"Loaded extended output: {EXT_OUTPUT_FILE}")
else:
    data_output_ext = None
    print(f"Extended output not found: {EXT_OUTPUT_FILE}")
    print("The script will create fallback plots from the standard output file.")

main_df, trib_df = prepare_reach_groups(reach_df)

psi = get_psi_from_output(data_output)
dmi_mm = phi_to_diameter_mm(psi)
n_classes = len(dmi_mm)

print("\nInput summary:")
print("Reach table:", reach_df.shape)
print("Main-channel reaches plotted:", len(main_df))
print(f"Main-channel FromN plotted up to: {MAX_MAIN_FROMN}")
print("Tributaries:", len(trib_df))
print("Sediment classes:", n_classes)

if q_df is not None:
    print("Q table:", q_df.shape)

print("\nOutput files:")
print(f"Standard output: {OUTPUT_FILE}")
print(f"Extended output: {EXT_OUTPUT_FILE}")


# ------------------------------------------------------------
# 5. PLOTS FROM EXTENDED OUTPUT, IF AVAILABLE
# ------------------------------------------------------------

if data_output_ext is not None:

    # --------------------------------------------------------
    # 5.1 Initial active-layer sediment fractions from fi_al
    # --------------------------------------------------------

    if "fi_al" in data_output_ext:
        fi_al = np.asarray(data_output_ext["fi_al"], dtype=float)

        print("\nFound fi_al in extended output.")
        print(f"fi_al shape: {fi_al.shape}")

        if fi_al.ndim != 3:
            raise ValueError(
                "fi_al must have shape time x reach x grain-size class. "
                f"Current shape is {fi_al.shape}."
            )

        if fi_al.shape[2] != n_classes:
            raise ValueError(
                f"fi_al has {fi_al.shape[2]} grain-size classes, "
                f"but psi gives {n_classes} classes."
            )

        # Initial active-layer fractions at time step 0
        fi_r_init = fi_al[0, :, :]

        fi_r_init = validate_fraction_matrix(
            fraction_matrix=fi_r_init,
            main_df=main_df,
            trib_df=trib_df,
            label="fi_al[0, :, :]"
        )

        plot_grain_fraction_two_panels(
            fraction_matrix=fi_r_init,
            main_df=main_df,
            trib_df=trib_df,
            dmi_mm=dmi_mm,
            title="Initial active-layer sediment fractions",
            ylabel="Volume fraction per grain size",
            output_path=FIGURE_FOLDER / "Tagliamento_Fi_r_init_t0.png"
        )

    else:
        print("\nKey not found in extended output: fi_al")
        print("Run the Tagliamento simulation script that saves:")
        print('    extended_output["fi_al"] = Fi_r[np.newaxis, :, :]')

    # --------------------------------------------------------
    # 5.2 Fraction of outgoing sediment volume per grain size
    # --------------------------------------------------------

    key_vout_grain = "Volume out per grain sizes [m^3]"

    if key_vout_grain in data_output_ext:
        qbi_mob = np.asarray(data_output_ext[key_vout_grain], dtype=float)

        print(f"\nFound {key_vout_grain} in extended output.")
        print(f"{key_vout_grain} shape: {qbi_mob.shape}")

        if qbi_mob.ndim != 3:
            raise ValueError(
                f"{key_vout_grain} must have shape time x reach x grain-size class. "
                f"Current shape is {qbi_mob.shape}."
            )

        vout_tot = np.sum(qbi_mob, axis=0)
        fir_vout = safe_fraction(vout_tot, axis=1)

        fir_vout = validate_fraction_matrix(
            fraction_matrix=fir_vout,
            main_df=main_df,
            trib_df=trib_df,
            label="Outgoing sediment volume fractions"
        )

        plot_grain_fraction_two_panels(
            fraction_matrix=fir_vout,
            main_df=main_df,
            trib_df=trib_df,
            dmi_mm=dmi_mm,
            title="Outgoing sediment volume fractions by grain size",
            ylabel="Volume fraction per grain size",
            output_path=FIGURE_FOLDER / "Tagliamento_Fi_r_vout.png"
        )

    else:
        print(f"\nKey not found in extended output: {key_vout_grain}")
        print("This does not stop the fi_al plot. It only means the outgoing grain-size plot cannot be made.")


# ------------------------------------------------------------
# 6. FALLBACK PLOTS FROM NORMAL OUTPUT
# ------------------------------------------------------------

# ------------------------------------------------------------
# 6.1 Total sediment volume out per reach
# ------------------------------------------------------------

if "Volume out [m^3]" in data_output:
    volume_out_total = np.sum(data_output["Volume out [m^3]"], axis=0)

    plot_reach_series_two_panels(
        values=volume_out_total,
        main_df=main_df,
        trib_df=trib_df,
        title="Total sediment volume out per reach",
        ylabel="Total volume out [m³]",
        output_path=FIGURE_FOLDER / "Tagliamento_total_volume_out.png"
    )
else:
    print("\nKey not found in standard output: Volume out [m^3]")


# ------------------------------------------------------------
# 6.2 Mean D50 of outgoing sediment per reach
# ------------------------------------------------------------

if "D50 volume out [m]" in data_output:
    d50_vout_mean_m = np.nanmean(data_output["D50 volume out [m]"], axis=0)

    plot_reach_series_two_panels(
        values=d50_vout_mean_m,
        main_df=main_df,
        trib_df=trib_df,
        title="Mean D50 of outgoing sediment",
        ylabel="Mean D50 volume out [m]",
        output_path=FIGURE_FOLDER / "Tagliamento_mean_D50_volume_out.png"
    )
else:
    print("\nKey not found in standard output: D50 volume out [m]")


# ------------------------------------------------------------
# 6.3 Mean D50 of active layer per reach
# ------------------------------------------------------------

if "D50 active layer [m]" in data_output:
    d50_al_mean_m = np.nanmean(data_output["D50 active layer [m]"], axis=0)

    plot_reach_series_two_panels(
        values=d50_al_mean_m,
        main_df=main_df,
        trib_df=trib_df,
        title="Mean D50 of active layer",
        ylabel="Mean D50 active layer [m]",
        output_path=FIGURE_FOLDER / "Tagliamento_mean_D50_active_layer.png"
    )
else:
    print("\nKey not found in standard output: D50 active layer [m]")


# ------------------------------------------------------------
# 6.4 Total sediment budget per reach
# ------------------------------------------------------------

if "Sediment budget [m^3]" in data_output:
    sediment_budget_total = np.sum(data_output["Sediment budget [m^3]"], axis=0)

    plot_reach_series_two_panels(
        values=sediment_budget_total,
        main_df=main_df,
        trib_df=trib_df,
        title="Total sediment budget per reach",
        ylabel="Sediment budget [m³]",
        output_path=FIGURE_FOLDER / "Tagliamento_total_sediment_budget.png"
    )
else:
    print("\nKey not found in standard output: Sediment budget [m^3]")


# ------------------------------------------------------------
# 7. PRINT AVAILABLE OUTPUT KEYS
# ------------------------------------------------------------

print("\nStandard output keys:")
for key in data_output.keys():
    value = data_output[key]
    print(f"  {key}: {getattr(value, 'shape', type(value))}")

if data_output_ext is not None:
    print("\nExtended output keys:")
    for key in data_output_ext.keys():
        value = data_output_ext[key]
        print(f"  {key}: {getattr(value, 'shape', type(value))}")

print("\nFinished.")