# -*- coding: utf-8 -*-
"""
Plot and export D-CASCADE results for four Tagliamento River scenarios.

Scenarios included:
1. Tagliamento_50bf_al005
2. Tagliamento_50bf_al002
3. Tagliamento_bf_al005
4. Tagliamento_bf_al002

For each scenario, the script:
1. Reads the correct river morphology reach file:
   - Reach_data_tag_50bf.csv for 50bf scenarios
   - Reach_data_tag_bf.csv for bf scenarios
2. Reads the normal D-CASCADE output pickle file.
3. Reads the extended D-CASCADE output pickle file if available.
4. Creates separate figure folders for each scenario.
5. Creates separate CSV folders for each scenario.
6. Produces plots for:
   - initial active-layer sediment fractions
   - outgoing sediment volume fractions by grain size
   - total sediment volume out per reach
   - mean D50 of outgoing sediment
   - mean D50 of active layer
   - total sediment budget per reach
7. Excludes FromN 52 to 58 only from the figures.
8. Keeps FromN 52 to 58 in all CSV outputs.
9. Labels tributaries using their actual reach_id, for example 59 (Rovadia), not 1 (Rovadia).

Author: adapted for Tagliamento
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

BASE_DIR = Path(r"D:\0Padova Study\sem 4\GitHub\dcascade-py")

INPUT_DIR = BASE_DIR / "inputs" / "Tagliamento_river"

RESULTS_DIR = BASE_DIR / "cascade_results"

Q_FILE = INPUT_DIR / "Tagliamento_Qdaily_4y_2021_2024.csv"

# FromN 52 to 58 are excluded only from figures.
# They remain included in CSV files.
MAX_MAIN_FROMN_FOR_FIGURES = 51

SCENARIOS = [
    {
        "scenario_name": "Tagliamento_50bf_al005",
        "reach_file": INPUT_DIR / "Reach_data_tag_50bf.csv",
        "output_file": RESULTS_DIR / "Tagliamento_50bf_al005.p",
        "ext_output_file": RESULTS_DIR / "Tagliamento_50bf_al005_ext.p",
    },
    {
        "scenario_name": "Tagliamento_50bf_al002",
        "reach_file": INPUT_DIR / "Reach_data_tag_50bf.csv",
        "output_file": RESULTS_DIR / "Tagliamento_50bf_al002.p",
        "ext_output_file": RESULTS_DIR / "Tagliamento_50bf_al002_ext.p",
    },
    {
        "scenario_name": "Tagliamento_bf_al005",
        "reach_file": INPUT_DIR / "Reach_data_tag_bf.csv",
        "output_file": RESULTS_DIR / "Tagliamento_bf_al005.p",
        "ext_output_file": RESULTS_DIR / "Tagliamento_bf_al005_ext.p",
    },
    {
        "scenario_name": "Tagliamento_bf_al002",
        "reach_file": INPUT_DIR / "Reach_data_tag_bf.csv",
        "output_file": RESULTS_DIR / "Tagliamento_bf_al002.p",
        "ext_output_file": RESULTS_DIR / "Tagliamento_bf_al002_ext.p",
    },
]


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
        raise FileNotFoundError(f"Output file not found: {file_path}")

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

    D-CASCADE convention used here:
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


def check_required_reach_columns(reach_df):
    """
    Check that the required reach columns exist.
    """
    required_cols = ["reach_id", "FromN", "ToN", "Trib"]
    missing = [c for c in required_cols if c not in reach_df.columns]

    if missing:
        raise ValueError(
            "Missing required columns in reach table: "
            + ", ".join(missing)
        )


def clean_reach_table(reach_df):
    """
    Clean and prepare reach table.
    """
    reach_df = reach_df.copy()

    check_required_reach_columns(reach_df)

    for col in ["reach_id", "FromN", "ToN"]:
        reach_df[col] = pd.to_numeric(reach_df[col], errors="coerce")

    reach_df = reach_df.dropna(subset=["reach_id", "FromN", "ToN", "Trib"]).copy()

    return reach_df


def add_tributary_names(trib_df):
    """
    Add clean tributary names and plot labels.

    Tributary label uses the actual reach_id.
    Example:
        59 (Rovadia)
        60 (Poschiedea)

    It does not use artificial numbering from 1 to 10.
    """
    trib_df = trib_df.copy()

    if "Name" in trib_df.columns:
        trib_df["name_clean"] = trib_df["Name"].fillna("tributary").astype(str)
    else:
        trib_df["name_clean"] = (
            "tributary_" + trib_df["reach_id"].astype(int).astype(str)
        )

    trib_df["tributary_label"] = trib_df["reach_id"].astype(int)

    trib_df["plot_name"] = (
        trib_df["tributary_label"].astype(str)
        + " ("
        + trib_df["name_clean"]
        + ")"
    )

    return trib_df


def prepare_reach_groups_for_figures(reach_df):
    """
    Prepare main-channel and tributary reach tables for plotting.

    For figures:
    - main-channel reaches are kept only up to FromN 51
    - tributaries are kept unchanged
    """
    reach_df = clean_reach_table(reach_df)

    main_df_fig = reach_df[
        reach_df["Trib"].astype(str).str.lower().str.strip() == "n"
    ].copy()

    main_df_fig = main_df_fig[
        main_df_fig["FromN"] <= MAX_MAIN_FROMN_FOR_FIGURES
    ].copy()

    trib_df_fig = reach_df[
        reach_df["Trib"].astype(str).str.lower().str.strip() != "n"
    ].copy()

    main_df_fig = main_df_fig.sort_values("FromN").reset_index(drop=True)
    trib_df_fig = trib_df_fig.sort_values("ToN").reset_index(drop=True)

    main_df_fig["array_idx"] = main_df_fig["reach_id"].astype(int) - 1
    trib_df_fig["array_idx"] = trib_df_fig["reach_id"].astype(int) - 1

    main_df_fig["x_plot"] = main_df_fig["FromN"].astype(int)

    trib_df_fig = add_tributary_names(trib_df_fig)

    return main_df_fig, trib_df_fig


def prepare_reach_groups_for_csv(reach_df):
    """
    Prepare main-channel and tributary reach tables for CSV export.

    For CSV:
    - all main-channel reaches are kept, including FromN 52 to 58
    - all tributaries are kept
    """
    reach_df = clean_reach_table(reach_df)

    main_df_csv = reach_df[
        reach_df["Trib"].astype(str).str.lower().str.strip() == "n"
    ].copy()

    trib_df_csv = reach_df[
        reach_df["Trib"].astype(str).str.lower().str.strip() != "n"
    ].copy()

    main_df_csv = main_df_csv.sort_values("FromN").reset_index(drop=True)
    trib_df_csv = trib_df_csv.sort_values("ToN").reset_index(drop=True)

    main_df_csv["array_idx"] = main_df_csv["reach_id"].astype(int) - 1
    trib_df_csv["array_idx"] = trib_df_csv["reach_id"].astype(int) - 1

    main_df_csv["x_plot"] = main_df_csv["FromN"].astype(int)

    trib_df_csv = add_tributary_names(trib_df_csv)

    return main_df_csv, trib_df_csv


def get_export_columns(df):
    """
    Select useful metadata columns for CSV export.
    """
    preferred_cols = [
        "reach_id",
        "FromN",
        "ToN",
        "Trib",
        "Name",
        "name_clean",
        "tributary_label",
        "plot_name",
        "array_idx",
        "x_plot"
    ]

    cols = [c for c in preferred_cols if c in df.columns]

    return cols


def export_reach_series_csv(values, main_df_csv, trib_df_csv, value_column, output_path):
    """
    Export one value per reach to CSV.

    CSV includes:
    - all main-channel reaches, including FromN 52 to 58
    - all tributaries
    """
    values = np.asarray(values, dtype=float)

    main_export = main_df_csv[get_export_columns(main_df_csv)].copy()
    main_export["panel"] = "main_channel"
    main_export[value_column] = values[main_df_csv["array_idx"].values]

    trib_export = trib_df_csv[get_export_columns(trib_df_csv)].copy()
    trib_export["panel"] = "tributary"
    trib_export[value_column] = values[trib_df_csv["array_idx"].values]

    export_df = pd.concat([main_export, trib_export], ignore_index=True)

    export_df.to_csv(output_path, index=False)
    print(f"Saved CSV: {output_path}")


def export_grain_fraction_csv(
    fraction_matrix,
    main_df_csv,
    trib_df_csv,
    dmi_mm,
    output_path,
    value_column="fraction"
):
    """
    Export grain-size data to CSV in long format.

    CSV includes:
    - all main-channel reaches, including FromN 52 to 58
    - all tributaries
    """
    fraction_matrix = np.asarray(fraction_matrix, dtype=float)
    dmi_mm = np.asarray(dmi_mm, dtype=float)

    rows = []

    for panel_name, df in [
        ("main_channel", main_df_csv),
        ("tributary", trib_df_csv)
    ]:
        metadata_cols = get_export_columns(df)

        for _, reach_row in df.iterrows():
            array_idx = int(reach_row["array_idx"])

            base_info = {
                "panel": panel_name
            }

            for col in metadata_cols:
                base_info[col] = reach_row[col]

            for class_idx, diameter_mm in enumerate(dmi_mm):
                row = base_info.copy()
                row["grain_class"] = class_idx + 1
                row["diameter_mm"] = diameter_mm
                row[value_column] = fraction_matrix[array_idx, class_idx]
                rows.append(row)

    export_df = pd.DataFrame(rows)

    export_df.to_csv(output_path, index=False)
    print(f"Saved CSV: {output_path}")


def add_tributary_lines(ax, trib_df_fig):
    """
    Add vertical lines showing where tributaries enter the main river.
    Tributary labels use actual reach_id, for example 59 (Rovadia).
    """
    ymin, ymax = ax.get_ylim()
    y_text = ymax * 0.55 if ymax != 0 else 0.8

    x_min, x_max = ax.get_xlim()

    for _, row in trib_df_fig.iterrows():
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


def style_main_axis(ax, main_df_fig, ylabel):
    """
    Format the main Tagliamento axis.
    """
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Main Tagliamento reaches, plotted by FromN", fontsize=10)

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    x_values = main_df_fig["x_plot"].values
    labels = main_df_fig["FromN"].astype(int).astype(str).values

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
    main_df_fig,
    trib_df_fig,
    dmi_mm,
    title,
    ylabel,
    output_path
):
    """
    Make a stacked bar plot with:
    - left panel: main Tagliamento channel up to FromN 51
    - right panel: tributaries
    """
    n_reaches, n_classes = fraction_matrix.shape
    colors = [plt.cm.jet(i / (n_classes - 1)) for i in range(n_classes)]

    fig = plt.figure()
    ax_main = plt.subplot2grid(shape=(1, 4), loc=(0, 0), colspan=3)
    ax_trib = plt.subplot2grid(shape=(1, 4), loc=(0, 3), colspan=1)

    main_idx = main_df_fig["array_idx"].values
    main_x = main_df_fig["x_plot"].values

    bottom = np.zeros(len(main_df_fig))

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

    add_tributary_lines(ax_main, trib_df_fig)
    style_main_axis(ax_main, main_df_fig, ylabel)
    ax_main.set_title(title, fontsize=12)

    trib_idx = trib_df_fig["array_idx"].values
    trib_x = np.arange(len(trib_df_fig))

    bottom = np.zeros(len(trib_df_fig))

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
        trib_df_fig["plot_name"],
        fontsize=9,
        rotation=90,
        horizontalalignment="center",
        verticalalignment="top"
    )

    ax_trib.tick_params(axis="y", which="major", labelsize=10)
    ax_trib.tick_params(axis="x", which="major", labelsize=8)
    ax_trib.set_title("Tributaries", fontsize=12)

    ax_trib.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.set_size_inches(1700 / fig.dpi, 650 / fig.dpi)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def plot_reach_series_two_panels(
    values,
    main_df_fig,
    trib_df_fig,
    title,
    ylabel,
    output_path
):
    """
    Plot one value per reach for:
    - main channel up to FromN 51
    - tributaries
    """
    values = np.asarray(values, dtype=float)

    fig = plt.figure()
    ax_main = plt.subplot2grid(shape=(1, 4), loc=(0, 0), colspan=3)
    ax_trib = plt.subplot2grid(shape=(1, 4), loc=(0, 3), colspan=1)

    main_idx = main_df_fig["array_idx"].values
    main_x = main_df_fig["x_plot"].values

    ax_main.bar(main_x, values[main_idx])
    add_tributary_lines(ax_main, trib_df_fig)
    style_main_axis(ax_main, main_df_fig, ylabel)
    ax_main.set_title(title, fontsize=12)

    trib_idx = trib_df_fig["array_idx"].values
    trib_x = np.arange(len(trib_df_fig))

    ax_trib.bar(trib_x, values[trib_idx])
    ax_trib.set_xticks(trib_x)

    ax_trib.set_xticklabels(
        trib_df_fig["plot_name"],
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

    print(f"Saved figure: {output_path}")


def process_scenario(scenario):
    """
    Process one Tagliamento scenario:
    - load reach file
    - load normal output
    - load extended output if available
    - create figures
    - export CSVs
    """
    scenario_name = scenario["scenario_name"]
    reach_file = scenario["reach_file"]
    output_file = scenario["output_file"]
    ext_output_file = scenario["ext_output_file"]

    print("\n" + "=" * 70)
    print(f"Processing scenario: {scenario_name}")
    print("=" * 70)

    figure_folder = RESULTS_DIR / f"figures_{scenario_name}"
    csv_folder = RESULTS_DIR / f"csv_{scenario_name}"

    figure_folder.mkdir(parents=True, exist_ok=True)
    csv_folder.mkdir(parents=True, exist_ok=True)

    print(f"Reach file: {reach_file}")
    print(f"Standard output file: {output_file}")
    print(f"Extended output file: {ext_output_file}")

    reach_df = read_reach_table(reach_file)

    data_output = load_pickle(output_file)

    if ext_output_file.exists():
        data_output_ext = load_pickle(ext_output_file)
        print(f"Loaded extended output: {ext_output_file}")
    else:
        data_output_ext = None
        print(f"Extended output not found: {ext_output_file}")
        print("The script will create fallback plots from the standard output file.")

    main_df_fig, trib_df_fig = prepare_reach_groups_for_figures(reach_df)
    main_df_csv, trib_df_csv = prepare_reach_groups_for_csv(reach_df)

    psi = get_psi_from_output(data_output)
    dmi_mm = phi_to_diameter_mm(psi)
    n_classes = len(dmi_mm)

    print("Reach table:", reach_df.shape)
    print("Main-channel reaches plotted:", len(main_df_fig))
    print(f"Main-channel FromN plotted up to: {MAX_MAIN_FROMN_FOR_FIGURES}")
    print("Main-channel reaches exported to CSV:", len(main_df_csv))
    print("Tributaries plotted:", len(trib_df_fig))
    print("Tributaries exported to CSV:", len(trib_df_csv))
    print("Sediment classes:", n_classes)

    if data_output_ext is not None:

        if "fi_al" in data_output_ext:
            fi_r_init = data_output_ext["fi_al"][0, :, :]

            plot_grain_fraction_two_panels(
                fraction_matrix=fi_r_init,
                main_df_fig=main_df_fig,
                trib_df_fig=trib_df_fig,
                dmi_mm=dmi_mm,
                title=f"{scenario_name}: initial active-layer sediment fractions",
                ylabel="Volume fraction per grain size",
                output_path=figure_folder / f"{scenario_name}_Fi_r_init_t0.png"
            )

            export_grain_fraction_csv(
                fraction_matrix=fi_r_init,
                main_df_csv=main_df_csv,
                trib_df_csv=trib_df_csv,
                dmi_mm=dmi_mm,
                output_path=csv_folder / f"{scenario_name}_Fi_r_init_t0.csv",
                value_column="initial_active_layer_fraction"
            )

        else:
            print("Key not found in extended output: fi_al")

        key_vout_grain = "Volume out per grain sizes [m^3]"

        if key_vout_grain in data_output_ext:
            qbi_mob = data_output_ext[key_vout_grain]

            vout_tot = np.sum(qbi_mob, axis=0)

            fir_vout = safe_fraction(vout_tot, axis=1)

            plot_grain_fraction_two_panels(
                fraction_matrix=fir_vout,
                main_df_fig=main_df_fig,
                trib_df_fig=trib_df_fig,
                dmi_mm=dmi_mm,
                title=f"{scenario_name}: outgoing sediment volume fractions by grain size",
                ylabel="Volume fraction per grain size",
                output_path=figure_folder / f"{scenario_name}_Fi_r_vout.png"
            )

            export_grain_fraction_csv(
                fraction_matrix=fir_vout,
                main_df_csv=main_df_csv,
                trib_df_csv=trib_df_csv,
                dmi_mm=dmi_mm,
                output_path=csv_folder / f"{scenario_name}_Fi_r_vout.csv",
                value_column="outgoing_sediment_volume_fraction"
            )

            export_grain_fraction_csv(
                fraction_matrix=vout_tot,
                main_df_csv=main_df_csv,
                trib_df_csv=trib_df_csv,
                dmi_mm=dmi_mm,
                output_path=csv_folder / f"{scenario_name}_volume_out_per_grain_size_total.csv",
                value_column="volume_out_per_grain_size_total_m3"
            )

        else:
            print(f"Key not found in extended output: {key_vout_grain}")

    if "Volume out [m^3]" in data_output:
        volume_out_total = np.sum(data_output["Volume out [m^3]"], axis=0)

        plot_reach_series_two_panels(
            values=volume_out_total,
            main_df_fig=main_df_fig,
            trib_df_fig=trib_df_fig,
            title=f"{scenario_name}: total sediment volume out per reach",
            ylabel="Total volume out [m³]",
            output_path=figure_folder / f"{scenario_name}_total_volume_out.png"
        )

        export_reach_series_csv(
            values=volume_out_total,
            main_df_csv=main_df_csv,
            trib_df_csv=trib_df_csv,
            value_column="total_volume_out_m3",
            output_path=csv_folder / f"{scenario_name}_total_volume_out.csv"
        )
    else:
        print("Key not found in standard output: Volume out [m^3]")

    if "D50 volume out [m]" in data_output:
        d50_vout_mean_m = np.nanmean(data_output["D50 volume out [m]"], axis=0)

        plot_reach_series_two_panels(
            values=d50_vout_mean_m,
            main_df_fig=main_df_fig,
            trib_df_fig=trib_df_fig,
            title=f"{scenario_name}: mean D50 of outgoing sediment",
            ylabel="Mean D50 volume out [m]",
            output_path=figure_folder / f"{scenario_name}_mean_D50_volume_out.png"
        )

        export_reach_series_csv(
            values=d50_vout_mean_m,
            main_df_csv=main_df_csv,
            trib_df_csv=trib_df_csv,
            value_column="mean_D50_volume_out_m",
            output_path=csv_folder / f"{scenario_name}_mean_D50_volume_out.csv"
        )
    else:
        print("Key not found in standard output: D50 volume out [m]")

    if "D50 active layer [m]" in data_output:
        d50_al_mean_m = np.nanmean(data_output["D50 active layer [m]"], axis=0)

        plot_reach_series_two_panels(
            values=d50_al_mean_m,
            main_df_fig=main_df_fig,
            trib_df_fig=trib_df_fig,
            title=f"{scenario_name}: mean D50 of active layer",
            ylabel="Mean D50 active layer [m]",
            output_path=figure_folder / f"{scenario_name}_mean_D50_active_layer.png"
        )

        export_reach_series_csv(
            values=d50_al_mean_m,
            main_df_csv=main_df_csv,
            trib_df_csv=trib_df_csv,
            value_column="mean_D50_active_layer_m",
            output_path=csv_folder / f"{scenario_name}_mean_D50_active_layer.csv"
        )
    else:
        print("Key not found in standard output: D50 active layer [m]")

    if "Sediment budget [m^3]" in data_output:
        sediment_budget_total = np.sum(data_output["Sediment budget [m^3]"], axis=0)

        plot_reach_series_two_panels(
            values=sediment_budget_total,
            main_df_fig=main_df_fig,
            trib_df_fig=trib_df_fig,
            title=f"{scenario_name}: total sediment budget per reach",
            ylabel="Sediment budget [m³]",
            output_path=figure_folder / f"{scenario_name}_total_sediment_budget.png"
        )

        export_reach_series_csv(
            values=sediment_budget_total,
            main_df_csv=main_df_csv,
            trib_df_csv=trib_df_csv,
            value_column="total_sediment_budget_m3",
            output_path=csv_folder / f"{scenario_name}_total_sediment_budget.csv"
        )
    else:
        print("Key not found in standard output: Sediment budget [m^3]")

    print("\nStandard output keys:")
    for key in data_output.keys():
        value = data_output[key]
        print(f"  {key}: {getattr(value, 'shape', type(value))}")

    if data_output_ext is not None:
        print("\nExtended output keys:")
        for key in data_output_ext.keys():
            value = data_output_ext[key]
            print(f"  {key}: {getattr(value, 'shape', type(value))}")

    print(f"\nFinished scenario: {scenario_name}")
    print(f"Figures saved in: {figure_folder}")
    print(f"CSV files saved in: {csv_folder}")


# ------------------------------------------------------------
# 4. MAIN SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":

    q_df = read_q_table(Q_FILE)

    if q_df is not None:
        print("Q table:", q_df.shape)

    for scenario in SCENARIOS:
        process_scenario(scenario)

    print("\n" + "=" * 70)
    print("All scenarios finished.")
    print("=" * 70)