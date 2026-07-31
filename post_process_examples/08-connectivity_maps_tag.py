# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:40:14 2025

@author: FPitscheider

Plot connectivity map

Show map with sediment path-length, per each time step

This version selects:
1. a mainstem reach range
2. automatically includes tributaries connected to that range
"""

import os
import sys
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch, FancyArrow
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString

# Add source (src) folder in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing import extract_Q, read_network
from reach_data import ReachData

# ---------------------Path to the extended pickle output
path = "..\\cascade_results\\"
name_simu = 'Tagliamento_bf'
name_simu_ext = 'Tagliamento_bf_ext'

# ---------------------Path to the input river network (.shp) or (.csv)
path_river_network = "..\\inputs\\Tagliamento_river\\shp\\"
name_river_network = "joinedSHP.shp"

# ---------------------Path to the discharge file
path_Q = "..\\inputs\\Tagliamento_river\\"
name_q = 'Tagliamento_Qdaily_4y_2021_2024.csv'

# ---------------------Folder to store the plots
figure_folder = path + 'figures_connectivity_maps\\'

if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

# --------------------Define time range
start_timestep = 540
end_timestep = 550

start_date = np.datetime64('2019-01-01')

# --------------------Mainstem selection
# Example: if you choose 1 to 10, all tributaries connected to this
# mainstem segment will also be included automatically
mainstem_min_reach = 1
mainstem_max_reach = 10

# --------------------Known mainstem maximum reach for this network
# Based on your inspected network:
# mainstem = 1 to 58
# tributaries = 59+
mainstem_max_id = 58

# --------------------Optional minimum volume threshold for plotting arrows
min_volume_to_plot = 0

# --------------------Gap to adjust for plotting
gap_plot = 15000


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def get_linestring_start_coords(geom):
    if isinstance(geom, LineString):
        return geom.coords[0]
    elif isinstance(geom, MultiLineString):
        first_line = list(geom.geoms)[0]
        return first_line.coords[0]
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")


def get_linestring_end_coords(geom):
    if isinstance(geom, LineString):
        return geom.coords[-1]
    elif isinstance(geom, MultiLineString):
        last_line = list(geom.geoms)[-1]
        return last_line.coords[-1]
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")


def detect_outlet_reach(reach_data):
    """
    Detect outlet reach for a network where:
    - FromN = current reach ID
    - ToN   = downstream reach ID

    For this inspected dataset, if there is no null ToN,
    use reach 58 as the outlet.
    """
    outlet_rows = reach_data[reach_data["ToN"].isna()]

    if len(outlet_rows) == 1:
        outlet_idx = outlet_rows.index[0]
        outlet_reach = int(outlet_rows.iloc[0]["FromN"])
        return outlet_idx, outlet_reach

    elif len(outlet_rows) > 1:
        raise ValueError(
            f"Multiple reaches have null ToN. Cannot uniquely detect outlet. "
            f"Outlet candidates: {outlet_rows['FromN'].tolist()}"
        )

    from_vals = set(reach_data["FromN"].dropna().astype(int))
    to_vals = set(reach_data["ToN"].dropna().astype(int))

    terminal_candidates = list(from_vals - to_vals)

    print("No null ToN found.")
    print("Candidate reaches in FromN but not in ToN:", sorted(terminal_candidates))

    if 58 in from_vals:
        outlet_rows = reach_data[reach_data["FromN"].astype(int) == 58]
        if len(outlet_rows) == 1:
            outlet_idx = outlet_rows.index[0]
            outlet_reach = 58
            print("Using reach 58 as outlet based on network inspection.")
            return outlet_idx, outlet_reach

    raise ValueError(
        "Could not detect outlet reach automatically. "
        "Please inspect ToN values for the downstream-most reach."
    )


def build_selected_reaches_with_tributaries(reach_data, mainstem_min_reach, mainstem_max_reach, mainstem_max_id):
    """
    Select:
    1. all mainstem reaches in the chosen range
    2. all tributaries connected to those reaches
    3. all upstream tributary branches connected to those tributaries

    Logic:
    - mainstem reaches are assumed to be 1..mainstem_max_id
    - tributaries are reaches > mainstem_max_id
    - ToN gives the downstream receiving reach
    """

    from_vals = reach_data["FromN"].astype(int).tolist()
    to_vals = reach_data["ToN"].tolist()

    # Build reverse connectivity: downstream reach -> list of upstream reaches
    upstream_map = {}
    for f, t in zip(from_vals, to_vals):
        if pd.notna(t):
            t = int(t)
            upstream_map.setdefault(t, []).append(int(f))

    # Seed with mainstem range
    selected = set(range(mainstem_min_reach, mainstem_max_reach + 1))

    # Start traversal from selected mainstem reaches
    stack = list(selected)

    while stack:
        current = stack.pop()

        upstream_reaches = upstream_map.get(current, [])

        for up in upstream_reaches:
            # If this upstream reach is a mainstem reach outside selected range,
            # do not include it
            if up <= mainstem_max_id and not (mainstem_min_reach <= up <= mainstem_max_reach):
                continue

            if up not in selected:
                selected.add(up)
                stack.append(up)

    return sorted(selected)


# -------------------------------------------------------------------
# Read data
# -------------------------------------------------------------------
reach_data = read_network(path_river_network + name_river_network)

if "geometry" not in reach_data.columns:
    raise ValueError(f"'geometry' column not found. Available columns: {list(reach_data.columns)}")

# Convert IDs to numeric
reach_data["FromN"] = pd.to_numeric(reach_data["FromN"], errors="coerce")
reach_data["ToN"] = pd.to_numeric(reach_data["ToN"], errors="coerce")

if reach_data["FromN"].isna().any():
    raise ValueError("Some FromN values could not be converted to numeric.")

Q = extract_Q(path_Q + name_q)

# Sort reach_data according to FromN and organize Q accordingly
sorted_indices = reach_data.sort_values(by="FromN").index
Q_new = np.zeros(Q.shape)

for i, idx in enumerate(sorted_indices):
    Q_new[:, i] = Q.iloc[:, idx]

Q = Q_new
reach_data = reach_data.sort_values(by="FromN", ignore_index=True)

print("Available FromN values:", sorted(reach_data["FromN"].dropna().astype(int).unique()))
print("Available ToN values:", sorted(reach_data["ToN"].dropna().astype(int).unique()))

# Build selected reach set from mainstem range + connected tributaries
selected_reaches = build_selected_reaches_with_tributaries(
    reach_data,
    mainstem_min_reach,
    mainstem_max_reach,
    mainstem_max_id
)

print("Selected mainstem range:", list(range(mainstem_min_reach, mainstem_max_reach + 1)))
print("Selected reaches including tributaries:", selected_reaches)

# Detect outlet reach
outlet_reach_idx, outlet_reach = detect_outlet_reach(reach_data)

print(f"Detected outlet reach index: {outlet_reach_idx}")
print(f"Detected outlet reach FromN: {outlet_reach}")

# Load outputs
data_output = pickle.load(open(path + name_simu + '.p', "rb"))
direct_connectivity = data_output['Direct connectivity [m^3]']

# Make timestep range consistent between Q and direct_connectivity
n_timesteps_Q = Q.shape[0]
n_timesteps_dc = direct_connectivity.shape[0]

print("Q timesteps:", n_timesteps_Q)
print("Direct connectivity timesteps:", n_timesteps_dc)

max_common_timestep = min(n_timesteps_Q, n_timesteps_dc) - 1

if end_timestep > max_common_timestep:
    print(f"Requested end_timestep={end_timestep} exceeds available common range.")
    print(f"Using end_timestep={max_common_timestep} instead.")
    end_timestep = max_common_timestep

# Discharge at the outlet
Q_outlet = Q[start_timestep:end_timestep + 1, outlet_reach_idx]


# -------------------------------------------------------------------
# Definition: plot connectivity for a specific timestep
# -------------------------------------------------------------------
def plot_connectivity(
    timestep,
    start_date,
    reach_data,
    direct_connectivity,
    output_folder,
    Q_outlet,
    gap_plot,
    outlet_reach,
    start_timestep,
    end_timestep,
    selected_reaches,
    min_volume_to_plot
):
    fig = plt.figure(figsize=(7.48, 8), dpi=300)
    gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1, 2], width_ratios=[1, 2, 1])

    # Top discharge plot
    ax_q = fig.add_subplot(gs[0, 1])

    # Bottom map plot
    ax = fig.add_subplot(gs[1, :])

    fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.07, hspace=0.3)

    # ----- Plot discharge
    q_idx = timestep - start_timestep
    x_q = np.arange(start_timestep, end_timestep + 1)

    ax_q.plot(x_q, Q_outlet)
    ax_q.plot(timestep, Q_outlet[q_idx], 'o')

    ax_q.set_xlabel('Time (day)', fontsize=14)
    ax_q.set_ylabel('Discharge [m3/s] (outlet)', fontsize=14)
    ax_q.tick_params(axis='both', which='major', labelsize=14)

    # ----- Plot map
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

    # Filter reach data for plotting only selected reaches
    reach_data_plot = reach_data[
        reach_data["FromN"].astype(int).isin(selected_reaches)
    ].copy()

    if reach_data_plot.empty:
        raise ValueError("No reaches found in selected_reaches.")

    # Plot reaches in black
    for _, row in reach_data_plot.iterrows():
        geom = row["geometry"]

        if isinstance(geom, LineString):
            x, y = geom.xy
            ax.plot(x, y, color='black')

        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                x, y = part.xy
                ax.plot(x, y, color='black')

    # Extract sediment transport data for given timestep
    transport_data = direct_connectivity[timestep, :, :-1]
    qout_data = direct_connectivity[timestep, :, -1]

    # Normalize color scale
    vmax = np.max(direct_connectivity)
    if vmax <= 1:
        vmax = 1.1

    norm = mcolors.LogNorm(vmin=1, vmax=vmax)
    cmap = cm.viridis

    # Central position only for selected reaches
    pos = {
        int(row["FromN"]): (row["geometry"].centroid.x, row["geometry"].centroid.y)
        for _, row in reach_data_plot.iterrows()
    }

    reach_FromN = reach_data["FromN"].astype(int).to_numpy()

    # Plot arrows for sediment cascades between selected reaches
    for i in range(transport_data.shape[0]):
        for j in range(transport_data.shape[1]):
            start_reach = reach_FromN[i]
            dest_reach = reach_FromN[j]

            if start_reach not in selected_reaches:
                continue
            if dest_reach not in selected_reaches:
                continue

            volume = transport_data[i, j]
            if volume > min_volume_to_plot:
                if start_reach in pos and dest_reach in pos:
                    arrow = FancyArrowPatch(
                        posA=pos[start_reach],
                        posB=pos[dest_reach],
                        connectionstyle="arc3,rad=-0.6",
                        arrowstyle="-|>",
                        mutation_scale=10,
                        facecolor='none',
                        edgecolor=cmap(norm(volume)),
                        alpha=1,
                        lw=1
                    )
                    ax.add_patch(arrow)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Cascade Volume [m³]", fontsize=14)

    # Outlet geometry
    outlet_row = reach_data.loc[reach_data["FromN"].astype(int) == outlet_reach]

    if outlet_row.empty:
        raise ValueError(f"No geometry found for outlet reach {outlet_reach}")

    outlet_geom = outlet_row.iloc[0]["geometry"]
    end_of_network_coords = get_linestring_end_coords(outlet_geom)

    # Plot arrows representing cascades going to the outlet
    for i, vol in enumerate(qout_data):
        reach_id = reach_FromN[i]

        if reach_id not in selected_reaches:
            continue

        if vol > min_volume_to_plot:
            if reach_id in pos:
                arrow = FancyArrowPatch(
                    posA=pos[reach_id],
                    posB=end_of_network_coords,
                    connectionstyle="arc3,rad=0.35",
                    arrowstyle="-|>",
                    mutation_scale=10,
                    facecolor='none',
                    edgecolor=cmap(norm(vol)),
                    alpha=1,
                    lw=1
                )
                ax.add_patch(arrow)

    # Plot node marker at reach starts
    for _, row in reach_data_plot.iterrows():
        geom = row["geometry"]
        start_x, start_y = get_linestring_start_coords(geom)

        ax.scatter(
            start_x, start_y,
            color='white',
            marker='o',
            s=15,
            edgecolors='black',
            linewidth=1,
            zorder=1000
        )

    # Plot outlet node only if outlet reach is in selected reaches
    if outlet_reach in selected_reaches:
        ax.scatter(
            *end_of_network_coords,
            color='none',
            marker='o',
            s=50,
            edgecolors='red',
            linewidth=1.5,
            label="Outlet",
            zorder=1100
        )

    # North arrow
    ax.annotate(
        'N',
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        fontsize=13,
        fontweight='bold',
        ha='center'
    )

    north_arrow = FancyArrow(
        0.95, 0.90, 0, 0,
        width=0,
        head_width=0.03,
        head_length=0.03,
        color='black',
        fill=False,
        overhang=0.2,
        transform=ax.transAxes
    )
    ax.add_patch(north_arrow)

    # Scale bar
    scale_x, scale_y = 0.85, 0.95
    scale_bar_length = 1000
    ax.plot(
        [scale_x, scale_x + 0.05],
        [scale_y, scale_y],
        color='black',
        lw=2,
        transform=ax.transAxes
    )
    ax.text(
        scale_x + 0.025,
        scale_y - 0.05,
        f'{int(scale_bar_length / 1000)} km',
        ha='center',
        fontsize=13,
        transform=ax.transAxes
    )

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Legend
    legend_handles = [
        Line2D([0, 1], [0, 0], color='black', linestyle='-', marker='>',
               markersize=10, markerfacecolor='white', markeredgecolor='black',
               label='Sediment cascade'),
        Line2D([], [], color='none', marker='o', markersize=5,
               markeredgecolor='black', label='Node'),
        Line2D([], [], color='none', marker='o', markersize=6,
               markeredgecolor='red', label='Outlet'),
        Line2D([], [], color='black', label='River Network'),
    ]

    ax.legend(handles=legend_handles, loc='lower left', fontsize=12)

    # Date
    date_plotted = np.datetime64(start_date) + np.timedelta64(timestep, 'D')
    ax.text(
        0.01, 0.99, f"Date: {date_plotted}",
        transform=ax.transAxes,
        fontsize=17,
        ha='left',
        va='top',
        color='black',
        fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none',
                  boxstyle='round,pad=0.3', alpha=0.8),
        zorder=1000
    )

    # Aspect and bounds based only on selected plotted reaches
    ax.set_aspect('equal', adjustable='box')
    x_min, y_min, x_max, y_max = reach_data_plot.geometry.total_bounds
    ax.set_xlim(x_min - gap_plot, x_max + gap_plot)
    ax.set_ylim(y_min - gap_plot, y_max + gap_plot)

    fig.set_size_inches(7000. / fig.dpi, 3000. / fig.dpi)
    fig.savefig(os.path.join(output_folder, f'connect_map_{timestep}.png'))
    fig.clf()
    plt.close("all")


# -------------------------------------------------------------------
# Loop through selected timesteps
# -------------------------------------------------------------------
for t in range(start_timestep, end_timestep + 1):
    plot_connectivity(
        t,
        start_date,
        reach_data,
        direct_connectivity,
        figure_folder,
        Q_outlet,
        gap_plot,
        outlet_reach,
        start_timestep,
        end_timestep,
        selected_reaches,
        min_volume_to_plot
    )