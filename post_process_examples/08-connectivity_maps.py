# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:40:14 2025

@author: FPitscheider

Plot connectivity map

Show map with sediment path-length, per each time step

"""

import os, sys
import numpy as np
import pickle
import geopandas as gpd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.patches import FancyArrow
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from shapely.geometry import Point
from shapely.geometry import LineString, MultiLineString
import matplotlib.animation as animation

# Add source (src) folder in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing import extract_Q, read_network                     
from reach_data import ReachData


%matplotlib inline


years = '2004_2006'
eq = 'W&C'
dep_l = 10
layer = 30
trib = '15'

# ---------------------Path to the pickle output
path = f'K:/Labo/20_OSR/Rhone_Fabio_Schneider/Rhone_Slocal_trib{trib}_res{years}_{eq}_layer{layer}cm/'
name_simu = f'Rhone_Martinez_res{years}_{eq}_{dep_l}m_trib{trib}_layer{layer}cm'
name_simu_ext = f'Rhone_Martinez_res{years}_{eq}_{dep_l}m_trib{trib}_layer{layer}cm_ext'

#---------------------Path to the input river network (.shp) or (.csv)
path_river_network = '../inputs/Rhone_river/Network/' #Path to the shp
name_river_network = "rhone_DCASCADE.shp"


#---------------------Path to the discharge file
path_Q = '../inputs/Rhone_river/discharge_martinez/'
name_q = f'Q_{years}_dam.csv' 


#---------------------Folder to store the plots
figure_folder = path+'figures_connectivity_maps_{years}_{eq}\\'          # where you will store the figure

if not os.path.exists(figure_folder):       
    os.makedirs(figure_folder)
        
#--------------------Define time range (i.e. the time step you want to plot)
start_timestep = 0      # start time step
end_timestep = 10        # end time step
      
start_date = np.datetime64('2003-09-21') # date of the starting time step, for the legend

#--------------------Indicate outlet reach index (FromN)
outlet_FromN = 67

#--------------------Gap to adjust for plotting
gap_plot = 15000



####################################################################

# Read the network
reach_data = read_network(path_river_network + name_river_network)

# Read/define the water discharge
Q, number_days = extract_Q(path_Q + name_q)

# Sort reach_data according to the from_n, and organise the Q file accordingly
sorted_indices = reach_data.sort_values(by="FromN").index
Q_new = np.zeros(Q.shape)
for i, idx in enumerate(sorted_indices):
    Q_new[:,i] = Q.iloc[:,idx]
Q = Q_new
reach_data = reach_data.sort_values(by="FromN", ignore_index = True)


# Discharge at the outlet
Q_outlet = Q[start_timestep : end_timestep + 1, outlet_FromN - 1]

# Load outputs
data_output = pickle.load(open(path + name_simu + '.p', "rb"))
direct_connectivity = data_output['Direct connectivity [m^3]']


#---Definition: plot connectivity for a specific timestep
def plot_connectivity(timestep, start_date, reach_data, direct_connectivity, output_folder, Q_outlet, gap_plot):
        
    
    # Create figure and its subplots
    fig = plt.figure(figsize=(7.48, 8), dpi=300)
    gs = gridspec.GridSpec(nrows=3, ncols=3, height_ratios=[1,2,1], width_ratios=[1,1,1])

    # Discharge plot: Top plot spans the middle column only, making it narrower and centered
    ax_q = fig.add_subplot(gs[1,:-1])
    
    # Network plot: Bottom plot spans all three columns, making it wider
    ax = fig.add_subplot(gs[:,2])
    
    fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.07, hspace=0.3)
        

    #-----Plot discharge
    ax_q.plot(Q_outlet)    
    ax_q.plot(timestep, Q_outlet[timestep], 'o')

    ax_q.set_xlabel('Time (day)', fontsize = 14)
    ax_q.set_ylabel('Discharge [m3/s] (outlet)', fontsize = 14)
    ax_q.tick_params(axis = 'both', which = 'major', labelsize = 14)
    
    
    #-----Plot map
        
    # Set font properties
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    
    # Plot reaches in black    
    for _, row in reach_data.iterrows():
        reach_id = row['FromN']

        # Plot the river reach with the assigned color and label  
        geom = row.geometry
        
        if isinstance(geom, LineString):
            x, y = geom.xy

        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                x, y = part.xy
        ax.plot(x, y, color='black')  # or same color if you prefer


    # Extract sediment transport data for given timestep
    transport_data = direct_connectivity[timestep, :, :-1] # Sediment depositing in reaches
    qout_data = direct_connectivity[timestep, :, -1] # Sediment passing the outlet
    
    # Normalize color scale for transport volume
    norm = mcolors.LogNorm(vmin=1, vmax=np.max(direct_connectivity))
    cmap = cm.viridis
    
    # Create network graph
    G = nx.DiGraph()
    
    # Central position of reaches
    pos = {row['FromN']: (row.geometry.centroid.x, row.geometry.centroid.y) for _, row in reach_data.iterrows()}    
    
    # Reach FromN index
    reach_FromN = reach_data.FromN.astype(int)
    
    # Plot arrows for representing the transport of sediment cascades
    transport_legend_added = False
    for i in range(transport_data.shape[0]):  # Iterate over start reaches
        for j in range(transport_data.shape[1]):  # Iterate over destination reaches
            volume = transport_data[i, j]
            if volume > 0:
                start_reach = reach_FromN[i]
                dest_reach = reach_FromN[j]
                label = "Sediment Transport" if not transport_legend_added else ""
                arrow = FancyArrowPatch(posA=pos[start_reach], posB=pos[dest_reach], connectionstyle=f"arc3,rad={-0.6}", # if start_reach < confluence_node else 0.6
                                        arrowstyle="-|>", mutation_scale=10, facecolor='none', edgecolor=cmap(norm(volume)), alpha=1, lw=1)
                ax.add_patch(arrow)
                transport_legend_added = True
    
    # Add color scale to legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)  # Make colorbar thinner
    cbar.set_label("Cascade Volume [m³]", fontsize = 14)
    
    # Coordinate of the outlet point
    end_of_network_coords = reach_data.loc[reach_data['FromN'] == outlet_FromN, 'geometry'].values[0].coords[-1]
        
    # Plot arrows representing cascades going to the outlet
    # with opposite curvature   
    for i, vol in enumerate(qout_data):
        if vol > 0:
            reach_id = reach_FromN[i]
            arrow = FancyArrowPatch(posA=pos[reach_id], posB=end_of_network_coords, connectionstyle=f"arc3,rad={0.35}", #if reach_id < confluence_node else -0.35
                                    arrowstyle="-|>", mutation_scale=10, facecolor='none', edgecolor=cmap(norm(vol)), alpha=1, lw=1)
            ax.add_patch(arrow)
    
    # Plot a circle marker at each start of a reach
    node_legend_added = False
    for _, row in reach_data.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            start_x, start_y = geom.coords[0]

        elif isinstance(geom, MultiLineString):
            # Access the first LineString in the MultiLineString
            first_line = list(geom.geoms)[0]  # or geom.geoms[0]
            start_x, start_y = first_line.coords[0]
        # start_x, start_y = row.geometry.coords[0]  # Get start node coordinates
        label = "Node" if not node_legend_added else ""
        ax.scatter(start_x, start_y, color='white', marker='o', s=15, edgecolors='black', linewidth=1, label=label, zorder = 1000)
        node_legend_added = True
    
    # Plot the outlet node at fixed coordinates
    ax.scatter(*end_of_network_coords, color='none', marker='o', s=50, edgecolors='red', linewidth=1.5, label="Monitoring Station")

    # Add north arrow
    ax.annotate('N', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=13, fontweight='bold', ha='center')

    arrow = FancyArrow(0.05, 0.85, 0, 0, 
                       width=0, head_width=0.03, head_length=0.03, 
                       color='black', fill=False, overhang=0.2,
                       transform=ax.transAxes)  # Use axes-relative coordinates
    ax.add_patch(arrow)
        
    # Add scale, next to north arrow
    scale_x, scale_y = 0.15, 0.9
    scale_bar_length = 1000  # Fixed to 1 km (you can change this if needed)    
    ax.plot([scale_x, scale_x + 0.05], [scale_y, scale_y], color='black', lw=2, transform=ax.transAxes)    
    # scale bar label (1 km)
    ax.text(scale_x + 0.025, scale_y - 0.05, f'{int(scale_bar_length / 1000)} km', ha='center', fontsize=13, transform=ax.transAxes)
    
    # Remove x and y labels and map border
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Customize legend with an empty arrow symbol for Sediment Transport, positioned lower left
    legend_handles = [
        Line2D([0, 1], [0, 0], color='black', linestyle='-', marker='>', markersize=10, markerfacecolor='white', markeredgecolor='black', label='Sediment cascade'),
        Line2D([], [], color='none', marker='o', markersize=8, markeredgecolor='black', label='Node'),
        Line2D([], [], color='none', marker='o', markersize=8, markeredgecolor='red', label='Outlet'),
        Line2D([], [], color='black', label='River Network'),
    ]
  
    ax.legend(handles=legend_handles, loc='lower right', fontsize = 15)
    
    # Plot the date in the top left
    date_plotted = np.datetime64(start_date) + np.timedelta64(timestep, 'D')
    ax.text(0.01, 0.99, f"Date: {date_plotted}", transform=ax.transAxes, fontsize=17, ha='left', va='top', color='black', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.8), zorder=1000)

    
    # Deal with some aspect of the plots
    ax.set_aspect('equal', adjustable='box')        
    x_min, y_min, x_max, y_max = reach_data.geometry.total_bounds
    ax.set_xlim(x_min - gap_plot, x_max + gap_plot)
    ax.set_ylim(y_min - gap_plot, y_max + gap_plot)
    
    
    fig.set_size_inches(7000./fig.dpi, 3000./fig.dpi)
    fig.savefig(figure_folder + 'connect_map_'+str(timestep))
    fig.clf()
    plt.close("all")


#---Loop through selected timesteps and call the plotting definition "plot_connectivity"
for t in range(start_timestep, end_timestep + 1):
    plot_connectivity(t, start_date, reach_data, direct_connectivity, figure_folder, Q_outlet, gap_plot)

