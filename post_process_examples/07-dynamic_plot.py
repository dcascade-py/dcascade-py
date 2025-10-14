# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:17:11 2025

@author: diane


Creates a dynamic plot

Choose between:
'Volume out [m^3]':         total volume of sediment leaving a reach per time step (= sediment flux x time step)
'Volume in [m^3]':          total volume of sediment entering a reach per time step
'Transport capacity [m^3]': total transport capacity computed in a reach per time step (= volume out if the supply is not limited)
'Sediment budget [m^3]':    total sediment budget per time step (+ deposition, - erosion) (= vol in - vol out)
'D50 active layer [m]':     D50 of the active layer per time step (used to computed the transport capacity)
'D50 volume out [m]' :      D50 of the volume leaving the reach per time step 

"""

# Libraries
import sys, os
import pandas as pd
import geopandas as gpd

# Add source (src) folder in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from plot_function import dynamic_plot

years = '2016_2018'
eq = 'W&P'
dep_l = 2

# path = f'K:/Labo/20_OSR/Rhone_Fabio_Schneider/Result_Cascade_21_67reaches_{ref2}GSD-silt/res_E{index}/res_hy{year}_E{index}/'
# name_simu = f'save_all_E{index}'

#---------------------Path to the pickle output

path = f'K:/Labo/20_OSR/Rhone_Fabio_Schneider/Rhone_Slocal_res{years}_{eq}/' 
name_simu = f'Rhone_Martinez_res{years}_{eq}_{dep_l}m'

#---------------------Path to the input river network (.shp) or (.csv)

path_river_network = "../inputs/Rhone_river/Network/" #Path to the shp
name_river_network = "rhone_DCASCADE.shp"


       

###########################################################################

data_output = pd.read_pickle(open( path + name_simu + '.p' , "rb"))
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network)

keep_slider = dynamic_plot(data_output, ReachData)