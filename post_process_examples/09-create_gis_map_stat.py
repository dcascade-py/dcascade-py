# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 10:25:47 2025

@author: andress251
"""


# import package
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

years = '2004_2006'
eq = 'W&C'
dep_l = 10

#---------------------Path to the extended pickle output
path = f'K:/Labo/20_OSR/Rhone_Fabio_Schneider/Rhone_Martinez_res{years}_{eq}/' 
# path = "..\\cascade_results\\" 

name_simu = f'Rhone_Martinez_res{years}_{eq}_{dep_l}m'

#---River shape files
path_river_network = Path('../inputs/Rhone_river/Network/')
# Reach data file (shp, but can also be a csv)
name_river_network = 'Rhone.shp'
filename_river_network = path_river_network / name_river_network

with open( path + name_simu + '.p' , "rb") as readF:
    a = pickle.load(readF)

aSed_budget = a['Sediment budget [m^3]']
aVout = a['Volume out [m^3]']
aVin = a['Volume in [m^3]']
aDep = a['Deposited [m^3]']
aTr = a['Transport capacity [m^3]']


aSed_budget = np.sum(aSed_budget, axis=0)
aVout = np.sum(aVout, axis=0)
aVin = np.sum(aVin, axis=0)
aDep = np.sum(aDep, axis=0)
aTr = np.sum(aTr, axis=0)

df_Sed_budget = pd.DataFrame({'SedBudget': aSed_budget})
df_Vout = pd.DataFrame({'Vout': aVout})
df_Vin = pd.DataFrame({'Vin': aVin})
df_Dep = pd.DataFrame({'Dep': aDep})
df_Tr = pd.DataFrame({'Tr': aTr})

# Process discharge stats
df_discharge = pd.read_csv(f'../inputs/Rhone_river/discharge_martinez/Q_{years}.csv')
df_Q = df_discharge.iloc[:,1:]

stats_df_Q = pd.DataFrame({
    'Qmean': df_Q.mean(),
    'Qmedian': df_Q.median(),
    'Qmax': df_Q.max(),
    'Qmin': df_Q.min()
})

stats_df_Q = stats_df_Q.T
stats_df_Q = stats_df_Q.T

# To correct the index
stats_df_Q.index = df_Sed_budget.index

df_res_output = pd.concat([
    stats_df_Q,
    df_Sed_budget,
    df_Vout,
    df_Vin,
    df_Dep,
    df_Tr
], axis=1)

csv_path = path + f'df_Martinez_res{years}_{eq}_{dep_l}m.csv'
df_res_output.to_csv(csv_path, index=False)

gdf_rhone = gpd.read_file(filename_river_network)
df_res_cascade = pd.read_csv(csv_path)
gdf_cascade = pd.concat([gdf_rhone, df_res_cascade], axis=1)
gdf_cascade.rename(columns={'fid': 'id'}, inplace=True)

# gdf_cascade.to_file(path_base + f'df_res_output_ext_{year}.shp', driver='ESRI Shapefile')
gdf_cascade.to_file(path + f'Martinez_res{years}_{eq}_{dep_l}m.gpkg', driver='GPKG')

