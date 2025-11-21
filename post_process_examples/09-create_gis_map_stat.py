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

years = '2004_2022'
eq = 'R'
dep_l = 100000
layer = 30
trib = '100000'
Dams = 'Dams'



#---------------------Path to the extended pickle output
path = f'K:/Labo/20_OSR/Rhone_Fabio_Schneider/DCASCADE_res{years}_{eq}_classes_{Dams}_Laval/'
name_simu = f'DCASCADE_res{years}_{eq}_{Dams}_Laval'
name_simu_ext = f'DCASCADE_res{years}_{eq}_{Dams}_Laval_ext'

#---River shape files
path_river_network = Path('../inputs/Rhone_river/Network/Laval_data/')
# Reach data file (shp, but can also be a csv)
name_river_network = f'Network_Laval_{Dams}_cal.shp'
filename_river_network = path_river_network / name_river_network

# open simulate
with open( path + name_simu + '.p' , "rb") as readF:
    a = pickle.load(readF)

aSed_budget = a['Sediment budget [m^3]']
aVout = a['Volume out [m^3]']
aVin = a['Volume in [m^3]']
aDep = a['Deposited [m^3]']
aTr = a['Transport capacity [m^3]']
aQ = a['Discharge [m^3/s]']
aQh = a['Flow depth [m]']


aSed_budget = np.sum(aSed_budget, axis=0)
aVout = np.sum(aVout, axis=0)
aVin = np.sum(aVin, axis=0)
aDep = np.sum(aDep, axis=0)
aTr = np.sum(aTr, axis=0)
aQ = np.max(aQ, axis=0)
aQh = np.max(aQh, axis=0)

df_Sed_budget = pd.DataFrame({'SedBudget': aSed_budget})
df_Vout = pd.DataFrame({'Vout': aVout})
df_Vin = pd.DataFrame({'Vin': aVin})
df_Dep = pd.DataFrame({'Dep': aDep})
df_Tr = pd.DataFrame({'Tr': aTr})
df_Qc = pd.DataFrame({'Qc': aQ})
df_Qh = pd.DataFrame({'Qh': aQh})

# Process discharge stats
df_discharge = pd.read_csv(f'../inputs/Rhone_river/Discharge_Laval/Q_Laval_{years}_{Dams}.csv')
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
    df_Tr,
    df_Qc,
    df_Qh
], axis=1)

csv_path = path + f'df_DCASCADE_res{years}_{eq}_{Dams}_Laval.csv'
df_res_output.to_csv(csv_path, index=False)

gdf_rhone = gpd.read_file(filename_river_network)
df_res_cascade = pd.read_csv(csv_path)
gdf_cascade = pd.concat([gdf_rhone, df_res_cascade], axis=1)
gdf_cascade.rename(columns={'fid': 'id'}, inplace=True)

# gdf_cascade.to_file(path_base + f'df_res_output_ext_{year}.shp', driver='ESRI Shapefile')
gdf_cascade.to_file(path + f'DCASCADE_res{years}_{eq}_{Dams}_Laval.gpkg', driver='GPKG')



# open simulate extend
with open( path + name_simu_ext + '.p' , "rb") as readF_ext:
    a_ext = pickle.load(readF_ext)

aSed_budget_class_ext = a_ext['Sediment budget per class [m^3]']
aVout_class_ext = a_ext['Volume out per grain sizes [m^3]']
aVin_class_ext = a_ext['Volume in per grain sizes [m^3]']
aDep_class_ext = a_ext['Deposited per grain sizes [m^3]']
aTr_class_ext = a_ext['Tr_cap per class [m^3]']

aVout_class_ext = np.sum(aVout_class_ext, axis=0)
aTr_class_ext = np.sum(aTr_class_ext, axis=0)
aVin_class_ext = np.sum(aVin_class_ext, axis=0)
aDep_class_ext = np.sum(aDep_class_ext, axis=0)
aSed_budget_class_ext = np.sum(aSed_budget_class_ext, axis=0)

df_Vout_class_ext = pd.DataFrame(aVout_class_ext)
df_Tr_class_ext = pd.DataFrame(aTr_class_ext)
df_Vin_class_ext = pd.DataFrame(aVin_class_ext)
df_Dep_class_ext = pd.DataFrame(aDep_class_ext)
df_Sed_budget_class_ext = pd.DataFrame(aSed_budget_class_ext)

for df, prefix in [
    (df_Vout_class_ext, 'Vout'),
    (df_Tr_class_ext, 'Tr_cap'),
    (df_Vin_class_ext, 'Vin'),
    (df_Dep_class_ext, 'Dep'),
    (df_Sed_budget_class_ext, 'Sed_bud')
]:
    df[f'{prefix}_m'] = df.iloc[:, [0,1,2,3]].sum(axis=1)
    df[f'{prefix}_gravel'] = df.iloc[:, [2,3,4,5,6]].sum(axis=1)
    df[f'{prefix}_sand'] = df.iloc[:, [7,8,9,10,11]].sum(axis=1)
    df[f'{prefix}_Tot_classes'] = df.iloc[:, :12].sum(axis=1)
    
for df, prefix in [
    (df_Tr_class_ext, 'Tr_cap'),
]:
    df[f'{prefix}_m'] = df.iloc[:, [0,1,2,3]].sum(axis=1)
    df[f'{prefix}_gravel'] = df.iloc[:, [2,3,4,5,6]].sum(axis=1)
    df[f'{prefix}_sand'] = df.iloc[:, [7,8,9,10,11]].sum(axis=1)
    df[f'{prefix}_Tot_classes'] = df.iloc[:, :12].sum(axis=1)   

df_Vout_class_def = df_Vout_class_ext.iloc[:, -4:]
df_Tr_class_def = df_Tr_class_ext.iloc[:, -4:]
df_Vin_class_def = df_Vin_class_ext.iloc[:, -4:]
df_Dep_class_def = df_Dep_class_ext.iloc[:, -4:]
df_Sed_budget_class_def = df_Sed_budget_class_ext.iloc[:, -4:]

df_res_output_ext = pd.concat([
    stats_df_Q,
    df_Vin_class_def,
    df_Vout_class_def,
    df_Dep_class_def,
    df_Sed_budget_class_def,
    df_Tr_class_def
], axis=1)

csv_path_ext = path + f'df_DCASCADE_res{years}_{eq}_classes_{Dams}_Laval.csv'
df_res_output_ext.to_csv(csv_path_ext, index=False)

gdf_rhone = gpd.read_file(filename_river_network)
df_res_cascade_ext = pd.read_csv(csv_path_ext)
gdf_cascade_ext = pd.concat([gdf_rhone, df_res_cascade_ext], axis=1)
gdf_cascade_ext.rename(columns={'fid': 'id'}, inplace=True)

# gdf_cascade.to_file(path_base + f'df_res_output_ext_{year}.shp', driver='ESRI Shapefile')
gdf_cascade_ext.to_file(path + f'DCASCADE_res{years}_{eq}_classes_{Dams}_Laval.gpkg', driver='GPKG')

