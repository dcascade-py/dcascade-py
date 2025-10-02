# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:24:43 2025

@author: andress251
"""

# import package
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd

# index value of the results
years = '2004_2006'
eq = 'W&P'
reaches = [33] # reaches objective

path = "..\\cascade_results\\" 

name_simu = f'Rhone_Martinez_res{years}_ext'

with open(path + name_simu + '.p', 'rb') as readF:
    a = pickle.load(readF)

aV_out_class = a['Volume out per grain sizes [m^3]']
aTr_cap_class = a['Tr_cap per class [m^3]']
aV_in_class = a['Volume in per grain sizes [m^3]']
aDep_class = a['Deposited per grain sizes [m^3]']
aSed_budget_class = a['Sediment budget per class [m^3]']

for reach_id in reaches:
    aV_out_class_reach = aV_out_class[:,reach_id,:]
    aTr_cap_class_reach = aTr_cap_class[:,reach_id,:]
    aV_in_class_reach = aV_in_class[:,reach_id,:]
    aDep_class_reach = aDep_class[:,reach_id,:]
    aSed_budget_class_reach = aSed_budget_class[:,reach_id,:]

    df_V_out_class = pd.DataFrame(aV_out_class_reach)
    df_Tr_cap_class = pd.DataFrame(aTr_cap_class_reach)
    df_V_in_class = pd.DataFrame(aV_in_class_reach)
    df_Dep_class = pd.DataFrame(aDep_class_reach)
    df_Sed_budget_class = pd.DataFrame(aSed_budget_class_reach)

    for df, prefix in [
        (df_V_out_class, 'Vout'),
        (df_V_in_class, 'Vin'),
        (df_Dep_class, 'Dep'),
        (df_Sed_budget_class, 'Sed_bud')
    ]:
        # df[f'{prefix}_boulder'] = df.iloc[:, [0,1,2,3]].sum(axis=1)
        df[f'{prefix}_gravel'] = df.iloc[:, :9].sum(axis=1)
        # df[f'{prefix}_sand'] = df.iloc[:, [9,10,11,12,13]].sum(axis=1)
        # df[f'{prefix}_Tot_classes'] = df.iloc[:, :14].sum(axis=1)
    
    for df, prefix in [
        (df_Tr_cap_class, 'Tr_cap'),
    ]:
        # df[f'{prefix}_boulder'] = df.iloc[:, [0,1,2,3]].mean(axis=1)
        df[f'{prefix}_gravel'] = df.iloc[:, :9].mean(axis=1)
        # df[f'{prefix}_sand'] = df.iloc[:, [9,10,11,12,13]].mean(axis=1)
        # df[f'{prefix}_Tot_classes'] = df.iloc[:, :14].mean(axis=1)    

    df_V_out_class_def = df_V_out_class.iloc[:, -1:]
    df_Tr_cap_class_def = df_Tr_cap_class.iloc[:, -1:]
    df_V_in_class_def = df_V_in_class.iloc[:, -1:]
    df_Dep_class_def = df_Dep_class.iloc[:, -1:]
    df_Sed_budget_class_def = df_Sed_budget_class.iloc[:, -1:]

    # Process discharge stats
    df_discharge = pd.read_csv(f'../inputs/Rhone_river/discharge_martinez/Q_{years}.csv')
    df_Q = df_discharge.iloc[::,[0,reach_id]]
    
    # To correct the index
    df_Q.index = df_V_out_class_def.index

    df_res_output_ext = pd.concat([
        df_Q,
        df_V_out_class_def, 
        df_Tr_cap_class_def, 
        df_V_in_class_def,
        df_Dep_class_def,
        df_Sed_budget_class_def
    ], axis=1)
    
    path_save = path

    # Save df_total to CSV
    if not os.path.exists(path_save):   #does the output folder exist ?
        os.makedirs(path_save)          # if not, create it.

    csv_path = path_save + f'df_res_reach{reach_id}_hy{years}_{eq}.csv'
    df_res_output_ext.to_csv(csv_path, index=False)
