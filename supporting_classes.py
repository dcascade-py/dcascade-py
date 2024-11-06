# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:58:54 2024

@author: diane
"""

import numpy as np




class Cascade:
    def __init__(self, provenance, elapsed_time, volume):
        self.provenance = provenance
        self.elapsed_time = elapsed_time
        self.volume = volume
        # To be filled during the time step
        self.velocities = np.nan


           
        
class ReachData:
    def __init__(self, geodataframe):
        self.n_reaches = len(geodataframe)
        
        # Mandatory attributes
        self.from_n = geodataframe['FromN'].astype(int).values
        self.to_n = geodataframe['ToN'].astype(int).values
        self.slope = geodataframe['Slope'].astype(float).values
        self.wac = geodataframe['Wac'].astype(float).values
        self.n = geodataframe['n'].astype(float).values
        self.D16 = geodataframe['D16'].astype(float).values
        self.D50 = geodataframe['D50'].astype(float).values
        self.D84 = geodataframe['D84'].astype(float).values
        self.length = geodataframe['Length'].astype(float).values
        self.el_fn = geodataframe['el_FN'].astype(float).values
        self.el_tn = geodataframe['el_TN'].astype(float).values

        # Optional attributes
        self.reach_id = geodataframe['reach_id'].values if 'reach_id' in geodataframe.columns else np.nan
        self.id = geodataframe['Id'].values if 'Id' in geodataframe.columns else np.nan
        self.q = geodataframe['Q'].values if 'Q' in geodataframe.columns else np.nan
        self.wac_bf = geodataframe['Wac_BF'].values if 'Wac_BF' in geodataframe.columns else np.nan
        self.D90 = geodataframe['D90'].values if 'D90' in geodataframe.columns else np.nan
        self.s_lr_gis = geodataframe['S_LR_GIS'].values if 'S_LR_GIS' in geodataframe.columns else np.nan
        self.tr_limit = geodataframe['tr_limit'].values if 'tr_limit' in geodataframe.columns else np.nan
        self.x_fn = geodataframe['x_FN'].values if 'x_FN' in geodataframe.columns else np.nan
        self.y_fn = geodataframe['y_FN'].values if 'y_FN' in geodataframe.columns else np.nan
        self.x_tn = geodataframe['x_TN'].values if 'x_TN' in geodataframe.columns else np.nan
        self.y_tn = geodataframe['y_TN'].values if 'y_TN' in geodataframe.columns else np.nan
        self.ad = geodataframe['Ad'].values if 'Ad' in geodataframe.columns else np.nan
        self.direct_ad = geodataframe['directAd'].values if 'directAd' in geodataframe.columns else np.nan
        self.strO = geodataframe['StrO'].values if 'StrO' in geodataframe.columns else np.nan
        self.deposit = geodataframe['deposit'].values if 'deposit' in geodataframe.columns else np.nan
        self.geometry = geodataframe['geometry'].values if 'geometry' in geodataframe.columns else np.nan
        
    def sort_values_by(self, sorting_array):
        """
        Function to sort the Reaches by the array given in input.
        """
        # Making sure the array given has the right length
        assert(len(sorting_array) == self.n_reaches)
        
        # Get the indices that would sort sorting_array
        sorted_indices = np.argsort(sorting_array)
        
        # Loop through all attributes
        for attr_name in vars(self):

            # Check if these are reach attributes
            attr_value = vars(self)[attr_name]
            if isinstance(attr_value, np.ndarray) and len(attr_value) == self.n_reaches:
                vars(self)[attr_name] = attr_value[sorted_indices]
                
        return sorted_indices