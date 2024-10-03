# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:36:06 2024

@author: Anne-Laure Argentin
"""

import numpy as np

class Cascade:
    def __init__(self, provenance, elapsed_time, volume):
        self.provenance = provenance
        self.elapsed_time = elapsed_time
        self.volume = volume


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
        if 'reach_id' in geodataframe.columns:
            self.reach_id = geodataframe['reach_id'].values
        else:
            self.reach_id = np.nan
        if 'Id' in geodataframe.columns:
            self.id = geodataframe['Id'].values
        else:
            self.id = np.nan
        if 'Wac_BF' in geodataframe.columns:
            self.wac_bf = geodataframe['Wac_BF'].values
        else:
            self.wac_bf = np.nan
        if 'D90' in geodataframe.columns:
            self.D90 = geodataframe['D90'].values
        else:
            self.D90 = np.nan
        if 'S_LR_GIS' in geodataframe.columns:
            self.s_lr_gis = geodataframe['S_LR_GIS'].values
        else:
            self.s_lr_gis = np.nan
        if 'Q' in geodataframe.columns:
            self.Q = geodataframe['Q'].values
        else:
            self.Q = np.nan
        if 'tr_limit' in geodataframe.columns:
            self.tr_limit = geodataframe['tr_limit'].values
        else:
            self.tr_limit = np.nan
        if 'x_FN' in geodataframe.columns:
            self.x_fn = geodataframe['x_FN'].values
        else:
            self.x_fn = np.nan
        if 'y_FN' in geodataframe.columns:
            self.y_fn = geodataframe['y_FN'].values
        else:
            self.y_fn = np.nan
        if 'x_TN' in geodataframe.columns:
            self.x_tn = geodataframe['x_TN'].values
        else:
            self.x_tn = np.nan
        if 'y_TN' in geodataframe.columns:
            self.y_tn = geodataframe['y_TN'].values
        else:
            self.y_tn = np.nan
        if 'Ad' in geodataframe.columns:
            self.ad = geodataframe['Ad'].values
        else:
            self.ad = np.nan        
        if 'StrO' in geodataframe.columns:
            self.strO = geodataframe['StrO'].values
        else:
            self.strO = np.nan
        if 'deposit' in geodataframe.columns:
            self.deposit = geodataframe['deposit'].values
        else:
            self.deposit = np.nan
        if 'geometry' in geodataframe.columns:
            self.geometry = geodataframe['geometry'].values
        else:
            self.geometry = np.nan
        if 'directAd' in geodataframe.columns:
            self.direct_ad = geodataframe['directAd'].values
        else:
            self.direct_ad = np.nan            

    def sort_values_by(self, sorting_array):
        """
        Function to sort the Reaches by the array given in input.
        """
        # Making sure the array given has the right length
        assert (len(sorting_array) == self.n_reaches)

        # Get the indices that would sort sorting_array
        sorted_indices = np.argsort(sorting_array)

        # Loop through all attributes
        for attr_name in vars(self):

            # Check if these are reach attributes
            attr_value = vars(self)[attr_name]
            if isinstance(attr_value, np.ndarray) \
                    and len(attr_value) == self.n_reaches:
                vars(self)[attr_name] = attr_value[sorted_indices]

        return sorted_indices


