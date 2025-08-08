"""
Created on Tue Oct 29 10:58:54 2024

@author: diane
"""

import numpy as np


class ReachData:
    """
    @brief Class to store reach inputs paramaters.

    Contains all input information per each reach (e.g. node indexes, initial grain size, width, slope...etc)

    @param geodataframe
        Dataframe table, that is read from the reach data input csv or shp file.

    """
    def __init__(self, geodataframe):
        self.geodf = geodataframe
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

        self.roughness = self.compute_roughness()

        # Optional attributes
        self.reach_id = geodataframe['reach_id'].values if 'reach_id' in geodataframe.columns else None
        self.id = geodataframe['Id'].values if 'Id' in geodataframe.columns else None
        self.q = geodataframe['Q'].values if 'Q' in geodataframe.columns else None
        self.wac_bf = geodataframe['Wac_BF'].values if 'Wac_BF' in geodataframe.columns else None
        self.D90 = geodataframe['D90'].values if 'D90' in geodataframe.columns else None
        self.s_lr_gis = geodataframe['S_LR_GIS'].values if 'S_LR_GIS' in geodataframe.columns else None
        self.tr_limit = geodataframe['tr_limit'].values if 'tr_limit' in geodataframe.columns else None
        self.x_fn = geodataframe['x_FN'].values if 'x_FN' in geodataframe.columns else None
        self.y_fn = geodataframe['y_FN'].values if 'y_FN' in geodataframe.columns else None
        self.x_tn = geodataframe['x_TN'].values if 'x_TN' in geodataframe.columns else None
        self.y_tn = geodataframe['y_TN'].values if 'y_TN' in geodataframe.columns else None
        self.ad = geodataframe['Ad'].values if 'Ad' in geodataframe.columns else None
        self.direct_ad = geodataframe['directAd'].values if 'directAd' in geodataframe.columns else None
        self.strO = geodataframe['StrO'].values if 'StrO' in geodataframe.columns else None
        self.deposit = geodataframe['deposit'].values if 'deposit' in geodataframe.columns else None
        self.geometry = geodataframe['geometry'].values if 'geometry' in geodataframe.columns else None



    def sort_values_by(self, sorting_array):
        """
        Function to sort the reaches by the array given in input (e.g the upstream node index).
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

    def compute_roughness(self):
        """
        Function to set the roughness attribute.
        If a roughness parameter is not given in the reach data, then is takes D90 or D84.
        """
        # to test and see if it is what we want in terms of physics
        if 'roughness' in self.geodf:
            roughness = self.geodf['roughness'].astype(float).values
        elif 'D90' in self.geodf:
            roughness = self.geodf['D90'].astype(float).values
        else:
            roughness = self.geodf['D84'].astype(float).values
        return roughness
