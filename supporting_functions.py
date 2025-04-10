"""
Created on Fri Oct 14 16:56:59 2022

This file contains the functions:

    (Functions from version 1 that we keep)
    - D_finder: finds the value of granulometry for the specified D_values for the sediment distribution Fi_r
    input: fi_r - grain size distribution of the active layer for a specific reach. In each column there is the % of the sediment class represented by that column (see psi)
           D_value = the quantile (eg. 50 for D50)

    - sortdistance: sorts the rows of Qbi_incoming matrix by increasing distance from the reach

    - layer_search: puts part of the incoming and deposited sediment volumes into the
      active layer according to the active layer volume and the incoming and deposited volumes

    - matrix_compact: takes a stratigraphy matrix V_layer and compact it by summing all the layers with the same source reach id

    - tr_cap_deposit: deposits part of the incoming and deposited sediment volumes
      according to the transport capacity calculated

    (New functions)
    - cascades_end_time_or_not
    - stop_or_not
    - deposit_from_passing_sediments
    - compute_time_lag



This script was adapted from the Matlab version by Marco Tangi
@author: Elisa Bozzolan
"""


from itertools import groupby

import numpy as np
import pandas as pd

# ignore divide by 0
np.seterr(divide='ignore', invalid='ignore')




def D_finder(fi_r, d_value, psi):
    """
    @brief Computes the grain size corresponding to a given cumulative percentage (DXX)
           for a reach with a specified grain size distribution.

    @param fi_r Array of fractions of material in each grain size class.
                For example, [0.15, 0.50, 0.35] represents 15%, 50%, and 35% of material
                in three grain size classes.
    @param d_value The target DXX value (e.g., 50 for D50, 90 for D90) representing the
                   grain size for which X% of the material is finer.
    @param psi Array of grain sizes in negative log base-2 scale. For example,
               `-3` represents 2³ = 8 mm, `-4` represents 2⁴ = 16 mm.

    @return Array of computed grain sizes (in meters) corresponding to the requested DXX
            for each input layer. If only one grain size class is present, the function
            returns that grain size. If `d_value` equals 100, it returns the smallest
            grain size.

    @details
    - The function supports both single-layer (`fi_r` as a vector) and multi-layer (`fi_r`
      as a matrix) inputs. For multi-layer cases, it processes each layer independently.
    - The function assumes that the provided grain sizes in `psi` are the maximum sizes
      of each class and not average sizes.

    @note The grain size distribution is interpolated in logarithmic (base-2) space,
          as is standard in sedimentology, and converted back to linear scale in meters.

    @example
    ```
    fi_r = [0.15, 0.50, 0.35]
    d_value = 50
    psi = [-3, -4, -5]
    result = D_finder(fi_r, d_value, psi)
    ```
    """

    dmi = np.power(2, -psi)/1000
    nb_classes = len(psi)

    # Handles the case of single layers input as vector and
    # multiple layers input as matrices.
    if fi_r.ndim == 1:
        nb_layers = 1
        fi_r = np.expand_dims(fi_r, axis=0)
    else: # If multiple layers: AL: Do we need it? Neither Vjosa nor Po use it.
        nb_layers = np.shape(fi_r)[0]

    # If one class only, the target DXX value is the grain size of the class.
    if dmi.size == 1:
        return np.full(nb_layers, dmi[0])

    # The D100 is the grain size of the biggest class.
    elif d_value == 100:
        return np.full(nb_layers, dmi[0])

    else:
        # Computes the inverse cumulative percentage of finer material,
        # class by class, for each layer.
        # Assumes that the given grain size is the maximum grain size of each
        # class, and not the average grain size, for example.
        perc_finer = np.empty(np.shape(fi_r))
        perc_finer[:] = np.nan
        perc_finer[:,0] = 100
        for i in range(1, nb_classes):
            perc_finer[:,i] = perc_finer[:,i-1] - fi_r[:,i-1] * 100

        # Computes the target DXX value.
        d_changes = np.zeros(nb_layers)
        for k in range(nb_layers):
            # Finds the class index of the percentage just above the target DXX value.
            class_index = np.where(perc_finer[k, :] > d_value)[0].max()
            # Ensure within valid range, which means that the interpolation to determine
            # the target DXX value, if below the smallest material fraction, will be
            # based on the adjacent above interval.
            class_index = np.minimum(class_index, nb_classes - 2)

            # Interpolation
            perc_diff = perc_finer[k, class_index] - perc_finer[k, class_index + 1]
            psi_diff = -psi[class_index] + psi[class_index + 1]  # Because psi is negative...
            # How much of the target cumulative percentage (d_value) lies above
            # the percentage for class_index + 1, divided by the cumulative percentage
            # difference to get a fraction.
            interpolated_fraction = (d_value - perc_finer[k, class_index + 1]) / perc_diff
            # Apply this fractional position to the grain size values and add the
            # grain size values for class index + 1.
            d_changes[k] = interpolated_fraction * psi_diff - psi[class_index + 1]

            # Converts back to meters
            d_changes[k] = np.power(2, d_changes[k]) / 1000
            # Ensures no negative sizes by replacing them with the smallest size in dmi
            d_changes[k] = d_changes[k] * (d_changes[k] > 0) + dmi[-1] * (d_changes[k] < 0)

        return d_changes


def sortdistance(Qbi, distancelist):
    '''
    '''

    idx = np.argwhere(Qbi[:, 0][:,None] == distancelist[~(np.isnan(distancelist))])[:,1]

    if idx.size != 0 and len(idx) != 1:  # if there is a match #EB check
        Qbi_sort = np.array(Qbi[(idx-1).argsort(), :])
    else:
        Qbi_sort = Qbi

    return Qbi_sort





