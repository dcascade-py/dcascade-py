# -*- coding: utf-8 -*-
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



# def layer_search(V_dep_old, V_lim_tot_n, roundpar, Qbi_incoming = None):
#     """
#     This function searches layers that are to be put in the maximum mobilisable
#     layer of a time step. (i.e. the maximum depth to be mobilised).

#     INPUTS:
#     V_dep_old is :      the reach deposit layer
#     V_lim_tot_n  :      is the total maximum volume to be mobilised
#     Qbi_incoming :      is the cascade stopping there from the previous time step

#     RETURN:
#     V_inc2act    :      Layers of the incoming volume to be put in the active layer
#     V_dep2act    :      layers of the deposit volume to be put in the active layer
#     V_dep        :      remaining deposit layer
#     Fi_r_reach   :      fraction of sediment in the active layer
#     """

#     if Qbi_incoming is None:
#         # Empty layer (for computation)
#         n_classes = V_dep_old.shape[1] - 1
#         empty_incoming_volume = np.hstack((0, np.zeros(n_classes)))
#         empty_incoming_volume = np.expand_dims(empty_incoming_volume, axis = 0)
#         Qbi_incoming = empty_incoming_volume

#     # if, considering the incoming volume, I am still under the threshold of the active layer volume...
#     if (V_lim_tot_n - np.sum(Qbi_incoming[:, 1:])) > 0:

#         # ... I put sediment from the deposit layer into the active layer
#         # remaining active layer volume after considering incoming sediment cascades
#         V_lim_dep = V_lim_tot_n - np.sum(Qbi_incoming[:, 1:])
#         csum = np.flipud(np.cumsum(np.flipud(np.sum(V_dep_old[:, 1:], axis=1)), axis = 0)) # EB check again

#         V_inc2act = Qbi_incoming  # all the incoming volume will end up in the active layer

#         # find active layer

#         if (np.argwhere(csum > V_lim_dep)).size == 0 :  # the vector is empty # EB check again
#             # if the cascades in the deposit have combined
#             # volume that is less then the active layer volume (i've reached the bottom)

#             print(' reach the bottom ....')

#             V_dep2act = V_dep_old  # I put all the deposit into the active layer
#             V_dep = np.c_[V_dep_old[0,0], np.zeros((1,Qbi_incoming.shape[1]-1))]


#         else:

#             index = np.max(np.argwhere(csum >= V_lim_dep))


#             # if i have multiple deposit layers, put the upper layers into the active layer until i reach the threshold.
#             # The layer on the threshold (defined by position index) gets divided according to perc_layer
#             perc_layer = (V_lim_dep - np.sum(V_dep_old[csum < V_lim_dep, 1:]))/sum(V_dep_old[index, 1:])  # EB check again  # percentage to be lifted from the layer on the threshold

#             # remove small negative values that can arise from the difference being very close to 0
#             perc_layer = np.maximum(0, perc_layer)

#             if ~np.isnan(roundpar):
#                 V_dep2act = np.vstack((np.hstack((V_dep_old[index, 0], np.around(V_dep_old[index, 1:]*perc_layer, decimals=roundpar))).reshape(1, -1), V_dep_old[csum<V_lim_dep,:]))
#                 V_dep = np.vstack((V_dep_old[0:index,:], np.hstack((V_dep_old[index,0], np.around(V_dep_old[index,1:]* (1-perc_layer), decimals=roundpar)))))
#             else:
#                 V_dep2act = np.vstack((np.hstack((V_dep_old[index, 0], np.around( V_dep_old[index, 1:]*perc_layer))).reshape(1, -1), V_dep_old[csum < V_lim_dep, :]))
#                 V_dep = np.vstack((V_dep_old[0:index, :], np.hstack((V_dep_old[index, 0], np.around(V_dep_old[index, 1:] * (1-perc_layer))))))


#     else:  # if the incoming sediment volume is enough to completely fill the active layer...

#         # ... deposit part of the incoming cascades
#         #    proportionally to their volume and the volume of the active layer,
#         #    and put the rest into the active layer

#         # percentage of the volume to put in the active layer for all the cascades
#         perc_dep = V_lim_tot_n / np.sum(Qbi_incoming[:, 1:])

#         if ~np.isnan(roundpar):
#             Qbi_incoming_dep = np.around(Qbi_incoming[:, 1:]*(1-perc_dep), decimals=roundpar)
#         else:
#             # this contains the fraction of the incoming volume to be deposited
#             Qbi_incoming_dep = Qbi_incoming[:, 1:]*(1-perc_dep)

#         V_inc2act = np.hstack((Qbi_incoming[:, 0][:,None], Qbi_incoming[:, 1:] - Qbi_incoming_dep))
#         V_dep2act = np.append(V_dep_old[0, 0], np.zeros((1, Qbi_incoming.shape[1]-1)))

#         if V_dep2act.ndim == 1:
#             V_dep2act = V_dep2act[None, :]

#         # if, given the round, the deposited volume of the incoming cascades is not 0...
#         if any(np.sum(Qbi_incoming[:, 1:]*(1-perc_dep), axis = 0)):
#             V_dep = np.vstack((V_dep_old, np.hstack((Qbi_incoming[:, 0][:,None], Qbi_incoming_dep))))
#         else:
#             V_dep = V_dep_old  # ... i leave the deposit as it was.

#     # remove empty rows (if the matrix is not already empty)
#     if (np.sum(V_dep2act[:, 1:], axis = 1)!=0).any():
#         V_dep2act = V_dep2act[np.sum(V_dep2act[:, 1:], axis = 1) != 0, :]

#     # find active layer GSD

#     # find the GSD of the active layer, for the transport capacity calculation
#     Fi_r_reach = (np.sum(V_dep2act[:, 1:], axis=0) + np.sum(V_inc2act[:, 1:], axis=0)) / (np.sum(V_dep2act[:, 1:]) + np.sum(V_inc2act[:, 1:]))
#     # if V_act is empty, i put Fi_r equal to 0 for all classes
#     Fi_r_reach[np.isinf(Fi_r_reach) | np.isnan(Fi_r_reach)] = 0


#     return V_inc2act, V_dep2act, V_dep, Fi_r_reach


# def matrix_compact(V_layer):
#     '''
#     '''

#     ID = np.unique(V_layer[:,0]) #, return_inverse=True
#     V_layer_cmpct = np.empty((len(ID), V_layer.shape[1]))
#     # sum elements with same ID
#     for ind, i in enumerate(ID):
#         vect = V_layer[V_layer[:,0] == i,:]
#         V_layer_cmpct[ind,:] = np.append(ID[ind], np.sum(vect[:,1:],axis = 0))

#     if V_layer_cmpct.shape[0]>1:
#         V_layer_cmpct = V_layer_cmpct[np.sum(V_layer_cmpct[:,1:], axis = 1)!=0]


#     if V_layer_cmpct.size == 0:
#         V_layer_cmpct = (np.hstack((ID[0], np.zeros((V_layer[:,1:].shape[1]))))).reshape(1,-1)

#     return V_layer_cmpct


# def tr_cap_deposit(V_inc2act, V_dep2act, V_dep, tr_cap, roundpar):
#     '''
#     '''
#     # V_dep and V_act identification
#     # classes for which the tr_cap is more than the incoming volumes in the active layer
#     class_sup_dep = tr_cap > np.sum(V_inc2act[:, 1:], axis=0)


#     # if there are sed classes for which the tr cap is more than the volume in V_inc2act...
#     if np.any(class_sup_dep):
#         # ...  sediment from V_dep2act will have to be mobilized, taking into consideration
#         # the sediment stratigraphy (upper layers get mobilized first)

#         # remaining active layer volume per class after considering V_inc2act
#         tr_cap_remaining = tr_cap[class_sup_dep] - np.sum(V_inc2act[:, np.append(False, class_sup_dep)], axis=0)
#         # take only the columns with the cascades of the classes class_sup_dep
#         V_dep2act_class = V_dep2act[:, np.append(False, class_sup_dep)]

#         csum = np.flipud(np.cumsum(np.flipud(V_dep2act_class), axis = 0))

#         # find the indexes of the first cascade above the tr_cap threshold, for each class
#         mapp =csum >= tr_cap_remaining

#         mapp[0, np.any(~mapp,axis = 0)] = True   # EB: force the first deposit layer to be true

#         # find position of the layer to be splitted between deposit and erosion
#         firstoverthresh = (mapp*1).argmin(axis=0)
#         firstoverthresh = firstoverthresh - 1
#         firstoverthresh[firstoverthresh == -1] = csum.shape[0]-1

#         mapfirst = np.zeros((mapp.shape))
#         mapfirst[firstoverthresh, np.arange(np.sum(class_sup_dep*1))] = 1

#         perc_dep = np.minimum((tr_cap_remaining - np.sum(np.where(mapp == False, V_dep2act_class, 0), axis=0))/V_dep2act_class[firstoverthresh, np.arange(np.sum(class_sup_dep*1))], 1)   # percentage to be lifted from the layer "on the threshold"

#         map_perc = mapfirst*perc_dep + ~mapp*1 # # EB check again  EB: is it adding 1 when true ?

#         # the matrix V_dep2act_new contains the mobilized cascades from the deposit layer, now corrected according to the tr_cap
#         V_dep2act_new = np.zeros((V_dep2act.shape))
#         V_dep2act_new[: , 0] = V_dep2act[: ,0]
#         V_dep2act_new[:,np.append(False, class_sup_dep)== True] = map_perc* V_dep2act_class

#         if ~np.isnan(roundpar):
#             V_dep2act_new[: , 1:]  = np.around(V_dep2act_new[: , 1:] , decimals = roundpar )

#         # the matrix V_2dep contains the cascades that will be deposited into the deposit layer.
#         # (the new volumes for the classes in class_sup_dep and all the volumes in the remaining classes)
#         V_2dep = np.zeros((V_dep2act.shape))
#         V_2dep[: , np.append(True, ~class_sup_dep) == True] = V_dep2act[: , np.append(True, ~class_sup_dep) == True]
#         V_2dep[: , np.append(False, class_sup_dep) == True] = (1 - map_perc)* V_dep2act_class

#         if ~np.isnan(roundpar):
#             V_2dep[: , 1: ]  = np.around(V_2dep[: ,1:] , decimals = roundpar )

#     else:
#         V_dep2act_new = np.zeros((V_dep2act.shape))
#         V_dep2act_new[0] = 0 # EB:0 because it should be the row index (check whether should be 1)
#         V_2dep = V_dep2act
#         # I re-deposit all the matrix V_dep2act into the deposit layer

#     # for the classes where V_inc2act is enough, I deposit the cascades
#     # proportionally

#     perc_inc = tr_cap[~class_sup_dep] / np.sum(V_inc2act[: , np.append(False, ~class_sup_dep) == True], axis = 0)
#     perc_inc[np.isnan(perc_inc)] = 0 #change NaN to 0 (naN appears when both tr_cap and sum(V_inc2act) are 0)
#     class_perc_inc = np.zeros((class_sup_dep.shape))
#     class_perc_inc[class_sup_dep == False] = perc_inc

#     V_mob = matrix_compact(np.vstack((V_dep2act_new, V_inc2act*(np.append(True,class_sup_dep)) + V_inc2act*np.append(False, class_perc_inc))))

#     if ~np.isnan( roundpar ):
#         V_mob[:,1:] = np.around( V_mob[:,1:] , decimals =roundpar )

#     class_residual = np.zeros((class_sup_dep.shape));
#     class_residual[class_sup_dep==False] = 1 - perc_inc

#     V_2dep = np.vstack((V_2dep, V_inc2act*np.hstack((1, class_residual)))) ## EB check again EB: here the 1 instead of the 0 should be correct +

#     if ~np.isnan( roundpar ):
#         V_2dep[:,1:]  = np.around( V_2dep[:,1:] , decimals = roundpar)

#     # Put the volume exceeding the transport capacity back in the deposit

#     #If the upper layer in the deposit and the lower layer in the volume to be
#     #deposited are from the same reach, i sum them
#     if (V_dep[-1,0] == V_2dep[0,0]):
#         V_dep[-1,1:] = V_dep[-1,1:] + V_2dep[0,1:]
#         V_dep = np.vstack((V_dep, V_2dep[1:,:]))
#     else:
#         V_dep = np.vstack((V_dep, V_2dep))


#     #remove empty rows
#     if not np.sum(V_dep2act[:,1:])==0:
#         V_dep = V_dep[np.sum(V_dep[:,1:],axis = 1)!=0]

#     return V_mob, V_dep



# def change_slope(Node_el_t, Lngt, Network , **kwargs):
#     """"CHANGE_SLOPE modify the Slope vector according to the changing elevation of
#     the nodes: It also guarantees that the slope is not negative or lower then
#     the min_slope value by changing the node elevation bofore findin the SLlpe"""

#     #define minimum reach slope


#     #initialization
#     if len(kwargs) != 0:
#         min_slope = kwargs['s']
#     else:
#         min_slope = 0

#     outlet = Network['n_hier'][-1]
#     down_node = Network['downstream_node']
#     down_node = np.array([int(n) for n in down_node])
#     down_node[int(outlet)] = (len(Node_el_t)-1)

#     Slope_t = np.zeros(Lngt.shape)

#     #loop for all reaches
#     for n in range(len(Lngt)):
#         #find the minimum node elevation to guarantee Slope > min_slope
#         min_node_el = min_slope * Lngt[n] + Node_el_t[down_node[n]]

#         #change the noide elevation if lower to min_node_el
#         Node_el_t[n] = np.maximum(min_node_el, Node_el_t[n] )

#         #find the new slope
#         Slope_t[n] = (Node_el_t[n] - Node_el_t[down_node[n]]) / Lngt[n]


#     return Slope_t, Node_el_t






# def cascades_end_time_or_not(cascade_list_old, reach_length, ts_length):
#     ''' Fonction to decide if the traveling cascades in cascade list stop in
#     the reach or not, due to the end of the time step.
#     Inputs:
#         cascade_list_old:    list of traveling cascades
#         reach_length:           reach physical length
#         ts_length:              time step length

#     Return:
#         cascade_list_new:       same cascade list updated. Stopping cascades or
#                                 partial volumes have been removed

#         depositing_volume:      the volume to be deposited in this reach.
#                                 They are ordered according to their arrival time
#                                 at the inlet, so that volume arriving first
#                                 deposit first.
#     '''
#     # Order cascades according to their arrival time, so that first arriving
#     # cascade are first in the loop and are deposited first
#     # Note: in the deposit layer matrix, first rows are the bottom layers
#     cascade_list_old = sorted(cascade_list_old, key=lambda x: np.mean(x.elapsed_time))

#     depositing_volume_list = []
#     cascades_to_be_completely_removed = []

#     for cascade in cascade_list_old:
#         # Time in, time travel, and time out in time step unit (not seconds)
#         t_in = cascade.elapsed_time
#         t_travel_n = reach_length / (cascade.velocities * ts_length)
#         t_out = t_in + t_travel_n
#         # Vm_stop is the stopping part of the cascade volume
#         # Vm_continue is the continuing part
#         Vm_stop, Vm_continue = stop_or_not(t_out, cascade.volume)

#         if Vm_stop is not None:
#             depositing_volume_list.append(Vm_stop)

#             if Vm_continue is None:
#                 # no part of the volume continues, we remove the entire cascade
#                 cascades_to_be_completely_removed.append(cascade)
#             else:
#                 # some part of the volume continues, we update the volume
#                 cascade.volume = Vm_continue

#         if Vm_continue is not None:
#             # update time for continuing cascades
#             cascade.elapsed_time = t_out
#             # put to 0 the elapsed time of the empty sediment classes
#             # i.e. the classes that have deposited, while other did not
#             # (Necessary for the time lag calculation later in the code)
#             cond_0 = np.all(cascade.volume[:,1:] == 0, axis = 0)
#             cascade.elapsed_time[cond_0] = 0


#     # If they are, remove complete cascades:
#     cascade_list_new = [casc for casc in cascade_list_old if casc not in cascades_to_be_completely_removed]

#     # If they are, concatenate the deposited volumes
#     if depositing_volume_list != []:
#         depositing_volume = np.concatenate(depositing_volume_list, axis=0)
#         if np.all(depositing_volume[:,1:] == 0):
#             raise ValueError("DD check: we have an empty layer stopping ?")
#     else:
#         depositing_volume = None

#     return cascade_list_new, depositing_volume



# def stop_or_not(t_new, Vm):
#     '''
#     Function that decides if a volume of sediments will stop in this
#     reach or not, depending on time. Part of the volume can stop or continue.

#     t_new: elapsed time since beginning of time step for Vm, for each sed class
#     Vm: traveling volume of sediments
#     '''
#     cond_stop = np.insert([t_new>1], 0, True)
#     Vm_stop = np.zeros_like(Vm)
#     Vm_stop[:, cond_stop] = Vm[:, cond_stop]

#     cond_continue = np.insert([t_new<=1], 0, True)
#     Vm_continue = np.zeros_like(Vm)
#     Vm_continue[:, cond_continue] = Vm[:, cond_continue]

#     if np.all(Vm_stop[:,1:] == 0) == True:
#         Vm_stop = None
#     if np.all(Vm_continue[:,1:] == 0) == True:
#         Vm_continue = None

#     return Vm_stop, Vm_continue


# def deposit_from_passing_sediments(V_remove, cascade_list, roundpar):
#     ''' This function remove the quantity V_remove from the list of cascades.
#     The order in which we take the cascade is from largest times (arriving later)
#     to shortest times (arriving first). Hypotheticaly, cascade arriving first
#     are passing in priority, in turn, cascades arriving later are deposited in priority.
#     If two cascades have the same time, they are processed as one same cascade.

#     INPUTS:
#     V_remove : quantity to remove, per sediment class (array of size number of sediment classes).
#     cascade_list : list of cascades. Reminder, a cascade is a Cascade class with attributes:
#                     direct provenance, elapsed time, and the volume
#     roundpar : number of decimals to round the cascade volumes (Vm)
#     RETURN:
#     r_Vmob : removed volume from cascade list
#     cascade_list : the new cascade list, after removing the volumes
#     V_remove : residual volume to remove
#     '''
#     removed_Vm_all = []

#     # Order cascades according to the inverse of their elapsed time
#     # and put cascade with same time in a sublist, in order to treat them together
#     sorted_cascade_list = sorted(cascade_list, key=lambda x: np.sum(x.elapsed_time), reverse=True)
#     sorted_and_grouped_cascade_list = [list(group) for _, group in groupby(sorted_cascade_list, key=lambda x: np.sum(x.elapsed_time))]

#     # Loop over the sorted and grouped cascades
#     for cascades in sorted_and_grouped_cascade_list:
#         Vm_same_time = np.concatenate([casc.volume for casc in cascades], axis=0)
#         if np.any(Vm_same_time[:,1:]) == False: #In case Vm_same_time is full of 0
#             del cascades
#             continue
#         # Storing matrix for removed volumes
#         removed_Vm = np.zeros_like(Vm_same_time)
#         removed_Vm[:,0] = Vm_same_time[:,0] # same first col with initial provenance
#         for col_idx in range(Vm_same_time[:,1:].shape[1]):  # Loop over sediment classes
#             if V_remove[col_idx] > 0:
#                 col_sum = np.sum(Vm_same_time[:, col_idx+1])
#                 if col_sum > 0:
#                     fraction_to_remove = min(V_remove[col_idx] / col_sum, 1.0)
#                     # Subtract the fraction_to_remove from the input cascades objects (to modify them directly)
#                     for casc in cascades:
#                         Vm = casc.volume
#                         removed_quantities = Vm[:, col_idx+1] * fraction_to_remove
#                         Vm[:, col_idx+1] -= removed_quantities
#                         # Round Vm
#                         Vm[:, col_idx+1] = np.round(Vm[:, col_idx+1], decimals = roundpar)
#                         # Ensure no negative values
#                         if np.any(Vm[:, col_idx+1] < -10**(-roundpar)) == True:
#                             raise ValueError("Negative value in VM is strange")

#                     # Store the removed quantities in the removed volumes matrix
#                     removed_Vm[:, col_idx+1] = Vm_same_time[:, col_idx+1] * fraction_to_remove
#                     # Update V_remove by subtracting the total removed quantity
#                     V_remove[col_idx] -= col_sum * fraction_to_remove
#                     # Ensure V_remove doesn't go under the number fixed by roundpar
#                     if np.any(V_remove[col_idx] < -10**(-roundpar)) == True:
#                         raise ValueError("Negative value in V_remove is strange")
#         # Round and store removed volumes
#         removed_Vm[:, 1:] = np.round(removed_Vm[:, 1:], decimals = roundpar)
#         removed_Vm_all.append(removed_Vm)
#     # Concatenate all removed quantities into a single matrix
#     r_Vmob = np.vstack(removed_Vm_all) if removed_Vm_all else np.array([])
#     # Gather layers of same original provenance in r_Vmob
#     r_Vmob = matrix_compact(r_Vmob)

#     # Delete cascades that are now only 0 in input cascade list
#     cascade_list = [cascade for cascade in cascade_list if not np.all(cascade.volume[:, 1:] == 0)]

#     # The returned cascade_list is directly modified by the operations on Vm
#     return r_Vmob, cascade_list, V_remove




# def compute_time_lag(cascade_list, n_classes, compare_with_tr_cap, time_lag_for_mobilised):
#     ''' The time lag is the time we use to mobilise from the reach,
#     before cascades from upstream reaches arrive at the outlet of the present reach.
#     We take it as the time for the first cascade to arrive at the outet.
#     Depending on the algorithm options,

#     cascade_list            : the list of cascade objects. Can be empty.
#     compare_with_tr_cap     : bool for the option if we conpare with tr_cap.
#     time_lag_for_mobilised  : bool for the option if we include a time lag.
#     '''

#     if compare_with_tr_cap == True:
#         if time_lag_for_mobilised == True:
#             if cascade_list == []:
#                 time_lag = np.ones(n_classes) # the time lag is the entire time step as no other cascade reach the outlet
#             else:
#                 time_arrays = np.array([cascade.elapsed_time for cascade in cascade_list])
#                 time_lag = np.min(time_arrays, axis=0)
#         else:
#             # in this condition (we compare with tr cap at the outlet,
#             # but no time lag is considered), we don't mobilised from the
#             # reach before the possible cascades arrive.
#             # At the exception that no cascades arrive at the outlet.
#             if cascade_list != []:
#                 time_lag = np.zeros(n_classes)
#             else:
#                 # If no cascades arrive at the outlet,
#                 # we mobilise from the reach itself
#                 time_lag = np.ones(n_classes)
#     else:
#         # in this condition (compare_with_tr_cap = False),
#         # we always mobilise from the reach itself and
#         # the passing cascades are passing the outlet, without
#         # checking the energy available to make them pass,
#         # like in version 1 of the code
#         time_lag = np.ones(n_classes)

#     return time_lag









# OLD version definitions:

# def track_sed_position( n , v_sed_day , Lngt , psi, Network ,  **kwargs):

#     """TRACK_SED_POSITION_TRCAP finds the position of a sediment parcel starting
#     from reach n after the timestep has passed, defined as the reach ID and
#     the position from the From_node of the starting reach.
#     To satisfy the transport capacity in the ToNode section, the starting
#     position of the volume is positioned in a location that guarantees that
#     all of it passes through the ToNode and leaves the reach """

#     ## define starting position
#     #the starting position is the position on the reach n from which the
#     #parcel start, defined as fraction of reach length
#     #if start_pos = 0, the parcel starts form the From_Node
#     #if start_pos = 1, the parcel starts form the To_Node

#     if len(kwargs) ==0:
#         start_pos = 0

#     ## find path downstream

#     #start_pos (between 0 and 1) defines the position in the reach n where the
#     #sediment parcel start, if 1 start form the From_node, if 0 starts from
#     #the To_Node
#     #if nargin < 5
#     #  start_pos = 1;
#     #end

#     timestep = 1
#     outlet = int(Network['outlet'])

#     #path2out contains the path from reach n to the outlet, defined as the IDs of the
#     #reach downstream ordered.
#     #downdist_path contains the distance from the reaches in path2out to the
#     #reach n From_node
#     path2out = np.array([int(m) for m in Network['downstream_path'][str(n)][str(outlet)]])
#     downdist_path = Network['downstream_distance'][n][path2out]

#     # find position and destination reach ID

#     #isolate the velocity of the downstream reaches
#     v_sed_path = v_sed_day[:,path2out]

#     if v_sed_path.ndim == 1:
#         v_sed_path = v_sed_path[:,None]

#     #change the length of the starting reach according to the starting
#     #position, different for each tr.cap
#     Lngt_pathout = np.repeat(np.array(Lngt[path2out]).reshape(1,-1),len(psi), axis=0)
#     Lngt_pathout[:,0]  = Lngt_pathout[:,0] * (1 - start_pos)

#     #calculate the time (in days) it takes to completely cross a reach
#     transit_time = Lngt_pathout/v_sed_path

#     # the cumulate of transit_time defines how long it takes to reach each
#     # downstream To_Node comnsidering the whole path to the reach
#     cum_tr_time = np.cumsum(transit_time,axis=1)

#     # given cum_tr_time, i can find the reach where the parcel is after the timestep
#     find_point = cum_tr_time - timestep
#     find_point[find_point<0] = 100000000000000000 # EB: maybe find a more elegant solution
#     indx_pos = np.argmin(find_point, axis=1) # (note, this is not the reach ID, it is the position of the reach in the downstream path)
#     indx_pos[find_point[:,-1] == 100000000000000000] = len(path2out)-1 # EB check if len + 1 #if the whole row in find_point is nan, it means that the parcel left the network

#     # I can find the time remaining for the parcel after if enters reach
#     # indx_pos, needed to find the position of the parcel
#     find_time = timestep - cum_tr_time
#     find_time[find_time<0] = 100000000000000000
#     indx_t = np.argmin(find_time,axis =1) #indx_t is the reach before indx_pos

#     time_left = find_time[np.arange(len(find_time)),indx_t] # EB: check whether find_time has two dims also with one column
#     time_left[time_left == 100000000000000000] = timestep #if time_left is nan, it means that the parcel remained in the starting reach m


#     sed_pos = time_left * v_sed_path[np.arange(len(psi)), indx_pos] + downdist_path[indx_pos]
#     #If the whole row in find_point is nan (the parcel left the network),
#     #use the velocity of the outlet to determine the final position
#     # (that will be outside the network)
#     if (sed_pos[(find_point[:,-1] == 100000000000000000)]).size != 0:
#             sed_pos[(find_point[:,-1]== 100000000000000000)] = downdist_path[len(path2out)-1] + Lngt[outlet] +  v_sed_path[(find_point[:,-1]== 100000000000000000), len(path2out)-1] * time_left[(find_point[:,-1]== 100000000000000000)]

#     #outind tells for which sed. size the point fell outside the
#     # network (1 - outside, 0 - inside)
#     outind = (find_point[:,-1] == 100000000000000000)

#     #sed_pos = sed_pos + Lngt(n) * (1 - start_pos);
#     end_reach_ID = path2out[indx_pos] # i find the ID of the destination reach from indx_pos, given the order defined by path2out

#     return sed_pos , end_reach_ID, outind

# def sed_transfer_simple(V_mob, n, v_sed_day, Lngt, Network, psi):
#     """SED_TRANSFER_SIMPLE takes the matrix of the mobilized layers(V_mob) and the vector of
#     the sed velocity for each class(v_sed_id) in a reach (n) and returns the 3D matrices containing the
#     sed volumes in V_mob in their new position at the end of the timestep.
#     This simple version of the function represents the volume as a point
#     sediment parcel delivered from the ToN of the reach n. Thus the volume
#     has a single destination reach and it never get split. """

#     ##initailize parameters
#     outlet = Network['n_hier'][-1]

#     ## find start and end reach of the sed volume after the timestep

#     # reach_dest is the id of the reach where the sed. volume stops after the timestep
#     #p_dest is the position from the from_node of the id reach where the sed. volume stops after the timestep

#     if n == outlet:
#         reach_dest = np.repeat(n , np.shape(v_sed_day)[0])
#         p_dest = v_sed_day[:,n] + np.array(Lngt[n])
#     else:
#         #to find p_end, i track the position of a sediment parcel starting
#         #from the To_Node of the reach n (i.e. the From_node of the downstream reach).
#         p_dest, reach_dest, outind = track_sed_position(int(Network['downstream_node'][int(n)]), v_sed_day, Lngt, psi, Network)
#         p_dest = p_dest + np.array(Lngt[n])

#     #downdist contains the distanche from the starting reach to all reaches
#     #downstream
#     downdist = Network['downstream_distance'][int(n)]

#     ## find position of the sediment volume

#     #setout is equal to 1 if the volume in the sed.class left the network
#     #via the outlet
#     setout = (np.squeeze(p_dest) - np.array(Lngt[reach_dest]) - downdist[reach_dest].T> 0)*1

#     #in each row, setplace is equal to 1 in the reach where the sed. volume
#     #of each class is delivered
#     setplace = np.zeros((len(v_sed_day), len(downdist)))
#     setplace[np.arange(len(v_sed_day)), reach_dest]  = 1



#     setplace[setout==1,:] = 0

#     ## place volume to destination reach

#     Qbi_tr_t = np.zeros((len(Lngt), len(Lngt) , len(setplace)))
#     Q_out_t = np.zeros ((len(Lngt), len(setplace)))

#     for c in range(len(setplace)):
#         Qbi_tr_t[[V_mob[:,0].astype(int)],:,c] = V_mob[:,c+1][:,None] * setplace[c,:][None,:]
#         Q_out_t[[V_mob[:,0].astype(int)],:] = V_mob[:,1:] * setout

#     return Qbi_tr_t, Q_out_t , setplace, setout







