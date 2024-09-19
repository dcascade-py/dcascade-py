# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:56:59 2022

This file contains the functions:
    
    - D_finder: finds the value of granulometry for the specified D_values for the sediment distribution Fi_r
    input: fi_r - grain size distribution of the active layer for a specific reach. In each column there is the % of the sediment class represented by that column (see psi)
           D_value = the quantile (eg. 50 for D50)

    - sortdistance: sorts the rows of Qbi_incoming matrix by increasing distance from the reach
    
    - layer_search: puts part of the incoming and deposited sediment volumes into the
      active layer according to the active layer volume and the incoming and deposited volumes
      
    - matrix_compact: takes a stratigraphy matrix V_layer and compact it by summing all the layers with the same source reach id
      
    - tr_cap_deposit: deposits part of the incoming and deposited sediment volumes
      according to the transport capacity calculated  
    

           
This script was adapted from the Matlab version by Marco Tangi            
@author: Elisa Bozzolan 
"""
import numpy as np
import pandas as pd

# ignore divide by 0 
np.seterr(divide='ignore', invalid='ignore')


def D_finder(fi_r, D_values, psi): 

    dmi = np.power(2, -psi)/1000
    if dmi.size == 1:
        return dmi[0]
    
    else:
        if fi_r.ndim == 1:
            fi_r = fi_r[:, None] # EB: needs to be a column vector
            
        D_changes = np.zeros((1, np.shape(fi_r)[1]))
        Perc_finer = np.empty((len(dmi),np.shape(fi_r)[1]))
        Perc_finer[:] = np.nan
        Perc_finer[0] = 100
    
        for i in range(1, len(Perc_finer)):
            Perc_finer[i,:] = Perc_finer[i-1,:] - fi_r[i-1,:]*100
        
        for k in range(np.shape(Perc_finer)[1]):
            a = np.minimum(np.where(Perc_finer[:,k] > D_values)[0].max(), len(psi)-2)
            D_changes[0,k] = (D_values - Perc_finer[a+1,k])/(Perc_finer[a,k] -Perc_finer[a+1,k])*(-psi[a]+psi[a+1])-psi[a+1]
            D_changes[0,k] = np.power(2, D_changes[0,k])/1000
            D_changes[0,k] = D_changes[0,k]*(D_changes[0,k]>0) + dmi[-1]*(D_changes[0,k]<0)
        return D_changes


def sortdistance(Qbi, distancelist):

    idx = np.argwhere(Qbi[:, 0][:,None] == distancelist[~(np.isnan(distancelist))])[:,1]

    if idx.size != 0 and len(idx) != 1:  # if there is a match #EB check
        Qbi_sort = np.array(Qbi[(idx-1).argsort(), :])           
    else:
        Qbi_sort = Qbi

    return Qbi_sort


def layer_search(Qbi_incoming, V_dep_old , V_lim_tot_n, roundpar):

    # if, considering the incoming volume, I am still under the threshold of the active layer volume...
    if (V_lim_tot_n - np.sum(Qbi_incoming[:, 1:])) > 0:

        # ... I put sediment from the deposit layer into the active layer
        # remaining active layer volume after considering incoming sediment cascades
        V_lim_dep = V_lim_tot_n - np.sum(Qbi_incoming[:, 1:])
        csum = np.flipud(np.cumsum(np.flipud(np.sum(V_dep_old[:, 1:], axis=1)), axis = 0)) # EB check again 

        V_inc2act = Qbi_incoming  # all the incoming volume will end up in the active layer

        # find active layer
         
        if (np.argwhere(csum > V_lim_dep)).size == 0 :  # the vector is empty # EB check again 
            # if the cascades in the deposit have combined
            # volume that is less then the active layer volume (i've reached the bottom)
            
            print(' reach the bottom ....')

            V_dep2act = V_dep_old  # I put all the deposit into the active layer
            V_dep = np.c_[V_dep_old[0,0], np.zeros((1,Qbi_incoming.shape[1]-1))]


        else:
            
            index = np.max(np.argwhere(csum >= V_lim_dep))


            # if i have multiple deposit layers, put the upper layers into the active layer until i reach the threshold.
            # The layer on the threshold (defined by position index) gets divided according to perc_layer
            perc_layer = (V_lim_dep - np.sum(V_dep_old[csum < V_lim_dep, 1:]))/sum(V_dep_old[index, 1:])  # EB check again  # percentage to be lifted from the layer on the threshold 

            # remove small negative values that can arise from the difference being very close to 0
            perc_layer = np.maximum(0, perc_layer)

            if ~np.isnan(roundpar):
                V_dep2act = np.vstack((np.hstack((V_dep_old[index, 0], np.around(V_dep_old[index, 1:]*perc_layer, decimals=roundpar))).reshape(1, -1), V_dep_old[csum<V_lim_dep,:]))
                V_dep = np.vstack((V_dep_old[0:index,:], np.hstack((V_dep_old[index,0], np.around(V_dep_old[index,1:]* (1-perc_layer), decimals=roundpar)))))
            else: 
                V_dep2act = np.vstack((np.hstack((V_dep_old[index, 0], np.around( V_dep_old[index, 1:]*perc_layer))).reshape(1, -1), V_dep_old[csum < V_lim_dep, :]))
                V_dep = np.vstack((V_dep_old[0:index, :], np.hstack((V_dep_old[index, 0], np.around(V_dep_old[index, 1:] * (1-perc_layer))))))
    

    else:  # if the incoming sediment volume is enough to completely fill the active layer...

        # ... deposit part of the incoming cascades
        #    proportionally to their volume and the volume of the active layer,
        #    and put the rest into the active layer

        # percentage of the volume to put in the active layer for all the cascades
        perc_dep = V_lim_tot_n / np.sum(Qbi_incoming[:, 1:])

        if ~np.isnan(roundpar):
            Qbi_incoming_dep = np.around(Qbi_incoming[:, 1:]*(1-perc_dep), decimals=roundpar)
        else:
            # this contains the fraction of the incoming volume to be deposited
            Qbi_incoming_dep = Qbi_incoming[:, 1:]*(1-perc_dep)

        V_inc2act = np.hstack((Qbi_incoming[:, 0][:,None], Qbi_incoming[:, 1:] - Qbi_incoming_dep))
        V_dep2act = np.append(V_dep_old[0, 0], np.zeros((1, Qbi_incoming.shape[1]-1)))
        
        if V_dep2act.ndim == 1: 
            V_dep2act = V_dep2act[None, :]

        # if, given the round, the deposited volume of the incoming cascades is not 0...
        if any(np.sum(Qbi_incoming[:, 1:]*(1-perc_dep), axis = 0)):
            V_dep = np.vstack((V_dep_old, np.hstack((Qbi_incoming[:, 0][:,None], Qbi_incoming_dep))))
        else:
            V_dep = V_dep_old  # ... i leave the deposit as it was.

    # remove empty rows (if the matrix is not already empty)
    if (np.sum(V_dep2act[:, 1:], axis = 1)!=0).any():       
        V_dep2act = V_dep2act[np.sum(V_dep2act[:, 1:], axis = 1) != 0, :]

    # find active layer GSD

    # find the GSD of the active layer, for the transport capacity calculation
    Fi_r_reach = (np.sum(V_dep2act[:, 1:], axis=0) + np.sum(V_inc2act[:, 1:], axis=0)) / (np.sum(V_dep2act[:, 1:]) + np.sum(V_inc2act[:, 1:]))
    # if V_act is empty, i put Fi_r equal to 0 for all classes
    Fi_r_reach[np.isinf(Fi_r_reach) | np.isnan(Fi_r_reach)] = 0
    

    return V_inc2act, V_dep2act, V_dep, Fi_r_reach

def matrix_compact(V_layer):
    
    ID = np.unique(V_layer[:,0]) #, return_inverse=True
    V_layer_cmpct = np.empty((len(ID), V_layer.shape[1]))
    # sum elements with same ID 
    for ind, i in enumerate(ID): 
        vect = V_layer[V_layer[:,0] == i,:]
        V_layer_cmpct[ind,:] = np.append(ID[ind], np.sum(vect[:,1:],axis = 0))
    
    if V_layer_cmpct.shape[0]>1: 
        V_layer_cmpct = V_layer_cmpct[np.sum(V_layer_cmpct[:,1:], axis = 1)!=0]


    if V_layer_cmpct.size == 0: 
        V_layer_cmpct = (np.hstack((ID[0], np.zeros((V_layer[:,1:].shape[1]))))).reshape(1,-1)
    
    return V_layer_cmpct


def tr_cap_deposit(V_inc2act, V_dep2act, V_dep, tr_cap, roundpar):
    # V_dep and V_act identification
    # classes for which the tr_cap is more than the incoming volumes in the active layer
    class_sup_dep = tr_cap > np.sum(V_inc2act[:, 1:], axis=0)
    

    # if there are sed classes for which the tr cap is more than the volume in V_inc2act...
    if np.any(class_sup_dep):
        # ...  sediment from V_dep2act will have to be mobilized, taking into consideration
        # the sediment stratigraphy (upper layers get mobilized first)

        # remaining active layer volume per class after considering V_inc2act
        tr_cap_remaining = tr_cap[class_sup_dep] - np.sum(V_inc2act[:, np.append(False, class_sup_dep)], axis=0)
        # take only the columns with the cascades of the classes class_sup_dep
        V_dep2act_class = V_dep2act[:, np.append(False, class_sup_dep)]

        csum = np.flipud(np.cumsum(np.flipud(V_dep2act_class), axis = 0)) 

        # find the indexes of the first cascade above the tr_cap threshold, for each class
        mapp =csum > tr_cap_remaining  

        mapp[0, np.any(~mapp,axis = 0)] = True   # EB: force the first deposit layer to be true 

        # find position of the layer to be splitted between deposit and erosion
        firstoverthresh = (mapp*1).argmin(axis=0)
        firstoverthresh = firstoverthresh - 1
        firstoverthresh[firstoverthresh == -1] = csum.shape[0]-1

        mapfirst = np.zeros((mapp.shape))
        mapfirst[firstoverthresh, np.arange(np.sum(class_sup_dep*1))] = 1 

        perc_dep = np.minimum((tr_cap_remaining - np.sum(np.where(mapp == False, V_dep2act_class, 0), axis=0))/V_dep2act_class[firstoverthresh, np.arange(np.sum(class_sup_dep*1))], 1)   # percentage to be lifted from the layer "on the threshold"

        map_perc = mapfirst*perc_dep + ~mapp*1 # # EB check again  EB: is it adding 1 when true ? 

        # the matrix V_dep2act_new contains the mobilized cascades from the deposit layer, now corrected according to the tr_cap
        V_dep2act_new = np.zeros((V_dep2act.shape))
        V_dep2act_new[: , 0] = V_dep2act[: ,0]
        V_dep2act_new[:,np.append(False, class_sup_dep)== True] = map_perc* V_dep2act_class

        if ~np.isnan(roundpar): 
            V_dep2act_new[: , 1:]  = np.around(V_dep2act_new[: , 1:] , decimals = roundpar )

        # the matrix V_2dep contains the cascades that will be deposited into the deposit layer.
        # (the new volumes for the classes in class_sup_dep and all the volumes in the remaining classes)
        V_2dep = np.zeros((V_dep2act.shape))
        V_2dep[: , np.append(True, ~class_sup_dep) == True] = V_dep2act[: , np.append(True, ~class_sup_dep) == True]
        V_2dep[: , np.append(False, class_sup_dep) == True] = (1 - map_perc)* V_dep2act_class

        if ~np.isnan(roundpar): 
            V_2dep[: , 1: ]  = np.around(V_2dep[: ,1:] , decimals = roundpar )

    else:
        V_dep2act_new = np.zeros((V_dep2act.shape))
        V_dep2act_new[0] = 0 # EB:0 because it should be the row index (check whether should be 1)
        V_2dep = V_dep2act
        # I re-deposit all the matrix V_dep2act into the deposit layer

    # for the classes where V_inc2act is enough, I deposit the cascades
    # proportionally

    perc_inc = tr_cap[~class_sup_dep] / np.sum(V_inc2act[: , np.append(False, ~class_sup_dep) == True], axis = 0)
    perc_inc[np.isnan(perc_inc)] = 0 #change NaN to 0 (naN appears when both tr_cap and sum(V_inc2act) are 0)
    class_perc_inc = np.zeros((class_sup_dep.shape))
    class_perc_inc[class_sup_dep == False] = perc_inc

    V_mob = matrix_compact(np.vstack((V_dep2act_new, V_inc2act*(np.append(True,class_sup_dep)) + V_inc2act*np.append(False, class_perc_inc))))
    
    if ~np.isnan( roundpar ):
        V_mob[:,1:] = np.around( V_mob[:,1:] , decimals =roundpar )

    class_residual = np.zeros((class_sup_dep.shape));
    class_residual[class_sup_dep==False] = 1 - perc_inc

    V_2dep = np.vstack((V_2dep, V_inc2act*np.hstack((1, class_residual)))) ## EB check again EB: here the 1 instead of the 0 should be correct + 
   
    if ~np.isnan( roundpar ):
        V_2dep[:,1:]  = np.around( V_2dep[:,1:] , decimals = roundpar) 

    # Put the volume exceeding the transport capacity back in the deposit

    #If the upper layer in the deposit and the lower layer in the volume to be
    #deposited are from the same reach, i sum them
    if (V_dep[-1,0] == V_2dep[0,0]):
        V_dep[-1,1:] = V_dep[-1,1:] + V_2dep[0,1:] 
        V_dep = np.vstack((V_dep, V_2dep[1:,:]))
    else:
        V_dep = np.vstack((V_dep, V_2dep))

    
    #remove empty rows
    if not np.sum(V_dep2act[:,1:])==0:
        V_dep = V_dep[np.sum(V_dep[:,1:],axis = 1)!=0]  
    
    return V_mob, V_dep

def track_sed_position( n , v_sed_day , Lngt , psi, Network ,  **kwargs):
    
    """TRACK_SED_POSITION_TRCAP finds the position of a sediment parcel starting
    from reach n after the timestep has passed, defined as the reach ID and
    the position from the From_node of the starting reach.
    To satisfy the transport capacity in the ToNode section, the starting
    position of the volume is positioned in a location that guarantees that
    all of it passes through the ToNode and leaves the reach """
    
    ## define starting position
    #the starting position is the position on the reach n from which the
    #parcel start, defined as fraction of reach length
    #if start_pos = 0, the parcel starts form the From_Node
    #if start_pos = 1, the parcel starts form the To_Node

    if len(kwargs) ==0:
        start_pos = 0
        
    ## find path downstream

    #start_pos (between 0 and 1) defines the position in the reach n where the 
    #sediment parcel start, if 1 start form the From_node, if 0 starts from
    #the To_Node
    #if nargin < 5
    #  start_pos = 1;
    #end

    timestep = 1
    outlet = int(Network['outlet'])
    
    #path2out contains the path from reach n to the outlet, defined as the IDs of the
    #reach downstream ordered.
    #downdist_path contains the distance from the reaches in path2out to the
    #reach n From_node
    path2out = np.array([int(m) for m in Network['downstream_path'][str(n)][str(outlet)]])
    downdist_path = Network['downstream_distance'][n][path2out]
    
    # find position and destination reach ID
    
    #isolate the velocity of the downstream reaches
    v_sed_path = v_sed_day[:,path2out]
    
    if v_sed_path.ndim == 1:
        v_sed_path = v_sed_path[:,None]
    
    #change the length of the starting reach according to the starting
    #position, different for each tr.cap
    Lngt_pathout = np.repeat(np.array(Lngt[path2out]).reshape(1,-1),len(psi), axis=0)
    Lngt_pathout[:,0]  = Lngt_pathout[:,0] * (1 - start_pos) 
    
    #calculate the time (in days) it takes to completely cross a reach 
    transit_time = Lngt_pathout/v_sed_path
    
    # the cumulate of transit_time defines how long it takes to reach each
    # downstream To_Node comnsidering the whole path to the reach
    cum_tr_time = np.cumsum(transit_time,axis=1)  
    
    # given cum_tr_time, i can find the reach where the parcel is after the timestep  
    find_point = cum_tr_time - timestep 
    find_point[find_point<0] = 100000000000000000 # EB: maybe find a more elegant solution 
    indx_pos = np.argmin(find_point, axis=1) # (note, this is not the reach ID, it is the position of the reach in the downstream path) 
    indx_pos[find_point[:,-1] == 100000000000000000] = len(path2out)-1 # EB check if len + 1 #if the whole row in find_point is nan, it means that the parcel left the network 

    # I can find the time remaining for the parcel after if enters reach
    # indx_pos, needed to find the position of the parcel 
    find_time = timestep - cum_tr_time
    find_time[find_time<0] = 100000000000000000
    indx_t = np.argmin(find_time,axis =1) #indx_t is the reach before indx_pos 
       
    time_left = find_time[np.arange(len(find_time)),indx_t] # EB: check whether find_time has two dims also with one column
    time_left[time_left == 100000000000000000] = timestep #if time_left is nan, it means that the parcel remained in the starting reach m


    sed_pos = time_left * v_sed_path[np.arange(len(psi)), indx_pos] + downdist_path[indx_pos] 
    #If the whole row in find_point is nan (the parcel left the network), 
    #use the velocity of the outlet to determine the final position
    # (that will be outside the network) 
    if (sed_pos[(find_point[:,-1] == 100000000000000000)]).size != 0: 
            sed_pos[(find_point[:,-1]== 100000000000000000)] = downdist_path[len(path2out)-1] + Lngt[outlet] +  v_sed_path[(find_point[:,-1]== 100000000000000000), len(path2out)-1] * time_left[(find_point[:,-1]== 100000000000000000)]
    
    #outind tells for which sed. size the point fell outside the
    # network (1 - outside, 0 - inside) 
    outind = (find_point[:,-1] == 100000000000000000)
    
    #sed_pos = sed_pos + Lngt(n) * (1 - start_pos); 
    end_reach_ID = path2out[indx_pos] # i find the ID of the destination reach from indx_pos, given the order defined by path2out 
    
    return sed_pos , end_reach_ID, outind

def sed_transfer_simple(V_mob , n , v_sed_day , Lngt, Network, psi):
    """SED_TRANSFER_SIMPLE takes the matrix of the mobilized layers(V_mob) and the vector of
    the sed velocity for each class(v_sed_id) in a reach (n) and returns the 3D matrices containing the
    sed volumes in V_mob in their new position at the end of the timestep.
    This simple version of the function represents the volume as a point 
    sediment parcel delivered from the ToN of the reach n. Thus the volume
    has a single destination reach and it never get split. """ 
    
    ##initailize parameters
    outlet = Network['NH'][-1]

    ## find start and end reach of the sed volume after the timestep
    
    # reach_dest is the id of the reach where the sed. volume stops after the timestep 
    #p_dest is the position from the from_node of the id reach where the sed. volume stops after the timestep 

    if n == outlet:  
        reach_dest = np.repeat( n , np.shape(v_sed_day)[0])
        p_dest = v_sed_day[:,n] + np.array(Lngt[n])
    else:
        #to find p_end, i track the position of a sediment parcel starting
        #from the To_Node of the reach n (i.e. the From_node of the downstream reach).
        p_dest , reach_dest , outind = track_sed_position( int(Network['Downstream_Node'][int(n)]) , v_sed_day , Lngt , psi, Network )
        p_dest = p_dest + np.array(Lngt[n])
        
    #downdist contains the distanche from the starting reach to all reaches
    #downstream
    downdist = Network['downstream_distance'][int(n)]
   
    ## find position of the sediment volume
        
    #setout is equal to 1 if the volume in the sed.class left the network
    #via the outlet
    setout = (np.squeeze(p_dest) - np.array(Lngt[reach_dest]) - downdist[reach_dest].T> 0)*1
     
    #in each row, setplace is equal to 1 in the reach where the sed. volume
    #of each class is delivered 
    setplace = np.zeros((len(v_sed_day), len(downdist)))
    setplace[np.arange(len(v_sed_day)), reach_dest]  = 1
    
    
    
    setplace[setout==1,:] = 0

    ## place volume to destination reach 
    
    Qbi_tr_t = np.zeros((len(Lngt), len(Lngt) , len(setplace)))
    Q_out_t = np.zeros ((len(Lngt), len(setplace)))
    
    for c in range(len(setplace)): 
        Qbi_tr_t[[V_mob[:,0].astype(int)],:,c] = V_mob[:,c+1][:,None] * setplace[c,:][None,:]
        Q_out_t[[V_mob[:,0].astype(int)],:] = V_mob[:,1:] * setout
               

    
    return Qbi_tr_t, Q_out_t , setplace, setout



def change_slope(Node_el_t, Lngt, Network , **kwargs):
    """"CHANGE_SLOPE modify the Slope vector according to the changing elevation of
    the nodes: It also guarantees that the slope is not negative or lower then
    the min_slope value by changing the node elevation bofore findin the SLlpe""" 
    
    #define minimum reach slope 
       
        
    #initialization
    if len(kwargs) != 0: 
        min_slope = kwargs['s'] 
    else: 
        min_slope = 0
     
    outlet = Network['NH'][-1]
    down_node = Network['Downstream_Node']    
    down_node = np.array([ int(n) for n in down_node])
    down_node[int(outlet)] = (len(Node_el_t)-1)
    
    Slope_t = np.zeros(Lngt.shape)
    
    #loop for all reaches
    for n in range(len(Lngt)): 
        #find the minimum node elevation to guarantee Slope > min_slope
        min_node_el = min_slope *  Lngt[n] + Node_el_t[down_node[n]]

        #change the noide elevation if lower to min_node_el
        Node_el_t[n] = np.maximum(min_node_el, Node_el_t[n] ) 
        
        #find the new slope
        Slope_t[n] = (Node_el_t[n] - Node_el_t[down_node[n]]) / Lngt[n]
    
    
    return Slope_t, Node_el_t




def stop_or_not(t_new, Vm):
    ''' 
    Function that decides if traveling cascades of sediments will stop in this 
    reach or not, depending on time.
    
    t_new: elapsed time since beginning of time step for Vm, for each sed class
    Vm: traveling cascade of sediments
    '''
    cond_stop = np.insert([t_new>1], 0, True)
    Vm_stop = np.zeros_like(Vm)
    Vm_stop[:, cond_stop] = Vm[:, cond_stop]
    
    cond_continue = np.insert([t_new<=1], 0, True)
    Vm_continue = np.zeros_like(Vm)
    Vm_continue[:, cond_continue] = Vm[:, cond_continue]
    
    if np.all(Vm_stop[:,1:] == 0) == True:
        Vm_stop = None
    if np.all(Vm_continue[:,1:] == 0) == True: 
        Vm_continue = None
        
    return Vm_stop, Vm_continue


def deposit_from_passing_sediments(V_remove, cascade_list):
    ''' This function remove the quantity V_remove from the list of cascades. 
    The order in which we take the cascade is from smallest times (arriving first) 
    to longest times (arriving later). If two cascades have the same time

    V_remove : quantity to remove, per sediment class.
    cascade_list : list of cascades. Reminder, a cascade is a tuple 
                    of direct provenance, elapsed time, and the Vmob (p, t, Vmob)
    '''
    removed_Vm_all = []    
    
    # order volumes Vm according to their elapsed time, 
    # and concatenate Vm with same time to be treated together in the loop
    df = pd.DataFrame(cascade_list, columns=['provenance', 'times', 'Vm'])   
    df['sum_times'] = df['times'].apply(np.sum)
    ordered_Vm_list = df.groupby('sum_times')['Vm'].apply(lambda x: np.concatenate(x, axis=0)).tolist()
    
    for Vm in ordered_Vm_list:
        if np.any(Vm[:,1:]) == False: #In case V_m is full of 0
            del Vm
            continue 
        removed_Vm = np.zeros_like(Vm)
        removed_Vm[:,0]=Vm[:,0] #first col with initial provenance
        for col_idx in range(Vm[:,1:].shape[1]):  # Loop over sediment classes
            if V_remove[col_idx] > 0:
                col_sum = np.sum(Vm[:, col_idx+1])        
                if col_sum > 0:
                    fraction_to_remove = min(V_remove[col_idx] / col_sum, 1.0)
                    removed_quantities = Vm[:, col_idx+1] * fraction_to_remove
                    # Subtract the removed quantities from V_m
                    Vm[:, col_idx+1] -= removed_quantities       
                    # Ensure no negative values
                    Vm[:, col_idx] = np.where(Vm[:, col_idx+1] < 0, 0, Vm[:, col_idx+1])       
                    # Store the removed quantities in the new matrix
                    removed_Vm[:, col_idx+1] = removed_quantities               
                    # Update V_remove by subtracting the total removed quantity
                    V_remove[col_idx] -= col_sum * fraction_to_remove                                
                    # Ensure V_remove doesn't go negative
                    V_remove[col_idx] = max(V_remove[col_idx], 0)                                                               
        removed_Vm_all.append(removed_Vm)
    # Concatenate all removed quantities into a single matrix
    r_Vmob = np.vstack(removed_Vm_all) if removed_Vm_all else np.array([])
    # Gather layers in r_Vmob 
    r_Vmob = matrix_compact(r_Vmob)
    
    return r_Vmob, cascade_list






