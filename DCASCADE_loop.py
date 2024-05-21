  # -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:00:36 2022

This script contains the time-space loop which assess the sediment routing through the network 

This script was adapted from the Matlab version by Marco Tangi 

@author: Elisa Bozzolan
"""

""" Libraries to import """
import numpy as np
import numpy.matlib
import pandas as pd
from tqdm import tqdm 
import copy
import sys

from widget import read_user_input
from supporting_functions import D_finder
from supporting_functions import sortdistance
from supporting_functions import layer_search
from supporting_functions import tr_cap_deposit
from supporting_functions import matrix_compact 
from supporting_functions import sed_transfer_simple
from supporting_functions import change_slope
from transport_capacity_computation import tr_cap_junction
from transport_capacity_computation import sed_velocity

np.seterr(divide='ignore', invalid='ignore')
             
""" MAIN FUNCTION SECTION """

def DCASCADE_main(ReachData , Network , Q , Qbi_input, Qbi_dep_in, timescale, psi, roundpar):
    """INPUT :
    ReachData      = nx1 Struct defining the features of the network reaches
    Network        = 1x1 struct containing for each node info on upstream and downstream nodes
    Q              = txn matrix reporting the discharge for each timestep
    Qbi_input       = per each reach and per each timestep is defined an external sediment input of a certain sediment class 
    Qbi_dep_in          = deposit of a sediment material known to be at a certain reach 
                     (it could be that for the same reach id, there are two strata defined so two rows of the dataframe with the top row is the deepest strata)
    timescale      = length for the time horizion considered
    psi            = sediment classes considered (from coarse to fine)
    roundpar       = mimimum volume to be considered for mobilization of subcascade 
                     (as decimal digit, so that 0 means not less than 1m3; 1 means no less than 10m3 etc.)
  
    
    OUTPUT: 
    data_output      = struct collecting the main aggregated output matrices 
    extended_output  = struct collecting the raw D-CASCADE output datasets"""
    
    
    

    # formula selection 
    
    # indx_tr_cap , indx_partition = read_user_input()
    indx_tr_cap = 3   #using engulend and Hansen
    indx_partition = 2 ## using BMF
    

    indx_velocity = 1 #    # EB: will need to create the option also for the index velocity (with fractional and total transport capacity)

    ################### fixed parameters
    phi = 0.4 #sediment porosity in the maximum active layer
    minvel = 0.00001
    outlet = Network['NH'][-1] #outlet reach ID identification
    n_reaches = len(ReachData)
    n_classes = len(psi)
    
    # initialise slope 
    min_slope = min(ReachData['Slope']) # put a minimum value to guarantee movement 
    Slope = np.zeros((timescale, n_reaches))
    Slope[0,:] = np.maximum(ReachData['Slope'], min_slope)
    Slope[1,:] = np.maximum(ReachData['Slope'], min_slope) 
    
    # initialize node elevation (for each reach the matrix reports the fromN elevation)
    # the last column reports the outlet ToNode elevation (last node of the network), which can never change elevation.
    Node_el = np.zeros((timescale, n_reaches+1))
    Node_el[0,:] = pd.concat([ReachData['el_FN'], ReachData['el_TN'][outlet]])
    Node_el[1,:] = pd.concat([ReachData['el_FN'], ReachData['el_TN'][outlet]])
    Node_el[:,-1] =  Node_el[1,-1]
    
    
    # initialise all sediment variables 
    Qbi_tr = [np.zeros((n_reaches,n_reaches,n_classes)) for _ in range(timescale)] # sediment within the reach AFTER transfer, which also gives the provenance 
    Qbi_mob = [np.zeros((n_reaches,n_reaches,n_classes))for _ in range(timescale)] # sediment within the reach BEFORE transfer, which also gives the provenance 
    Qbi_dep = [[np.expand_dims(np.zeros(n_classes+1), axis = 0) for _ in range(n_reaches)] for _ in range(timescale)] # information on deposited sediments (gives the strata composition at the end of the simulation)
    Fi_r_act = [np.empty((n_classes, n_reaches)) for _ in range(timescale)] # contains grain size distribution of the active layer
    Fi_r_act[0][:]= np.nan
    Q_out =[np.zeros((n_reaches, n_classes)) for _ in range(timescale)] # amount of material delivered outside the network in each timestep
    D50_AL = np.zeros((timescale,n_reaches)) # D50 of the active layer in each reach in each timestep
    V_sed = [np.zeros((n_classes, n_reaches)) for _ in range(timescale)] 
    Delta_V_all = np.zeros((timescale,n_reaches)) # reach mass balance (volumes eroded or deposited)
    tr_cap_all = [np.zeros((n_reaches, n_classes)) for _ in range(timescale)]
    Qbi_act = [np.empty((n_classes, n_reaches)) for _ in range(timescale)]  
   
    #""" EB constant slope """
    Slope[:,:] = Slope[0,:]
    Node_el[:,: ] =  Node_el[0,:]

    
    #initialise sediment deposit in the reaches 
    for n in Network['NH']:        
        # if no inputs are defined, initialize deposit layer with a single cascade with no volume and GSD equal to 0
        n = int(n)
        q_bin = np.array([Qbi_dep_in[n]])
        
        # if n == 25 :
        #      print ("why")    

            
        if not q_bin.any(): #if all zeros 
           Qbi_dep[0][n] = np.hstack((n, np.zeros(n_classes))).reshape(1,-1)
        else:
            
           Qbi_dep[0][n] = np.float32(np.hstack((np.ones(q_bin.shape[0])*n, Qbi_dep_in[n]))).reshape(1,-1) 
           Fi_r_act[0][:,n] = np.sum(q_bin, axis=0)/np.sum(q_bin) 
           D50_AL[0,n] = D_finder(Fi_r_act[0][:,n], 50, psi)
         

        if len(Qbi_dep[0][n].shape) == 2: 
            Qbi_dep[1][n] = Qbi_dep[0][n]
        else: 
            Qbi_dep[1][n] = Qbi_dep[0][n].reshape(1,-1) # keep vectors in the same matlab dimensions for clarity 
        
    # set limit for erosion in 1st timestep, given by the parameter mlim, that
    # is the maximum depth in meter that can be eroded for each reach. In this case mlim is constant and defined by maxH_ActLayer
    maxH_ActLayer = 0.3
    # if n == 10 :
    #       print ("why")    
    
    mlim = np.ones((1,n_reaches))  * maxH_ActLayer 
    V_lim_tot = np.round( mlim*ReachData['Wac'].values*ReachData['Length'].values, roundpar) 
 
    # support variable recording the total transport capacity in each reach in each timestep
    tr_cap_sum = np.zeros((timescale, n_reaches))

    # start waiting bar 
     
    for t in tqdm(range(timescale-1)):
        #bar.update(t)   
        #tqdm(range(1, timescale-1))
        
        
        if t == 10:
            if n == 10 :
                  print ("why")
        
        # calculate new water dept for all reaches via Manning equation
        h = np.power(Q.iloc[t,:].astype('float')*ReachData['n'].astype('float')/(ReachData['Wac'].astype('float')*np.sqrt(Slope[t])), 3/5)
        v = 1/ReachData['n'].astype('float')*np.power(h,2/3)*np.sqrt(Slope[t])
        
        
        
        # loop for all reaches
        for n in Network['NH']:

            n = int(n)
            
            
            # if t == 10:
            #     if n == 25 :
            #           print ("why")
    
       
            V_dep_old = Qbi_dep[t][n]# extract the deposit layer of the reach from the relative cell in the previous timestep
            

            # 1) extract the deposit layer from the storage matrix and load the incoming cascades
            
            if Qbi_input[t][n,:].ndim == 1:   
                vect = np.expand_dims(np.append(n, Qbi_input[t][n,:]), axis = 0)
            else: 
                vect = np.c_[np.repeat(n, Qbi_input[t][n,:].shape[0]), Qbi_input[t][n,:]]
            
            Qbi_incoming  =  np.r_[(np.c_[np.array(range(len(Network['NH']))), Qbi_tr[t][:, n,:]]), vect] # the material present at that time step + potential external mat
            Qbi_incoming  = np.delete(Qbi_incoming, np.sum(Qbi_incoming[:,1:], axis = 1)==0, axis = 0) # sum all classes and delete the zeros  (rows represents provenance)
            
            if Qbi_incoming.size == 0:
                Qbi_incoming = np.hstack((n, np.zeros(len(psi)))) # put an empty cascade if no incoming volumes are present (for computation)
            
            if Qbi_incoming.ndim == 1:
                Qbi_incoming = np.expand_dims(Qbi_incoming, axis = 0)


            # sort incoming matrix according to distance, in this way sediment coming from closer reaches will be deposited first 
            Qbi_incoming = sortdistance(Qbi_incoming, Network['upstream_distance_list'][n] )

     

            # 2) find cascades to be included into the active layer according to the limit V_lim_tot, and use the cumulative GSD to compute tr_cap
            # find the volume of sediment from the incoming load (V_inc2act) and deposit layer (V_dep2act) to be included in the active layer
            V_inc2act , V_dep2act ,  V_dep , Fi_r_act[t][:,n] = layer_search(Qbi_incoming, V_dep_old, V_lim_tot[0,n], roundpar)
           
            # if t == 297:
            #  if n == 10:
            #    print ("why")
    
            if np.sum(Fi_r_act[t][:,n])==0:
               Fi_r_act[t][:,n] = Fi_r_act[t-1][:,n] # in case the active layer is empty, i use the GSD of the previous timesteep
            
            #volumes in the active layer per class
            Qbi_act[t][:,n]=(np.sum(V_dep2act[:, 1:], axis=0) + np.sum(V_inc2act[:, 1:], axis=0))
            
           
            # if t == 297:
            #     if n == 9:
            #       print ("why")
             
            
            D50_AL[t,n] = D_finder(Fi_r_act[t][:,n], 50, psi)
            
            #calculate transport capacity using the Fi of the active layer, the resulting tr_cap is in m3/s and is converted in m3/day
           
            tr_cap = tr_cap_junction(Fi_r_act[t][:,n] , D50_AL[t,n], Slope[t,n] , Q.iloc[t,n], ReachData['Wac'][n], v[n] , h[n], psi, indx_tr_cap, indx_partition)*24*60*60
            tr_cap_all[t][n,:]=tr_cap
            tr_cap_sum[t,n] = np.sum(tr_cap)
            
            # if t == 328:
            #     if n == 4:
            #         print ("why")
           
            
            if np.sum(tr_cap) < (np.sum(V_dep2act[:,1:]) + np.sum(V_inc2act[:,1:])): # if the total transport capacity is lower than the active layer volume...
               #... deposit part of the active layer cascades, 
               #    proportionally to their volume and the volume of the active layer
              
              [V_mob, V_dep ] = tr_cap_deposit( V_inc2act, V_dep2act, V_dep, tr_cap, roundpar)   
            else:
               # if not, the mobilized layer is equal to the active layer
               V_mob = matrix_compact(np.vstack((V_dep2act ,V_inc2act)))

            # if t == 642:
            #     if n == 10:
            #       print ("why")
            
            # (after this passage, V_mob contains only the volumes actually mobilized)     
            Qbi_dep[t+1][n] = np.float32(V_dep)
            
            #remove empty rows
            Qbi_dep[t+1][n] = Qbi_dep[t+1][n][np.sum(Qbi_dep[t+1][n][:,1:],axis = 1)!= 0]

            # Qbi_mob contains the volume mobilized in the reach, that is about to be transfer downstream
            Qbi_mob[t][V_mob[:,0].astype(int),n,:] = np.float32(V_mob[:,1:]) 

            #if removing empty rows leaves only an Qbi_dep{t,n} empty
            # matrix, put an empty layer
            if  (Qbi_dep[t+1][n]).size == 0 :
                Qbi_dep[t+1][n] = np.float32(np.append(n, np.zeros(len(psi))).reshape(1,-1))
                
            # 4) Compute the changes in bed elevation
            # modify bed elevation according to increased deposit
        
                
            Delta_V = np.sum(Qbi_dep[t+1][n][:,1:]) -  np.sum(Qbi_dep[t][n][:,1:])
           
            #record Delta_V
            Delta_V_all[t,n] = Delta_V
            
           
            #""" EB constant slope """
            #Node_el[t+1][n]= Node_el[t,n] + Delta_V/( np.sum(ReachData['Wac'][np.append(n, Network['Upstream_Node'][n])] * ReachData['Length'][np.append(n, Network['Upstream_Node'][n])]) * (1-phi) )
        # end of the reach loop
        
        # 5) Move the mobilized volumes to the destination reaches according to the sediment velocity
        #loop for all reaches, now that i have the Fi_r and thus can compute transfer rates for all reaches
        #print(n, t, 'v dep', V_dep)

        for n in Network['NH']:
            #load mobilized volume for reach n
            
            V_mob = np.zeros((len(Network['NH']),len(psi)+1))
            V_mob[:,0] = np.arange(len(Network['NH']))
            
            V_mob[:,1:len(psi)+1] = np.squeeze(Qbi_mob[t][:,n,:])
            V_mob = matrix_compact(V_mob)
            
            #calculate GSD of mobilized volume
            Fi_mob = (np.sum(V_mob[:,1:],axis = 0)/np.sum(V_mob[:,1:]))[:,None] # EB: must be a column vector
            if np.isnan(Fi_mob).any():
                Fi_mob = Fi_r_act[t][:,n]
                
              
            # if t == 10:
            #     if n == 4:
            #         print ("why")
              
                
            #calculate sediment velocity for the mobilized volume in each reach
            v_sed = sed_velocity( np.matlib.repmat(Fi_mob, 1, len(Network['NH'])), Slope[t,:] , Q.iloc[t,:], ReachData['Wac'] , v , h ,psi,  minvel , phi , indx_tr_cap, indx_partition, indx_velocity )
            
            #transfer the sediment volume downstream
            Qbi_tr_t, Q_out_t, setplace, setout = sed_transfer_simple( V_mob , n , v_sed*(60*60*24) , ReachData['Length'], Network, psi )

            # Sum the volumes transported from reach n with all the othe 
            # volumes mobilized by all the other reaches at time
            #print(t, n, np.sum(Qbi_tr[t+1]), np.sum(Qbi_tr_t))
            Qbi_tr[t+1] = Qbi_tr[t+1] + np.float32(Qbi_tr_t)
            Q_out[t] =  Q_out[t] + Q_out_t
            V_sed[t] = v_sed*(60*60*24)
            del Qbi_tr_t,Q_out_t
        
       


        #change the slope accordingly to the bed elevation
        """ EB constant slope """
        #Slope[t+1,:], Node_el[t+1,:] = change_slope( Node_el[t+1,:] ,ReachData['Length'], Network, s = min_slope )

        
        #measure time of routing
        #time2   = clock;

        #if np.remainder(10, t) == 0:  # save time only at certain timesteps 
        #   timerout = etime(time2, time1);
         
        """end of the time loop"""
    
      
    # output processing
    # aggregated matrixes
    
    QB_mob_t = [np.sum(x, axis = 2) for x in Qbi_mob[0:timescale-1]] #sum along sediment classes
    Qbi_mob_class = [np.sum(x, axis = 0) for x in Qbi_mob[0:timescale-1]] #sum along sediment classes
    QB_mob = np.rollaxis(np.dstack(QB_mob_t),-1) 
    QB_mob_sum = np.sum(QB_mob, 1) #total sediment mobilized in that reach for that time step (all sediment classes, from all reaches)
    
    #total sediment delivered in each reach (column), divided by reach provenance (row) 
    QB_tr_t = [np.sum(x, axis = 2) for x in Qbi_tr[0:timescale-1]] 
    QB_tr = np.rollaxis(np.dstack(QB_tr_t),-1)
    
    
    V_dep_sum = np.zeros((timescale-1, n_reaches ))  # EB : last time step would be equal to 0 - delete to avoid confusion 
    V_class_dep = [[np.expand_dims(np.zeros(n_classes+1), axis = 0) for _ in range(n_reaches)] for _ in range(timescale)]

    
    for t in (np.arange(len(Qbi_dep)-1)):
        for n in range(len(Qbi_dep[t])): 
            q_t = Qbi_dep[t][n] 
            #total material in the deposit layer 
            V_dep_sum[t,n] = np.sum(q_t[:,1:])
            # total volume in the deposit layer for each timestep, divided by sed.class 
            V_class_dep[t][n] = np.sum(q_t[:,1:], axis = 0) 
            
            
            
    #total material in a reach in each timestep (both in the deposit layer and mobilized layer)
    tot_sed = V_dep_sum + np.sum(QB_tr, axis = 1) 
    
    #total material transported 
    tot_tranported = np.sum(QB_tr, axis = 1) 
    
    
    #total material in a reach in each timestep, divided by class 
    tot_sed_temp = []
    Qbi_dep_class = []
    D50_tot = np.zeros((timescale-1, n_reaches))
 
    for t in np.arange(len(Qbi_mob_class)):
        v_dep_t = np.vstack(V_class_dep[t])
        tot_sed_temp.append(Qbi_mob_class[t] + v_dep_t)
        Qbi_dep_class.append(v_dep_t)
        Fi_tot_t = tot_sed_temp[t]/ (np.sum(tot_sed_temp[t],axis = 1).reshape(-1,1))
        Fi_tot_t[np.isnan(Fi_tot_t)] = 0
        for i in np.arange(n_reaches):
            D50_tot[t,i] = D_finder(Fi_tot_t[i,:], 50, psi)
            

    
    #EB: add as above but D50 of only mobilised volume 
    D50_mob = np.zeros((timescale-1, n_reaches))
 
    for t in np.arange(len(Qbi_mob_class)):
            
        Fi_mob_t = Qbi_mob_class[t]/ (np.sum(Qbi_mob_class[t],axis = 1).reshape(-1,1))
        Fi_mob_t[np.isnan(Fi_mob_t)] = 0
        for i in np.arange(n_reaches):
            
          D50_mob[t,i] = D_finder(Fi_mob_t[i,:], 50, psi)
            
    #EB: add as above but D50 of only deposited volume 
    dep_sed_temp = []
    D50_dep = np.zeros((timescale-1, n_reaches))
    
    # stack the deposited volume 
    dep_sed_temp = []
    D50_dep = np.zeros((timescale-1, n_reaches))
    
    for t in np.arange(len(Qbi_dep_class)):
        Fi_dep_t = Qbi_dep_class[t]/ (np.sum(Qbi_dep_class[t],axis = 1).reshape(-1,1))
        Fi_dep_t[np.isnan(Fi_dep_t)] = 0
        for i in np.arange(n_reaches):
            D50_dep[t,i] = D_finder(Fi_dep_t[i,:], 50, psi)
            
            
    #total material in a reach in each timestep, divided by class 
    tot_sed_class =  [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    q_d = np.zeros((1, n_reaches))
    
    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_t = Qbi_dep[t] # get the time step
            for i, reaches in enumerate(q_t): # get the elements of that class per reach 
                q_d[0,i] = np.sum(reaches[:,c+1])
            q_tt = Qbi_tr[t][:,:,c]
            tot_sed_class[c][t,:] = q_d + np.sum(q_tt, axis = 0)
            
             
    deposited_class =  [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]

    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_t = Qbi_dep[t]
            deposited_class[c][t,:] = np.array([np.sum(item[:,c+1], axis = 0) for item in q_t]) 
   
  
    #mobilised per class
    mobilised_class =  [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    
    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_m = Qbi_mob[t][:,:,c]
            mobilised_class[c][t,:] = np.sum(q_m, axis = 0)
    
    #transported per class        
    transported_class =  [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    
    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_m = Qbi_tr[t][:,:,c]
            transported_class[c][t,:] = np.sum(q_m, axis = 0)
            
    #Active layer per class (put in same format as mob and trans per class)
    AL_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_m = Qbi_act[t][c,:]
            AL_class[c][t,:] = q_m  
            
    #Tranport capacity per class (put in same format as mob and trans per class)
    tr_cap_class = [np.empty((timescale-1, n_reaches)) for _ in range(n_classes)]
    for c in range(n_classes): 
        for t in range(timescale-1): 
            q_m = tr_cap_all[t][:,c]
            tr_cap_class[c][t,:] = q_m       
    
    
    
    #total sediment volume leaving the network
    outcum_tot = np.array([np.sum(x) for x in Q_out])
    
    #set all NaN transport capacity to 0
    tr_cap_sum[np.isnan(tr_cap_sum)] = 0 
    
    #set all NaN active layer D50 to 0; 
    D50_AL[np.isnan(D50_AL)] = 0
    
    Q = np.array(Q)
    
    ## output struct definition 
    #data_plot contais the most important D_CASCADE outputs 
    data_output = { 'Channel Width [m]': np.repeat(np.array(ReachData['Wac']).reshape(1,-1),len(Qbi_dep), axis = 0), 
                   'Mobilized volume [m^3]' : QB_mob_sum , 'Transported + deposited sed in the reach [m^3]':tot_sed,  
                   'Transported [m^3]':  tot_tranported, 
                   'Reach Slope':Slope, 'D50 deposited layer [m]' :D50_dep, 'D50 mobilised layer [m]':D50_mob,
                   'D50 active layer [m]' :D50_AL,  
                   'Daily trasport capacity [m^3/day]': tr_cap_sum, 'Deposited volume[m^3]': V_dep_sum, 
                   'Discharge [m^3/s]': Q[0:timescale,:], 'Transported + deposited sed - per class [m^3/s]':  tot_sed_class, 
                 'Deposited sed in the reach - per class [m^3/s]': deposited_class,
                  'Mobilised sed in the reach - per class [m^3/s]': mobilised_class, 
                  'Transported sed in the reach - per class [m^3/s]': transported_class,
                  'Active layer sed in the reach - per class [m^3/s]':   AL_class,
                  'Tr cap sed in the reach - per class [m^3/s]': tr_cap_class}
   
    #all other outputs are included in the extended_output cell variable 
    extended_output = { 'Qbi_tr': Qbi_tr,  
                   'Qbi_mob' : Qbi_mob  , 'Q_out' : Q_out ,  
                   'Qbi_dep':Qbi_dep, 'Fi_r_ac' :Fi_r_act ,  
                   'Node_el' : Node_el }
    

    return data_output,extended_output



















