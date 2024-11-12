# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:58:54 2024

@author: diane
"""
# General imports
import numpy as np
import numpy.matlib
import pandas as pd
from tqdm import tqdm 
import copy
import sys
import os
np.seterr(divide='ignore', invalid='ignore')

# Supporting functions

# from supporting_functions import D_finder, sortdistance, matrix_compact, change_slope
# from supporting_functions import layer_search, tr_cap_deposit
# # from supporting_functions import cascades_end_time_or_not
# from supporting_functions import deposit_from_passing_sediments
# from supporting_functions import compute_time_lag

# from transport_capacity_computation import compute_transport_capacity, compute_cascades_velocities

from flow_depth_calc import choose_flow_depth
from slope_reduction import choose_slopeRed

from supporting_classes import Cascade

from supporting_functions import sortdistance, change_slope, D_finder



class DCASCADE:
    def __init__(self, sedim_sys, indx_flo_depth, indx_slope_red):
        
        self.sedim_sys = sedim_sys
        self.reach_data = sedim_sys.reach_data
        self.network = sedim_sys.network
        self.n_reaches = sedim_sys.n_reaches
        self.n_classes = sedim_sys.n_classes
        
        # Simulation attributes
        self.timescale = sedim_sys.timescale   # time step number
        self.ts_length = sedim_sys.ts_length                  # time step length
        self.save_dep_layer = sedim_sys.save_dep_layer        # option for saving the deposition layer or not
        self.update_slope = sedim_sys.update_slope            # option for updating slope
        
        # Indexes
        self.indx_flo_depth = indx_flo_depth
        self.indx_slope_red = indx_slope_red
        self.indx_tr_cap = None
        self.indx_tr_partition = None
        self.indx_velocity = None
        self.indx_vel_partition = None
        
        # Algorithm options
        self.passing_cascade_in_outputs = None
        self.passing_cascade_in_trcap = None        
        self.time_lag_for_mobilised = None
    
    def set_transport_indexes(self, indx_tr_cap, indx_tr_partition):
        self.indx_tr_cap = indx_tr_cap 
        self.indx_tr_partition = indx_tr_partition
        
    def set_velocity_indexes(self, indx_velocity, indx_vel_partition):
        self.indx_velocity = indx_velocity 
        self.indx_vel_partition = indx_vel_partition
        
    def set_algorithm_options(self, passing_cascade_in_outputs, passing_cascade_in_trcap, 
                                   time_lag_for_mobilised):
        self.passing_cascade_in_outputs = passing_cascade_in_outputs
        self.passing_cascade_in_trcap = passing_cascade_in_trcap        
        self.time_lag_for_mobilised = time_lag_for_mobilised
        
        self.check_algorithm_compatibility()
    
    def check_algorithm_compatibility(self):
        # Constrain on the option of the algorithm:
        if self.passing_cascade_in_trcap == True and self.passing_cascade_in_outputs == False:
            raise ValueError("You can not use this combination of algorithm options")
        if self.time_lag_for_mobilised == True and (self.passing_cascade_in_outputs == False or self.passing_cascade_in_trcap == False):
            raise ValueError("You can not use this combination of algorithm options")
     

      
    

        
    
    def run(self, Q, roundpar):
        # DD: Should we create a subclass in SedimentarySystem to handle the temporary parameters for one time step
        # like Qbi_pass, Qbi_dep_0 ? Could be SedimentarySystemOneTime ?
        
        SedimSys = self.sedim_sys
        
        # start waiting bar    
        for t in tqdm(range(self.timescale - 1)):
            
            # TODO: DD see with Anne Laure and Felix, which slope are we using for the flow?
            
            # Define flow depth and flow velocity for all reaches at this time step:
            h, v = choose_flow_depth(self.reach_data, SedimSys.slope, Q, t, self.indx_flo_depth)
            SedimSys.flow_depth[t] = h
            
            # Slope reduction functions
            SedimSys.slope = choose_slopeRed(self.reach_data, SedimSys.slope, Q, t, h, self.indx_slope_red)
           
            # deposit layer from previous timestep
            Qbi_dep_old = copy.deepcopy(self.sedim_sys.Qbi_dep_0)
            
            # volumes of sediment passing through a reach in this timestep,
            # ready to go to the next reach in the same time step.
            Qbi_pass = [[] for n in range(self.n_reaches)]
                    
            # loop for all reaches:
            for n in self.network['n_hier']:  
                
                # TODO : How to include the lateral input ?
                                   
                # Extracts the deposit layer left in previous time step          
                Vdep_init = Qbi_dep_old[n] # extract the deposit layer of the reach 
                                           
                ###------Step 1 : Cascades generated from the reaches upstream during 
                # the present time step, are passing the inlet of the reach
                # (stored in Qbi_pass[n]).
                # This computational step make them pass to the outlet of the reach 
                # or stop in the reach, depending if their velocity make them
                # arrive at the outlet before the end of the time step or not.
                
                # Temporary condition (if False, reproduces v1).
                if self.passing_cascade_in_outputs == True:            
                    # Store the arriving cascades in the transported matrix (Qbi_tr)
                    # Note: we store the volume by original provenance
                    for cascade in Qbi_pass[n]:
                        SedimSys.Qbi_tr[t][[cascade.volume[:,0].astype(int)], n, :] += cascade.volume[:, 1:]
                        # DD: If we want to store instead the direct provenance
                        # Qbi_tr[t][cascade.provenance, n, :] += np.sum(cascade.volume[:, 1:], axis = 0)  
                
                    
                # Compute the velocity of the cascades in this reach [m/s] 
                if Qbi_pass[n] != []:
                    # Define the velocity section height:
                    # coef_AL_vel = 0.1
                    # SedimSys.vl_height[t,n] = coef_AL_vel * h[n]                 
                    SedimSys.vl_height[t,n] = SedimSys.al_depth[t,n]    #the velocity height is the same as the active layer depth
                    
                    SedimSys.compute_cascades_velocities(Qbi_pass[n], Vdep_init,
                                               Q[t,n], v[n], h[n], roundpar, t, n,                           
                                               self.indx_velocity, self.indx_vel_partition, 
                                               self.indx_tr_cap, self.indx_tr_partition)

                
                # Decides weather cascades, or parts of cascades, 
                # finish the time step here or not.
                # After this step, Qbi_pass[n] contains volume that do not finish
                # the time step in this reach
                if Qbi_pass[n] != []:
                    Qbi_pass[n], to_be_deposited = SedimSys.cascades_end_time_or_not(Qbi_pass[n], n)                    
                else:
                    to_be_deposited = None
                
                # Temporary to reproduce v1. Stopping cascades are stored at next time step.
                if self.passing_cascade_in_outputs == False:    
                    if to_be_deposited is not None:
                        SedimSys.Qbi_tr[t+1][[to_be_deposited[:,0].astype(int)], n, :] += to_be_deposited[:, 1:]
                                                        
                # After this step, Qbi_pass[n] contains volume that do not finish
                # the time step in this reach, i.e the passing cascades
                                
                    
                ###------Step 2 : Mobilise volumes from the reach considering the 
                # eventual passing cascades.
                
                # Temporary container to store the reach mobilised cascades:
                reach_mobilized_cascades = [] 
                
                # An optional time lag vector (x n_classes) is used to mobilise reach sediment  
                # before the eventual first passing cascade arrives at the outlet. 
                # (NB: it is a proportion of the time step)                 
                if self.time_lag_for_mobilised == True and Qbi_pass[n] != []:  
                    time_lag = SedimSys.compute_time_lag(Qbi_pass[n]) 
                    # Transport capacity is only calculated on Vdep
                    tr_cap_per_s, Fi_al, D50_al = SedimSys.compute_transport_capacity(Vdep_init, roundpar, t, n, Q, v, h,
                                                                             self.indx_tr_cap, self.indx_tr_partition)
                    # Store values: 
                    SedimSys.tr_cap_before_tlag[t, n, :] = tr_cap_per_s * time_lag * self.ts_length
                    SedimSys.Fi_al_before_tlag[t, n, :] = Fi_al
                    SedimSys.D50_al_before_tlag[t, n] = D50_al                                       
                    
                    # Mobilise during the time lag
                    Vmob, _, Vdep = SedimSys.compute_mobilised_volume(Vdep_init, tr_cap_per_s, 
                                                                      n, roundpar,
                                                                      time_fraction = time_lag) 
                                       
                    # Add the possible mobilised cascade to a temporary container
                    if Vmob is not None: 
                        elapsed_time = np.zeros(self.n_classes) # it start its journey at the end of the time lag
                        provenance = n
                        reach_mobilized_cascades.append(Cascade(provenance, elapsed_time, Vmob))
                        
                    #remaining time
                    r_time_lag = 1 - time_lag 
                    
                else:                  
                    time_lag = None
                    r_time_lag = None 
                    Vdep = Vdep_init
                    
                # To reproduce v1, we leave the option to consider passing cascades or not
                # in the transport capacity and mobilisation calculation
                if self.passing_cascade_in_trcap == True:
                    passing_cascades = Qbi_pass[n]
                else:
                    passing_cascades = None
                
                # Now compute transport capacity and mobilise  
                # considering eventually the passing cascades during the remaining time
                tr_cap_per_s, Fi_al, D50_al = SedimSys.compute_transport_capacity(Vdep, roundpar, t, n, Q, v, h,
                                                                                  self.indx_tr_cap, self.indx_tr_partition,
                                                                                  passing_cascades = passing_cascades,
                                                                                  per_second = True)                
               
                # Store transport capacity and active layer informations: 
                SedimSys.Fi_al[t, n, :] = Fi_al
                SedimSys.D50_al[t, n] = D50_al                     
                if r_time_lag is None:
                    # No time lag
                    SedimSys.tr_cap[t, n, :] = tr_cap_per_s * self.ts_length
                else:
                    # We sum the tr_caps from before and after the time lag
                    tr_cap_after_tlag = (tr_cap_per_s * r_time_lag * self.ts_length)
                    SedimSys.tr_cap[t, n, :] = SedimSys.tr_cap_before_tlag[t, n, :] + tr_cap_after_tlag
                SedimSys.tr_cap_sum[t,n] = np.sum(SedimSys.tr_cap[t, n, :])    
               
                
                # Mobilise:
                Vmob, passing_cascades, Vdep_end = SedimSys.compute_mobilised_volume(Vdep, tr_cap_per_s, 
                                                                                n, roundpar,
                                                                                passing_cascades = passing_cascades,
                                                                                time_fraction = r_time_lag)
                if passing_cascades is not None:
                    Qbi_pass[n] = passing_cascades
                    
                # Add the possible newly mobilised cascade to a temporary container
                if Vmob is not None: 
                    if time_lag is None:
                        elapsed_time = np.zeros(self.n_classes) 
                    else: 
                        elapsed_time = time_lag
                    provenance = n
                    reach_mobilized_cascades.append(Cascade(provenance, elapsed_time, Vmob))
                                        

                
                
                # # In the case time_lag is not all zeros, we mobilise from the reach
                # # before the cascades arrive, i.e. during this time_lag
                # mobilized_cascades = []
                # Vdep = Vdep_init
                # if np.any(time_lag != 0):                                
                #     tr_cap_per_s = SedimSys.compute_transport_capacity(Vdep, roundpar)
                                       
                #     # Mobilise during the time lag
                #     Vmob, Vdep, _ = compute_mobilised_volume(Vdep, 
                #                                              self.eros_max_vol[0,n], 
                #                                              tr_cap_per_s, time_lag)
                    
                
      
                    
                    
                    # # Get the sediment fraction and D50 in the active layer. 
                    # # DD: to be discussed if we keep the active layer (2D90) or a bigger active layer,
                    # # since we don't consider any incoming sediment now                          
                    # _,_,_, self.Fi_r_act[t,n,:] = layer_search(V_dep_init, self.al_vol_all[0,n], roundpar)
                    # self.D50_AL[t,n] = D_finder(self.Fi_r_act[t,n,:], 50, self.psi)           
                    # # In case the active layer is empty, I use the GSD of the previous timestep
                    # if np.sum(self.Fi_r_act[t,n,:]) == 0:
                    #    self.Fi_r_act[t,n,:] = self.Fi_r_act[t-1,n,:] 
                    # # Transport capacity in m3/s 
                    # tr_cap_per_s, Qc = tr_cap_function(self.Fi_r_act[t,n,:] , self.D50_AL[t,n], self.slope[t,n], 
                    #                                    Q[t,n], reach_data.wac[n], v[n], 
                    #                                    h[n], self.psi, self.indx_tr_cap, self.indx_tr_partition)                                   
                    # # Store tr_cap (and Qc)
                    # self.tr_cap_all[t,n,:] = tr_cap_per_s
                    # self.tr_cap_sum[t,n] = np.sum(tr_cap_per_s)
                    # if self.indx_tr_cap == 7:
                    #     self.Qc_class_all[t,n,:] = Qc   
                        
                #     # Total volume possibly mobilised during the time lag (per class)
                #     time_for_mobilising = time_lag * self.ts_length
                #     volume_mobilisable = tr_cap_per_s * time_for_mobilising
                    
                #     # Erosion maximum during the time lag 
                #     # (we take the mean time lag among the classes)
                #     eros_max_vol_t = self.eros_max_vol[0,n] * np.mean(time_lag)
                    
                #     # Mobilise from the reach layers
                #     V_inc_EL, V_dep_EL, V_dep_not_EL, _ = layer_search(V_dep_init, eros_max_vol_t, roundpar)
                #     [V_mob, V_dep_init] = tr_cap_deposit(V_inc_EL, V_dep_EL, V_dep_not_EL, volume_mobilisable, roundpar)
                                    
                #     if np.any(V_mob[:,1:] != 0): 
                #         # The mobilised cascade is added to a temporary container
                #         elapsed_time = np.zeros(self.n_classes) # its elapsed time is 0
                #         provenance = n
                #         mobilized_cascades.append(Cascade(provenance, elapsed_time, V_mob))
                    
                #         # Store this volume if we consider only this one
                #         if self.passing_cascade_in_outputs == False:
                #             self.Qbi_mob[t][[V_mob[:,0].astype(int)], n, :] += V_mob[:, 1:]
    
                
                # # New deposit layer after this step 2.
                # V_dep_after_tlag = V_dep_init
                                                 
                # ###-----Step 3 : Now mobilise material considering that cascades are passing
                # # at the outlet. We recalculate the transport capacity by including 
                # # now the passing cascades.  
                # if np.any(time_lag != 1):
                #     # Compute the transport capacity considering continuous arriving flux [m3/s]
                #     Q_pass_volume = np.concatenate([cascade.volume for cascade in Qbi_pass[n]], axis=0)
                #     Q_pass_volume = matrix_compact(Q_pass_volume)
                #     Q_pass_volume_per_s = copy.deepcopy(Q_pass_volume)
                #     # DD: should we put it in per second ? 
                #     Q_pass_volume_per_s[:,1:] = Q_pass_volume_per_s[:,1:] / self.ts_length                                            
                #     _,_,_, Fi_r = layer_search(V_dep_after_tlag, self.al_vol_all[0,n], roundpar, Qbi_incoming = Q_pass_volume_per_s) 
                #     self.D50_AL2[t,n] = float(D_finder(Fi_r, 50, self.psi))   
                #     tr_cap_per_s, Qc = tr_cap_function(Fi_r , self.D50_AL2[t,n], self.slope[t,n], 
                #                                        Q[t,n], reach_data.wac[n], v[n], 
                #                                        h[n], self.psi, self.indx_tr_cap, self.indx_tr_partition)                                   
                #     # Total volume possibly mobilised during (1 - time_lag)
                #     time_for_mobilising = (1-time_lag) * self.ts_length
                #     volume_mobilisable = tr_cap_per_s * time_for_mobilising
                    
                #     # Total volume arriving
                #     sum_pass = np.sum(Q_pass_volume[:,1:], axis=0)
                                  
                #     # Compare sum of passing cascade to the mobilisable volume (for each sediment class)               
                #     diff_with_capacity = volume_mobilisable - sum_pass
                                        
                #     # Sediment classes with positive values in diff_with_capacity are mobilised from the reach n
                #     diff_pos = np.where(diff_with_capacity < 0, 0, diff_with_capacity)     
                #     if np.any(diff_pos): 
                #         # Erosion maximum during (1 - time_lag)
                #         eros_max_vol_t = self.eros_max_vol[0,n] * (1 - np.mean(time_lag))  # erosion maximum
                        
                #         # Mobilise from the reach layers
                #         V_inc_EL, V_dep_EL, V_dep_not_EL, _ = layer_search(V_dep_after_tlag, eros_max_vol_t, roundpar)
                #         [V_mob, V_dep_after_tlag] = tr_cap_deposit(V_inc_EL, V_dep_EL, V_dep_not_EL, diff_pos, roundpar)
                        
                #         if np.any(V_mob[:,1:] != 0): 
                #             # The Vmob is added to the temporary container 
                #             elapsed_time = time_lag # it start its journey at the end of the time lag
                #             provenance = n
                #             mobilized_cascades.append(Cascade(provenance, elapsed_time, V_mob))
                                            
                #     # Sediment classes with negative values in diff_with_capacity
                #     # are over capacity
                #     # They are deposited, i.e. directly added to Vdep
                #     diff_neg = -np.where(diff_with_capacity > 0, 0, diff_with_capacity)     
                #     if np.any(diff_neg):  
                #         Vm_removed, Qbi_pass[n], residual = deposit_from_passing_sediments(np.copy(diff_neg), Qbi_pass[n], roundpar)
                #         V_dep_after_tlag = np.concatenate([V_dep_after_tlag, Vm_removed], axis=0) 
                        
                                    
                # # After this step, Qbi_pass[n] contains the traveling volumes that effectively 
                # # reach and pass the outlet of the reach.
                
                # # Update the deposit layer
                # V_dep_final = V_dep_after_tlag      
                
                
                ###-----Step 4: Finalisation.
                
                # Add the cascades that were mobilised from this reach to Qbi_pass[n]:
                if reach_mobilized_cascades != []:
                    Qbi_pass[n].extend(reach_mobilized_cascades) 
                
                # Store all the cascades in the mobilised volumes 
                if self.passing_cascade_in_outputs == True:
                    for cascade in Qbi_pass[n]:
                        SedimSys.Qbi_mob[t][[cascade.volume[:,0].astype(int)], n, :] += cascade.volume[:, 1:]
                        # DD: If we want to store instead the direct provenance
                        # Qbi_mob[t][cascade.provenance, n, :] += np.sum(cascade.volume[:, 1:], axis = 0)
                else:
                    for cascade in reach_mobilized_cascades:
                        SedimSys.Qbi_mob[t][[cascade.volume[:,0].astype(int)], n, :] += cascade.volume[:, 1:]     
                        
                # Deposit the stopping cascades in Vdep 
                if to_be_deposited is not None:                   
                    to_be_deposited = sortdistance(to_be_deposited, self.network['upstream_distance_list'][n])
                    Vdep_end = np.concatenate([Vdep_end, to_be_deposited], axis=0)
                    
                # ..and store Vdep for next time step
                SedimSys.Qbi_dep_0[n] = np.float32(Vdep_end)
    
                # Finally, pass these cascades to the next reach (if we are not at the outlet)  
                if n != SedimSys.outlet:
                    n_down = np.squeeze(self.network['downstream_node'][n], axis = 1) 
                    n_down = int(n_down) # Note: This is wrong if there is more than 1 reach downstream (to consider later)
                    Qbi_pass[n_down].extend(copy.deepcopy(Qbi_pass[n]))
                else:
                    n_down = None
                    # If it is the outlet, we add the cascades to Qout
                    for cascade in Qbi_pass[n]:
                        SedimSys.Q_out[t, [cascade.volume[:,0].astype(int)], :] += cascade.volume[:,1:]
                
                    
                # Compute the changes in bed elevation
                # Modify bed elevation according to increased deposit
                Delta_V = np.sum(SedimSys.Qbi_dep_0[n][:,1:]) -  np.sum(Qbi_dep_old[n][:,1:])
                # # Record Delta_V
                # self.Delta_V_all[t,n] = Delta_V 
                # # And Delta_V per class
                # self.Delta_V_class = np.sum(self.Qbi_dep_0[n][:,1:], axis=0) - np.sum(Qbi_dep_old[n][:,1:], axis=0)
                # self.Delta_V_class_all[t,n,:] = self.Delta_V_class            
                
                # Update slope, if required.
                if self.update_slope == True:
                    self.node_el[t+1][n]= self.node_el[t,n] + Delta_V/( np.sum(self.reach_data.Wac[np.append(n, self.network['upstream_node'][n])] * self.reach_data.length[np.append(n, self.network['Upstream_Node'][n])]) * (1-self.phi) )
                
            """End of the reach loop"""
            
            # Save Qbi_dep according to saving frequency
            if self.save_dep_layer == 'always':
                SedimSys.Qbi_dep[t+1] = copy.deepcopy(SedimSys.Qbi_dep_0)            
            if self.save_dep_layer == 'yearly':
                if int(t+2) % 365 == 0 and t != 0:
                    t_y = int((t+2)/365)
                    SedimSys.Qbi_dep[t_y] = copy.deepcopy(SedimSys.Qbi_dep_0)
                                
                
            # in case of changing slope..
            if self.update_slope == True:
                #..change the slope accordingly to the bed elevation
                self.slope[t+1,:], self.node_el[t+1,:] = change_slope(self.node_el[t+1,:], self.reach_data.length, self.network, s = self.min_slope)
                    
        """end of the time loop"""    
    
    def output_processing(self, Q):
        
        SedimSys = self.sedim_sys
        # output processing
        # aggregated matrixes
        
        QB_mob_t = [np.sum(x, axis = 2) for x in SedimSys.Qbi_mob[0:self.timescale-1]] #sum along sediment classes
        Qbi_mob_class = [np.sum(x, axis = 0) for x in SedimSys.Qbi_mob[0:self.timescale-1]] #sum along sediment classes
        QB_mob = np.rollaxis(np.dstack(QB_mob_t),-1) 
        QB_mob_sum = np.sum(QB_mob, 1) #total sediment mobilized in that reach for that time step (all sediment classes, from all reaches)
        
        #total sediment delivered in each reach (column), divided by reach provenance (row) 
        QB_tr_t = [np.sum(x, axis = 2) for x in SedimSys.Qbi_tr[0:self.timescale-1]] 
        QB_tr = np.rollaxis(np.dstack(QB_tr_t),-1)
        
        
        V_dep_sum = np.zeros((len(SedimSys.Qbi_dep)-1, self.n_reaches ))  # EB : last time step would be equal to 0 - delete to avoid confusion 
        V_class_dep = [[np.expand_dims(np.zeros(self.n_classes+1), axis = 0) for _ in range(self.n_reaches)] for _ in range(len(SedimSys.Qbi_dep))]
       
        for t in (np.arange(len(SedimSys.Qbi_dep)-1)):
            for n in range(len(SedimSys.Qbi_dep[t])): 
                q_t = SedimSys.Qbi_dep[t][n] 
                #total material in the deposit layer 
                V_dep_sum[t,n] = np.sum(q_t[:,1:])
                # total volume in the deposit layer for each timestep, divided by sed.class 
                V_class_dep[t][n] = np.sum(q_t[:,1:], axis = 0) 
                
        #--Total material in a reach in each timestep (both in the deposit layer and mobilized layer)                       
        if self.save_dep_layer=='always':           
            tot_sed = V_dep_sum + np.sum(QB_tr, axis = 1) 
        else:
            tot_sed= []
            
        #--Total material transported 
        tot_tranported = np.sum(QB_tr, axis = 1) 
        
        
        #total material in a reach in each timestep, divided by class 
        tot_sed_temp = []
        Qbi_dep_class = []
        # D50_tot = np.zeros((timescale-1, n_reaches))
     
        for t in np.arange(len(SedimSys.Qbi_dep)-1):
            v_dep_t = np.vstack(V_class_dep[t])
            # tot_sed_temp.append(Qbi_mob_class[t] + v_dep_t)
            Qbi_dep_class.append(v_dep_t)
            # Fi_tot_t = tot_sed_temp[t]/ (np.sum(tot_sed_temp[t],axis = 1).reshape(-1,1))
            # Fi_tot_t[np.isnan(Fi_tot_t)] = 0
            # for i in np.arange(n_reaches):
            #     D50_tot[t,i] = D_finder(Fi_tot_t[i,:], 50, psi)
        
        #--D50 of mobilised volume 
        D50_mob = np.zeros((self.timescale-1, self.n_reaches))
     
        for t in np.arange(len(Qbi_mob_class)):
            Fi_mob_t = Qbi_mob_class[t]/ (np.sum(Qbi_mob_class[t],axis = 1).reshape(-1,1))
            Fi_mob_t[np.isnan(Fi_mob_t)] = 0
            for i in np.arange(self.n_reaches):
                D50_mob[t,i] = D_finder(Fi_mob_t[i,:], 50, SedimSys.psi)
                
        #--D50 of deposited volume 
        dep_sed_temp = []
        D50_dep = np.zeros((self.timescale-1, self.n_reaches))
        
        # stack the deposited volume 
        dep_sed_temp = []
        D50_dep = np.zeros((self.timescale-1, self.n_reaches))
        
        for t in np.arange(len(Qbi_dep_class)):
            Fi_dep_t = Qbi_dep_class[t]/ (np.sum(Qbi_dep_class[t],axis = 1).reshape(-1,1))
            Fi_dep_t[np.isnan(Fi_dep_t)] = 0
            for i in np.arange(self.n_reaches):
                D50_dep[t,i] = D_finder(Fi_dep_t[i,:], 50, SedimSys.psi)
                
                
        #--Total material in a reach in each timestep, divided by class (transported + dep)
        tot_sed_class =  [np.empty((len(SedimSys.Qbi_dep), self.n_reaches)) for _ in range(self.n_classes)]
        q_d = np.zeros((1, self.n_reaches))
        
        for c in range(self.n_classes): 
            for t in range(len(SedimSys.Qbi_dep)): 
                q_t = SedimSys.Qbi_dep[t] # get the time step
                for i, reaches in enumerate(q_t): # get the elements of that class per reach 
                    q_d[0,i] = np.sum(reaches[:,c+1])
                q_tt = SedimSys.Qbi_tr[t][:,:,c]
                tot_sed_class[c][t,:] = q_d + np.sum(q_tt, axis = 0)
                
        #--Deposited per class         
        deposited_class =  [np.empty((len(SedimSys.Qbi_dep), self.n_reaches)) for _ in range(self.n_classes)]
    
        for c in range(self.n_classes): 
            for t in range(len(SedimSys.Qbi_dep)): 
                q_t = SedimSys.Qbi_dep[t]
                deposited_class[c][t,:] = np.array([np.sum(item[:,c+1], axis = 0) for item in q_t]) 
       
        
        #--Mobilised per class
        mobilised_class =  [np.empty((self.timescale-1, self.n_reaches)) for _ in range(self.n_classes)]
        
        for c in range(self.n_classes): 
            for t in range(self.timescale-1): 
                q_m = SedimSys.Qbi_mob[t][:,:,c]
                mobilised_class[c][t,:] = np.sum(q_m, axis = 0)
    
        #--Transported per class        
        transported_class =  [np.empty((self.timescale-1, self.n_reaches)) for _ in range(self.n_classes)]
        
        for c in range(self.n_classes): 
            for t in range(self.timescale-1): 
                q_m = SedimSys.Qbi_tr[t][:,:,c]
                transported_class[c][t,:] = np.sum(q_m, axis = 0)
                            
        #--Tranport capacity per class (put in same format as mob and trans per class)
        tr_cap_class = [np.empty((self.timescale-1, self.n_reaches)) for _ in range(self.n_classes)]
        for c in range(self.n_classes): 
            for t in range(self.timescale-1): 
                q_m = SedimSys.tr_cap[t,:,c]
                tr_cap_class[c][t,:] = q_m     
        
        #--Critical discharge per class (put in same format as mob and trans per class)
        if self.indx_tr_cap == 7:   
            Qc_class = [np.empty((self.timescale-1, self.n_reaches)) for _ in range(self.n_classes)]
            for c in range(self.n_classes): 
                for t in range(self.timescale-1): 
                    q_m = self.Qc_class_all[t,:,c]
                    Qc_class[c][t,:] = q_m  
                
        Q_out_class = [np.empty((self.timescale-1, self.n_reaches)) for _ in range(self.n_classes)]
        for c in range(self.n_classes): 
            for t in range(self.timescale-1): 
                q_m = SedimSys.Q_out[t,:,c]
                Q_out_class[c][t,:] = q_m 
        
        
        V_sed_class = [np.empty((self.timescale-1, self.n_reaches)) for _ in range(self.n_classes)]
        for t in range(self.timescale-1):
            for c in range(self.n_classes):
                q_m = SedimSys.V_sed[t,:,c]
                V_sed_class[c][t, :] = q_m
            
        #--Total sediment volume leaving the network
        outcum_tot = np.array([np.sum(x) for x in SedimSys.Q_out])
        df = pd.DataFrame(outcum_tot)
        
        #set all NaN transport capacity to 0
        SedimSys.tr_cap_sum[np.isnan(SedimSys.tr_cap_sum)] = 0 
        
        #set all NaN active layer D50 to 0; 
        SedimSys.D50_al[np.isnan(SedimSys.D50_al)] = 0
        SedimSys.D50_al_before_tlag[np.isnan(SedimSys.D50_al_before_tlag)] = 0
        
        Q = np.array(Q)
        
        #--Output struct definition 
        #data_plot contains the most important D_CASCADE outputs 
        data_output = {'Channel Width [m]': np.repeat(np.array(self.reach_data.wac).reshape(1,-1),len(SedimSys.Qbi_dep), axis = 0), 
                       'Reach slope': SedimSys.slope,
                       'Discharge [m^3/s]': Q[0:self.timescale,:],
                       'Mobilized [m^3]': QB_mob_sum,
                       'Transported [m^3]': tot_tranported,
                       'Transported + deposited [m^3]': tot_sed,
                       'D50 deposit layer [m]': D50_dep,
                       'D50 mobilised layer [m]': D50_mob,
                       'D50 active layer before time lag[m]': SedimSys.D50_al_before_tlag,
                       'D50 active layer [m]': SedimSys.D50_al,
                       'Transport capacity [m^3]': SedimSys.tr_cap_sum,
                       'Deposit layer [m^3]': V_dep_sum,
                       # 'Delta deposit layer [m^3]': self.Delta_V_all,
                       'Transported + deposited - per class [m^3]': tot_sed_class,
                       'Deposited - per class [m^3]': deposited_class,
                       'Mobilised - per class [m^3]': mobilised_class,
                       'Transported- per class [m^3]': transported_class,
                       # 'Delta deposit layer - per class [m^3]': self.Delta_V_class,
                       'Transport capacity - per class [m^3]': tr_cap_class,
                       'Sed_velocity [m/day]': SedimSys.V_sed,
                       'Sed_velocity - per class [m/day]': V_sed_class,
                       'Flow depth': SedimSys.flow_depth,
                       'Active layer [m]': SedimSys.al_depth,
                       'Maximum erosion layer [m]': SedimSys.eros_max_depth,
                       'Q_out [m^3]': SedimSys.Q_out,
                       'Q_out_class [m^3]': Q_out_class,
                       'Q_out_tot [m^3]': outcum_tot
                       }
    
        if self.indx_tr_cap == 7:
            data_output["Qc - per class"] = Qc_class
             
        #all other outputs are included in the extended_output cell variable 
        extended_output = {'Qbi_tr': SedimSys.Qbi_tr,
                           'Qbi_mob': SedimSys.Qbi_mob,
                           'Q_out': SedimSys.Q_out,
                           'Qbi_dep': SedimSys.Qbi_dep,
                           'Fi_r_ac': SedimSys.Fi_al,
                           'node_el': SedimSys.node_el
                           }
        
        return data_output, extended_output