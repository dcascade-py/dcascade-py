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


from flow_depth_calc import choose_flow_depth
from slope_reduction import choose_slopeRed
from supporting_classes import Cascade, SedimentarySystem
from supporting_functions import sortdistance, D_finder
from flow_depth_calc import hypso_manning_Q
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

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
        
        #JR extra saving storage
        self.T_record_days = np.zeros(self.timescale) #time storage, days #ccJR
        self.wac_save = np.zeros((self.timescale, self.n_reaches)) #hydraulics storage,  #ccJR
        self.h_save = np.zeros((self.timescale, self.n_reaches)) #hydraulics storage,  #ccJR
        self.v_save = np.zeros((self.timescale, self.n_reaches)) #hydraulics storage,  #ccJR
        self.Vfracsave = np.zeros((self.timescale, self.n_reaches)) #hypso slicing storage,  #ccJR
        
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

    def set_JR_indexes(self, vary_width,vary_roughness,hypsolayers):
        self.vary_width = vary_width
        self.vary_roughness = vary_roughness        
        self.hypsolayers = hypsolayers                              
    
    def check_algorithm_compatibility(self):
        # Constrain on the option of the algorithm:
        if self.passing_cascade_in_trcap == True and self.passing_cascade_in_outputs == False:
            raise ValueError("You can not use this combination of algorithm options")
        if self.time_lag_for_mobilised == True and (self.passing_cascade_in_outputs == False or self.passing_cascade_in_trcap == False):
            raise ValueError("You can not use this combination of algorithm options")
     

      #ccJR have not transferred:  Qbi_dep_0 = [np.zeros((n_layers, n_classes + 1), dtype=np.float32) for _ in range(n_reaches)]  # Assuming 4 layers
    

        
    
    def run(self, Q, roundpar):
        # DD: Should we create a subclass in SedimentarySystem to handle the temporary parameters for one time step
        # like Qbi_pass, Qbi_dep_0 ? Could be SedimentarySystemOneTime ?
        
        SedimSys = self.sedim_sys
        
        # start waiting bar    
        for t in tqdm(range(self.timescale - 1)):
            
            self.T_record_days[t]=t*self.ts_length / (60*60*24);
            
            # TODO: DD see with Anne Laure and Felix, which slope are we using for the flow?
            
            # Define flow depth and flow velocity for all reaches at this time step:
            h, v = choose_flow_depth(self.reach_data, SedimSys.slope, Q, t, self.indx_flo_depth)
            SedimSys.flow_depth[t] = h
            
            if self.vary_width: #ccJR carying width with hydraulic geometry. Replace this with hypsometric hydraulics next. 
                self.wac_save[t,:] =  self.reach_data.width_a[:] * Q[t,:]**self.reach_data.width_b[:]
                #this is a good place to raplace Q[t,n] with a 'single' Q to test the old constant Wac
                self.reach_data.wac[:] = self.wac_save[t,:]   
            
            # Slope reduction functions
            SedimSys.slope = choose_slopeRed(self.reach_data, SedimSys.slope, Q, t, h, self.indx_slope_red)
           
            # deposit layer from previous timestep
            Qbi_dep_old = copy.deepcopy(self.sedim_sys.Qbi_dep_0)
            
            # volumes of sediment passing through a reach in this timestep,
            # ready to go to the next reach in the same time step.
            Qbi_pass = [[] for n in range(self.n_reaches)]
                    
            # loop for all reaches:
            for n in self.network['n_hier']:  
                
                # TODO : How to include the lateral input ? JR attempting this here:
                
            #ccJR - add our inputs to the bed, for the next timestep to deal with. it will at least be available and mass conserving..
            #V_dep_init[0, 1:] += Qbi_input[t,n,:] #didn't work
                if t>1 and SedimSys.Qbi_input[t,n,:].sum()>0:
                    # The mobilised cascade is added to a temporary container
                    et0 = np.ones(self.n_classes) # its elapsed time is 0
                    #how it's done:                 mobilized_cascades.append(Cascade(provenance, elapsed_time, V_mob))
                    arrgh = SedimSys.Qbi_input[t, n, :] #the first dim is the source reach.  
                    Qtoadd =  np.insert(arrgh, 0, 0)
                    Qtoadd = Qtoadd.reshape(1, -1)
                    Qbi_pass[n].append(Cascade(n, et0, Qtoadd)) #ccJR adding source volume here.     
                     
                #TODO: bring in       reach_hypsometry[n] choice in               C:\bin\cascade\dcascade_py-JR_oct29_dWac_dVactive\DCASCADE_loop_vWac_dVact.py
                
                
                #ccJR hypso - REMOVE 'overburden' which is volume of (wacmax - wac) for now. the 2.0 or 1.5 changes the range. adding 1 slicevol keeps it away from the edge.
                if self.hypsolayers:
                    #volume we want to use this timestep. how to keep away from the edges
                    slicevol = Qbi_dep_old[n].sum() *(self.reach_data.wac[n] / self.reach_data.wac_bf[n])
                    
                    #BIG QUESTION from JR for others - which here is the 'right way up?'
                    hypso_V_above = slicevol #from one direction
                    #hypso_V_above = Qbi_dep_old[n].sum()- slicevol # test, from the other direction
                    
                    
                    self.Vfracsave[t,n] = hypso_V_above / Qbi_dep_old[n].sum() #fraction of total volume we are removing, to save.
                    
                    #ccJR thjs definitely needs checking with my understanding of where cascades deposit to 
                    #and erode from. I am trying to set a datum near the 'middle' and let widths alter
                    #where we access teh bed volume.
                    #strip off overburden here, which is thought of as 'under' but I am thinking of as 'farther in width'
                    _,Vdep_init,Vdep_overburden, Fi_slice = SedimSys.layer_search(Qbi_dep_old[n], hypso_V_above, roundpar)
                    
                    if np.isnan(self.Vfracsave[t,n]):
                        print(t,n,self.Vfracsave[t,n])
                        
                else:
                # Extracts the deposit layer left in previous time step          
                    Vdep_init = Qbi_dep_old[n] # extract the deposit layer of the reach 
                

                if self.vary_width:
                    
                    #wac has changed, reset erosmax
                    SedimSys.set_erosion_maximum(SedimSys.eros_max_depth[0,n], roundpar)                           
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
                    
                    #ccJR - we need to dynamically calculate AL volume.
                    SedimSys.al_vol[t, n] = self.reach_data.wac[n] * self.reach_data.length[n] * np.maximum(2*self.reach_data.D84[n], 0.01) 
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
                    # Transport capacity is only calculated on Vdep_init
                    tr_cap_per_s, Fi_al, D50_al, Qc = SedimSys.compute_transport_capacity(Vdep_init, roundpar, t, n, Q, v, h,
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
                        elapsed_time = np.zeros(self.n_classes) # it start its journey at the beginning of the time step
                        provenance = n
                        reach_mobilized_cascades.append(Cascade(provenance, elapsed_time, Vmob))
                        
                    # Remaining time after time lag
                    r_time_lag = 1 - time_lag 
                    
                else:   
                    # If no time lag is used:
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
                # considering eventually the passing cascades during the remaining time:
                tr_cap_per_s, Fi_al, D50_al, Qc = SedimSys.compute_transport_capacity(Vdep, roundpar, t, n, Q, v, h,
                                                                                  self.indx_tr_cap, self.indx_tr_partition,
                                                                                  passing_cascades = passing_cascades,
                                                                                  per_second = True)                
               
                # Store transport capacity and active layer informations: 
                SedimSys.Fi_al[t, n, :] = Fi_al
                SedimSys.D50_al[t, n] = D50_al
                SedimSys.Qc_class_all[t, n] = Qc

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
                    
                # Add the possible reach mobilised cascade to a temporary container
                if Vmob is not None: 
                    if time_lag is None:
                        elapsed_time = np.zeros(self.n_classes) 
                    else: 
                        elapsed_time = time_lag
                    provenance = n
                    reach_mobilized_cascades.append(Cascade(provenance, elapsed_time, Vmob))
                                        

                ###-----Step 3: Finalisation.
                
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
                    # to reproduce v1, we only store the cascade mobilised from the reach
                    for cascade in reach_mobilized_cascades:
                        SedimSys.Qbi_mob[t][[cascade.volume[:,0].astype(int)], n, :] += cascade.volume[:, 1:]     
                        
                # Deposit the stopping cascades in Vdep 
                if to_be_deposited is not None:                   
                    to_be_deposited = sortdistance(to_be_deposited, self.network['upstream_distance_list'][n])
                    Vdep_end = np.concatenate([Vdep_end, to_be_deposited], axis=0)
                    
                # ..and store Vdep for next time step
                #put 'overburden' back
                if self.hypsolayers:
                    SedimSys.Qbi_dep_0[n] = np.concatenate([Vdep_overburden, np.float32(Vdep_end)], axis=0)
                else:
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
                self.Delta_V_class = np.sum(SedimSys.Qbi_dep_0[n][:,1:], axis=0) - np.sum(Qbi_dep_old[n][:,1:], axis=0)
                SedimSys.Delta_V_class_all[t,n,:] = self.Delta_V_class            
                
                # Update slope, if required.
                if self.update_slope == True:
                    self.sedim_sys.node_el[t+1][n]= self.sedim_sys.node_el[t,n] + Delta_V/( np.sum(self.reach_data.wac[np.append(n, self.network['upstream_node'][n])] * self.reach_data.length[np.append(n, self.network['upstream_node'][n])]) * (1-self.sedim_sys.phi) )
                
                vary_roughness_inloop = True 
                #update roughness within the time loop.ccJR - assuming uniform width GSD for now. 
                if self.vary_roughness:
                    #need updated D84 at least
                    #reach_data.D16[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 16, psi)
                    #reach_data.D50[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 50, psi)
                    SedimSys.reach_data.D84[n] = D_finder(SedimSys.Fi_al[t,n,:], 84, SedimSys.psi)  
                    
                    Hbar = h[n]
                    s8f_keul = (1 / 0.41) * np.log((12.14 * Hbar) / (3*SedimSys.reach_data.D84[n]))
                    C_keul = s8f_keul * np.sqrt(9.81)
                    n_keul = Hbar**(1/6) / C_keul
                    SedimSys.reach_data.n[n] = n_keul   
    
            """End of the reach loop"""
            

            
            
            # Save Qbi_dep according to saving frequency
            if self.save_dep_layer == 'always':
                SedimSys.Qbi_dep[t+1] = copy.deepcopy(SedimSys.Qbi_dep_0)            
            if self.save_dep_layer == 'yearly':
                if int(t+2) % 365 == 0 and t != 0:
                    t_y = int((t+2)/365)
                    SedimSys.Qbi_dep[t_y] = copy.deepcopy(SedimSys.Qbi_dep_0)
            if self.save_dep_layer == 'monthhour':
                if int(t+2) % 720 == 0 and t != 0:
                    t_y = int((t+2)/720)
                    SedimSys.Qbi_dep[t_y] = copy.deepcopy(SedimSys.Qbi_dep_0)
                    #save stratigraphy of Fi, by using layer_search.
                    #think really hard about which way is up. layer_search works from the top. nl 0 is the top then?
                    if self.hypsolayers:
                        for n in self.network['n_hier']:            
                            for nl in range(SedimSys.n_layers):  #this will loop in increments. could go finer. and could turn into the whole hypso transport curve.
                                slicevol = (nl+1)* (SedimSys.Qbi_dep_0[n].sum() / SedimSys.n_layers)  #even slices
                                _,V_dep_init,V_dep_overburden, Fi_slice = SedimSys.layer_search(SedimSys.Qbi_dep_0[n], slicevol, roundpar)
                                D50_custom = D_finder(Fi_slice, 50, SedimSys.psi) 
                                #print(1000*D50_custom, (Fi_slice[5:7].sum()*100).round(3) ) #great - the sand is at the top,
                                SedimSys.Qbi_FiLayers[t_y,n,nl,:] = Fi_slice #might lose a bit at the bottom if major aggradation?
                    else:
                        for n in self.network['n_hier']: 
                            #handle saving an active layer Fi at save_dep_layer times. 
                            _,_,_, Fi_slice = SedimSys.layer_search(SedimSys.Qbi_dep_0[n], SedimSys.al_vol[t, n], roundpar)
                            SedimSys.Qbi_FiLayers[t_y,n,0,:] = Fi_slice                    
                
            # in case of changing slope..
            if self.update_slope == True:
                #..change the slope accordingly to the bed elevation
                self.sedim_sys.slope[t+1,:], self.sedim_sys.node_el[t+1,:] = SedimSys.change_slope(self.sedim_sys.node_el[t+1,:], self.reach_data.length, self.network, s = self.sedim_sys.min_slope)
                    
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
                    q_m = SedimSys.Qc_class_all[t,:,c]
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
        data_output = {'wac': self.wac_save, #
                       'slope': SedimSys.slope,   #
                       'Q': Q[0:self.timescale,:],  #Discharge [m^3/s]
                       'QB_mob_sum': QB_mob_sum, #Mobilized [m^3]
                       'tot_tranported': tot_tranported, # DD: instead have what is deposited or stopping
                       'tot_sed': tot_sed,  # Transported + deposited [m^3]
                       'D50_dep': D50_dep, # D50 deposit layer [m]
                       'D50_mob': D50_mob,
                       'D50_AL_preTL': SedimSys.D50_al_before_tlag, # depending on the option
                       'D50_AL': SedimSys.D50_al,
                       'tr_cap_sum': SedimSys.tr_cap_sum, #Transport capacity [m^3]
                       'V_dep_sum': V_dep_sum, #Deposit layer [m^3]
                       # 'Delta deposit layer [m^3]': self.Delta_V_all, # --> add the budget
                       'tot_sed_class': tot_sed_class, #Transported + deposited - per class [m^3]
                       'deposited_class': deposited_class, # Deposited - per class [m^3] flag per class ?
                       'mobilised_class': mobilised_class, #Mobilised - per class [m^3]
                       'transported_class': transported_class, #Transported - per class [m^3]
                       'Delta_V_class_all': SedimSys.Delta_V_class_all, #Delta deposit layer - per class [m^3] TODO ccJR why this out?
                       'tr_cap_class': tr_cap_class, #Transport capacity - per class [m^3]
                       'V_sed': SedimSys.V_sed, #Sed_velocity [m/day]
                       'V_sed_class': V_sed_class, #Sed_velocity - per class [m/day]
                       'flow_depth': SedimSys.flow_depth, #
                       'al_depth': SedimSys.al_depth, # rename
                       'eros_max_depth': SedimSys.eros_max_depth, #Maximum erosion layer [m]
                       # output to say when we reach the maximum erosion layer
                       'Q_out': SedimSys.Q_out, # Q_out [m^3]
                       'Q_out_class': Q_out_class, # [m^3]
                       'outcum_tot': outcum_tot #  [m^3]
                       }
    
        if self.indx_tr_cap == 7:
            data_output["Qc - per class"] = Qc_class
        dmi = 2**(-SedimSys.psi).reshape(-1,1)     
        #all other outputs are included in the extended_output cell variable 
        extended_output = {'Qbi_tr': SedimSys.Qbi_tr,
                           'Qbi_mob': SedimSys.Qbi_mob,
                           'Q_out': SedimSys.Q_out,
                           'Qbi_dep': SedimSys.Qbi_dep,
                           'Fi_r_ac': SedimSys.Fi_al,  #
                           'Node_el': SedimSys.node_el, # return if the option update_slope is true
                           'Length': SedimSys.reach_data.length,
                            'D16': SedimSys.reach_data.D16,
                            'D50': SedimSys.reach_data.D50,
                            'D84': SedimSys.reach_data.D84,
                            'psi': SedimSys.psi,
                            'dmi': dmi,
                            'Tdays': self.T_record_days,
                            'Vfracsave': self.Vfracsave,
                            'v_save': self.v_save,
                            'h_save': self.h_save,
                            'Qbi_FiLayers': SedimSys.Qbi_FiLayers,
                           }
        
        
        return data_output, extended_output
