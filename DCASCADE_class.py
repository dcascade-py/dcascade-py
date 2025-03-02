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
from flow_depth_calc import hypso_manning_Q, hypso_ferguson_Q_vec
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fsolve
from line_profiler import profile

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
        self.h_save = np.zeros((self.timescale, self.n_reaches, 200)) #hydraulics storage,  #ccJR
        self.v_save = np.zeros((self.timescale, self.n_reaches, 200)) #hydraulics storage,  #ccJR
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
    

        
    @profile    
    def run(self, Q, roundpar):
        # DD: Should we create a subclass in SedimentarySystem to handle the temporary parameters for one time step
        # like Qbi_pass, Qbi_dep_0 ? Could be SedimentarySystemOneTime ?
        
        SedimSys = self.sedim_sys
        
    
        #define some reach hypso constants once. Move this to class defintions once working   
        
        heights_at_Xgrid = {}  # Store heights for each reach
        q_to_H_interp_func = {}
        dX = 5 #stable with 2. Storage set for dX = 5 
        rounded_max_width = np.ceil(max(SedimSys.reach_data.wac_bf) / dX) * dX
        SedimSys.Xgrid =  np.arange(0, rounded_max_width + dX, dX) #width regular grid now
        for n in self.network['n_hier']:
            if SedimSys.reach_hypsometry[n] == True:     
            # Precompute a 2D interpolant
                heights_at_flow = []  # Store heights for each flow rate for this reach
                for i, qstep in enumerate(SedimSys.qsteps):
                    # Interpolate height for this flow rate and reach
                    width_interp_func = interp1d(
                        SedimSys.Wvec_q[n, i, :],  # Widths for this flow
                        SedimSys.Hvec[n],          # Heights corresponding to widths
                        bounds_error=False,
                        fill_value=10,
                    )
                    heights_at_flow.append(width_interp_func(SedimSys.Xgrid))  # Interpolate on self.Xgrid
                heights_at_Xgrid[n] = (np.array(heights_at_flow))  # Save array for this reach
                qsteps_0 = SedimSys.qsteps; qsteps_0[0]=1 #need a lower minimum as was going out of bounds. 
                interp_func = RegularGridInterpolator(
                   (qsteps_0, SedimSys.Xgrid),
                   heights_at_Xgrid[n],  # Precomputed heights
                   bounds_error=False,
                   fill_value=10,
                   )
                
                q_to_H_interp_func[n] = interp_func
                #test: this should give heights at every Xgrid:
                #q_to_H_interp_func[n]((120, self.Xgrid))
                
                    
        # start waiting bar
        etasave = np.zeros(self.n_reaches)
        for t in tqdm(range(self.timescale - 1)):
            
            self.T_record_days[t]=t*self.ts_length / (60*60*24);
            
            # TODO: DD see with Anne Laure and Felix, which slope are we using for the flow?
            
            #Have not removed choose_flow_depth calls; we may need wac defined BEFORE that call, and we still use hydraulic
            #geom to intialize that even if later we use the hypso solver. 
            if self.vary_width: #ccJR carying width with hydraulic geometry. Replace this with hypsometric hydraulics in reaches with data. 
                self.wac_save[t,:] =  self.reach_data.width_a[:] * Q[t,:]**self.reach_data.width_b[:]
                self.wac_save[t,:] = np.minimum(self.wac_save[t,:],self.reach_data.wac_bf) #limit hydr geom to max
                #this is a good place to raplace Q[t,n] with a 'single' Q to test the old constant Wac
                self.reach_data.wac[:] = self.wac_save[t,:]   
            
            SedimSys.slope[t,4] = 0.01 #ccJR HARDCODED gorge reach slope is underestimated, need to move sand along...
            SedimSys.slope[t,3] = 0.002 #ccJR HARDCODED above reach slope is underestimated, need to move gravel...
            
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
                
                if SedimSys.reach_hypsometry[n] == True:
                    #reach_hypsometry_data[n]['Zvec']
                    #redo h_ferguson to run a Q loop, redolfi style. 
                    #print(SedimSys.Hvec[n])
                    
                    
                    #Zgrid1D = Xinterp_func(Xgrid)
                    Zgrid = q_to_H_interp_func[n]((Q[t,n], SedimSys.Xgrid))  #now interpolates at this q. One function for each reach, pre-allocated.
                    
                    #Zgrid = np.nan_to_num(Zgrid, posinf=10, neginf=10) 
                    #print(Zgrid)
                    
                    #try this with truncated / trimmed Zgrid?  8.5 it/s before. 
                
                    def func(D):
                        if self.indx_flo_depth==1:
                            Q_Eng, b_Eng, JS = hypso_manning_Q(D, Zgrid, dX, self.reach_data.n[n], SedimSys.slope[t,n])
                        if self.indx_flo_depth==2:
                            Q_Eng, b_Eng, JS = hypso_ferguson_Q_vec(D, Zgrid, dX, self.reach_data.C84_fac[n] *self.reach_data.D84[n], SedimSys.slope[t,n])
                            
                        return (Q_Eng/Q[t,n])-1
            
                # Solve for eta using fsolve. needed a higher guess for some reaches to converge - 4x Manning for now?
                    if t>1:
                        hguess = etasave[n] # max(self.h_save[t-1,n])
                    else:
                        hguess = 3*h[n] #eta is bigger by far than the average depth. 
                    try:
                        eta, info, ier, msg = fsolve(func, hguess, full_output=True)  #13.5% of all time
                        if ier == 1:
                            etasave[n] = eta
                        if ier != 1:
                            print("Reach", n, " did not converge,",h[n],eta, ". Message:", msg)
                            # Handle the case where fsolve did not converge, for example:
                            eta = etasave[n]  # prior fallback value, improve if there are many of these?
                    except Exception as e:
                        print("fsolve fail:", e)
                        raise    
                    #use eta to get h 
                    if self.indx_flo_depth==1:
                        Q_Eng, b_Eng, JS = hypso_manning_Q(eta, Zgrid, dX, self.reach_data.n[n], SedimSys.slope[t,n])
                    if self.indx_flo_depth==2:
                        Q_Eng, b_Eng, JS = hypso_ferguson_Q_vec(eta, Zgrid, dX, self.reach_data.C84_fac[n] *self.reach_data.D84[n], SedimSys.slope[t,n])    
                    #WHERE TO FROM HERE. I now have a variety of h,v data. 
                    #Cut off zeroes here. JS retains full length which is good for saving consistent matrix. 
                    # Remove trailing zeros
                    Vsave_trimmed = np.trim_zeros(JS['Vsave'], 'b')        
                    Hsave_trimmed = np.trim_zeros(JS['Hsave'], 'b')        
                    #Csave_trimmed = np.trim_zeros(JS['Csave'], 'b') #do I need this? 
                    #I can destroy the new information by making h = mean(JS.) for nonzero returns. let's try that first. 
                    #print([reach_data.wac[n]] / b_Eng )
                    try:
                        # Code that might throw an error
                        self.v_save[t, n, :len(Vsave_trimmed)] = Vsave_trimmed
                    except Exception as e:
                        print(f"Vsave_trimmed shape: {Vsave_trimmed.shape}")
                        #error here? the 'floodplain 10m hypsometry' issue occurred when slope was reduced (unrealistically)
                        print(e)
                    if len(Vsave_trimmed)>150:
                        print(n,t,len(Vsave_trimmed))
                        
                    self.v_save[t,n,:len(Vsave_trimmed)] = Vsave_trimmed
                    self.h_save[t,n,:len(Hsave_trimmed)] = Hsave_trimmed
                    #errors here - did not allocate enough spaec for river width. 
                    
                    #print('v,h frac of manning:', self.v_save[t,n]/v[n], self.h_save[t,n]/h[n])
                    #self.h_save[t,n]/h[n]
                    
                    h[n] = Hsave_trimmed.mean()
                    v[n] = Vsave_trimmed.mean()
                    self.reach_data.wac[n] = b_Eng  #overwrites hydraulic geom wac
                    self.wac_save[t,n] = b_Eng #and save for output
                    SedimSys.flow_depth[t,n] = h[n]
                    #can I summarize by my height bins? or are we working in width now?
                    #print(n,Vsave_trimmed)
                else: #save normal Manning h and v
                    self.h_save[t,n,0] = h[n]
                    self.v_save[t,n,0] = v[n]     
                
                
                    
            #ccJR - add our inputs to the bed, for the next timestep to deal with. it will at least be available and mass conserving..
            #V_dep_init[0, 1:] += Qbi_input[t,n,:] #didn't work
                if t>1 and SedimSys.Qbi_input[t,n,:].sum()>0:
                    # The mobilised cascade is added to a temporary container
                    et0 = np.zeros(self.n_classes) # its elapsed time is 0
                    #how it's done:                 mobilized_cascades.append(Cascade(provenance, elapsed_time, V_mob))
                    arrgh = SedimSys.Qbi_input[t, n, :] #the first dim is the source reach.  
                    Qtoadd =  np.insert(arrgh, 0, n)
                    Qtoadd = Qtoadd.reshape(1, -1)
                    Qbi_pass[n].append(Cascade(n, et0, Qtoadd)) #ccJR adding source volume here.     
                     
                #TODO: bring in       reach_hypsometry[n] choice in               C:\bin\cascade\dcascade_py-JR_oct29_dWac_dVactive\DCASCADE_loop_vWac_dVact.py
                
                
                #ccJR hypso - REMOVE 'overburden' which is volume of (wacmax - wac) for now. the 2.0 or 1.5 changes the range. adding 1 slicevol keeps it away from the edge.
                if self.hypsolayers:
                    #volume we want to use this timestep. how to keep away from the edges
                    #slicevol = Qbi_dep_old[n].sum() *(self.reach_data.wac[n] / self.reach_data.wac_bf[n])
                    #change slicevol to be explicit from reach_data.deposit * reach_data.length
                    #rev8 - was slicing off too much?
                    #slicevol = self.reach_data.deposit[n] * SedimSys.n_layers * self.reach_data.length[n] * self.reach_data.wac[n]  
                    slicevol = self.reach_data.deposit[n] * self.reach_data.length[n] * self.reach_data.wac[n]  
                    
                    #Todo ccJR SUM is including the 0,1,2 provenance! index from 1:end (and check everywhere!)
                    
                    #OK! think it's sorting through cascades in the way I want it to. Now, totalV * widthratio means
                    #we would always use the whole volume at full width. I want to be careful changing active height as it 
                    #would mess with my accounting: if active layer ht changed during run, the scaling of width vs volume would
                    #magically alter. but how to 
                    
                    #BIG QUESTION from JR for others - which here is the 'right way up?'
                    hypso_V_above = slicevol #from one direction
                    #hypso_V_above = Qbi_dep_old[n].sum()- slicevol # test, from the other direction
                    
                    
                    self.Vfracsave[t,n] = hypso_V_above / Qbi_dep_old[n].sum() #fraction of total volume we are removing, to save.
                    
                    #ccJR thjs definitely needs checking with my understanding of where cascades deposit to 
                    #and erode from. I am trying to set a datum near the 'middle' and let widths alter
                    #where we access teh bed volume.
                    #strip off overburden here, which is thought of as 'under' but I am thinking of as 'farther in width'
                    
                    #if width is large, we slice off a large Vdep_init from Qbi_dep_old. width->volume based length * 1m active layer
                    #but since actual active layer is smaller, I'm using too small a proportion. 
                    _,Vdep_init,Vdep_overburden, Fi_slice = SedimSys.layer_search(Qbi_dep_old[n], hypso_V_above, roundpar)
                    
                    #Vdep_init = np.flipud(Vdep_init) #this should 'insert' new cascades on the correct 'side' of Vdep_init. YES, worked, i think. 
                    
                    #needs flipped back below. 
                    if np.isnan(self.Vfracsave[t,n]):
                        print(t,n,self.Vfracsave[t,n])
                        
                else:
                # Extracts the deposit layer left in previous time step          
                    Vdep_init = Qbi_dep_old[n] # extract the deposit layer of the reach 
                

                if self.vary_width:
                    
                    #wac has changed, reset erosmax. THIS proved limiting, never letting river access fines deposited at large widths.
                    #SedimSys.set_erosion_maximum(SedimSys.eros_max_depth[n], roundpar)      
                    #ccJR TESTING HARDCODED there is no eros max vol, is whole reach volume. 
                    #Idea - limit this to competent width (of previous timestep I guess)
                    #i could have hypso transport save that width as a bed volume. 
                    self.eros_max_vol = np.round(Vdep_init.sum(), roundpar).astype(np.float32)     
                    
                    # self.eros_max_vol / sum(sum(Vdep_init[1:]))
                    
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
                    
                    #this will produce a sediment step length based on mean rectancular hydraulics. 
                    if self.hypsolayers:
                        SedimSys.al_vol[t, n] = slicevol
                    else:
                        SedimSys.al_vol[t, n] = np.float32(self.reach_data.wac[n] * self.reach_data.length[n] * np.maximum(2*self.reach_data.D84[n], 0.01) )
                        
                    SedimSys.compute_cascades_velocities(Qbi_pass[n], Vdep_init,
                                               Q[t,n], v[n], h[n], roundpar, t, n,                           
                                               self.indx_velocity, self.indx_vel_partition, 
                                               self.indx_tr_cap, self.indx_tr_partition)

                
                # Decides whether cascades, or parts of cascades, 
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
                    print('warning: JR did not flip around any of this time lag Vdep')                   
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
                    #JR - work here if not doing time lag. which not, for testing. 
                tr_cap_per_s, Fi_al, D50_al, Qc = SedimSys.compute_transport_capacity(Vdep, roundpar, t, n, Q, v, h,
                                                                                  self.indx_tr_cap, self.indx_tr_partition,
                                                                                  passing_cascades = passing_cascades,
                                                                                  per_second = True) 
                #for hypso, we need a width, a Fs which we'll calculate here first, and local h,v
                #Also need Q proportion? h*v*h_wac
                  #  self.v_save[t,n] = JS['Vsave'].mean()
                  #  self.h_save[t,n] = JS['Hsave'].mean()    
                if SedimSys.reach_hypsometry[n] == True:  
                     
                    Xwac = SedimSys.Xgrid[:len(Vsave_trimmed)+1].copy() #ccJR this was altering Xgrid without the copy!!   
                    Xwac[-1] = self.reach_data.wac[n] #last cell lengthen to match width
                    #31% of all time! could be vectorized somehow?
                    #extra return - w_inac is FRACTION of Vdep within which 95% of the cumulative transport occurred. For where to deposit, below. 
                    htr_cap_per_s, Fi_al, D50_al, Qc, w_inac = SedimSys.hypso_transport_capacity(Vdep, roundpar, t, n, Q, Xwac,Vsave_trimmed, Hsave_trimmed,
                                                                                  self.indx_tr_cap, self.indx_tr_partition,
                                                                                  passing_cascades = passing_cascades,
                                                                                  per_second = True)
                    #print('Qs ratio', htr_cap_per_s / tr_cap_per_s) #JR todo - save this by class
                    #or just save both for reporting and comparison later
                    tr_cap_per_s = htr_cap_per_s
                # Store transport capacity and active layer informations: 
                    #TODO these are saved but represent the last loop (shallowest) so perhaps are inaccurate. they don't appear used though.
                    #I could average them in the hypso_transport_capacity loop or not worry atm. 
                    #if n==13 and int(t) % 450 == 0:  #should hit some low and the max flows DIAGNOSTIC output
                if n==11 and int(t) % 450==0 and t>1:  #should hit some low and the max flows DIAGNOSTIC output
                    print('Diag Fi at n=',n,' t=',t)
                    print(Fi_al)
                    #how is  input looking vs. trcap?
                    print(tr_cap_per_s*3600)
                    print(arrgh)
                    print(tr_cap_per_s*3600 - arrgh[:])
                        
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
                if SedimSys.reach_hypsometry[n] == True: 
                    #MobFlipVdep = np.flipud(Vdep) # IF we have reach hypso, mobilize from the deepest part of the channel (SECOND flip returns to original)
                    #NEW 14Feb ccJR, to allow simultaneous erosion / deposition but limit erosion - only search the volume where active   
                    SedimSys.eros_max_vol[n] = 0.95 * w_inac *np.sum(np.sum(Vdep[:,1:]))
                else:
                    #MobFlipVdep = np.flipud(Vdep) #I still have width variation. So work from the thalweg. I can't turn off hypsolayers I guess. rename it horizontal?
                    SedimSys.eros_max_vol[n] = np.sum(np.sum(Vdep[:,1:]))
                    #ccJR - issue? can this routine deposit? I don't think so. 
                
                Vmob, passing_cascades, Vdep_end = SedimSys.compute_mobilised_volume(Vdep, tr_cap_per_s, 
                                                                                n, roundpar,
                                                                                passing_cascades = passing_cascades,
                                                                                time_fraction = r_time_lag)
                
                #and flip it back, so we DEPOSIT in between layers. Do this everywhere to solve non-hypso reach issues Rev20
                #if SedimSys.reach_hypsometry[n] == True: 
                #Vdep_end = np.flipud(Vdep_end)
                #no else, leave it. 
                
                if passing_cascades is not None:
                   for cascade in passing_cascades:
                        print('passing from U/S:',t,n,cascade.volume[:, :])
                    
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
                
                        
                # Deposit the stopping cascades in Vdep - ccJR did negative occur here? 
                if to_be_deposited is not None:                   
                    to_be_deposited = sortdistance(to_be_deposited, self.network['upstream_distance_list'][n])
                    if n>6 and np.sum(to_be_deposited[:,-1]) > 100: #  and int(t) % 450 == 0:  #should hit some low and the max flows at 2448
                        print('Diag to_be_deposited at n=',n,' t=',t)
                        print(to_be_deposited)
                    
                    #ccJR - testing code to deposit NOT at the water's edge, which is sequestering too much stuff way out wide,
                    #but at some intermediate slice. I can probably fix a lot of this flipping up and downas I understand things better
                    #take off 1/4 of it for now, replace with something from hypso transport calc that's smart? where d50 goes incompetent?
                    #todo: make sure slicevol  exists for non-hypso solver reaches?
                    # htr_cap_per_s search D50 incompetent:
                    #print(htr_cap_per_s)
                    
                    if self.hypsolayers:
                                                
                        
                        if SedimSys.reach_hypsometry[n] == True: 
                            if n==11 and int(t) == 450:  #should hit some low and the max flows
                                print('diag competent edge search n=',n,' t=',t)
                                print(w_inac)   
                            #Rev14 - deposit where transport ceases (defined as 95% of transport threshold bed width(vol))
                            _,Vdep_act,Vdep_inact, Fi_slice = SedimSys.layer_search(Vdep_end, w_inac*slicevol, roundpar)
                                
        #REV30 - distribute sand across ALL WET but INACTIVe cells, evenly.
                            #loop by dx slices of bed (need to know nslices for even division)
                            singleslicevol = self.reach_data.deposit[n] * self.reach_data.length[n] * dX
                            #singleslicevol / sum(sum(Vdep_inact[:,1:]))
                            nslices = np.ceil(1/(singleslicevol / np.sum(np.sum(Vdep_inact[:,1:]))))
                            
                            #for each row in to_be_deposited,
                            # Loop through each row in to_be_deposited
                            to_be_deposited_backup = np.copy(to_be_deposited)
                            #debug  to_be_deposited = np.copy(to_be_deposited_backup)
                            Vdep_remaining = np.copy(Vdep_inact)  # Initialize once before looping
                            for cl in range(to_be_deposited.shape[0]):  # Iterating over columns (sediment classes)
                                
                                sand_to_distribute = np.float32(to_be_deposited[cl, SedimSys.sand_indices+1])  # Get sand from the current column, MAKE SURE TO COPY with float
                                sand_increment = sand_to_distribute / nslices
                                sandsource = to_be_deposited[cl,0]
                                to_be_deposited[cl, SedimSys.sand_indices+1] = 0 #KILL the sane from tobedepostited
                                
                                Vdep_inact_new = np.zeros((0, Vdep_inact.shape[1]))
                                for sl in range(int(nslices)): 
                                    # slice off a set volume, look for a row to add onto. 
                                    _,Vdep_slice,Vdep_remaining, _ = SedimSys.layer_search(Vdep_remaining, singleslicevol, roundpar+2) #need a bit more precision here
                                    
            #before this compact: started nearly 20 it/s on my machine,  9-10 at 50%, and the end
                                    Vdep_slice = SedimSys.matrix_compact(Vdep_slice)
             #after this compact: approaching 20 it/s the whole time! saves search speedup, 
                                    
                                    slfind = np.where(Vdep_slice[:, 0] == sandsource)[0] 
                                    if slfind.size > 0:  # Match the source
                                        
                                        Vdep_slice[slfind[0], -1] += sand_increment  # Add to only first found cascade
                                       
                                        Vdep_inact_new = np.concatenate([Vdep_inact_new, Vdep_slice]) 
                                    else:    
                                        new_row = np.zeros((1, Vdep_inact.shape[1]))
                                        new_row[0,0] =sandsource
                                        new_row[0,-1] = sand_increment  # Distribute sand
                            
                                        # Append the new row to Vdep_inact_new
                                        Vdep_inact_new = np.concatenate([Vdep_inact_new,new_row,Vdep_slice])    
                                    #end sl loop
                                
                                Vdep_remaining = np.copy(Vdep_inact_new) #after going through all slices - we go on to next 'cl' cascade list, so we reset new vdep_remaining
                                #end cl loop    
                                    
                        else:  #no reach hypso
                            #Rev13 and reaches without hypso transport: - deposit at 75% of width(volume). 
                            _,Vdep_act,Vdep_inact_new, Fi_slice = SedimSys.layer_search(Vdep_end, 0.75*slicevol, roundpar)

                        #print(Vdep_act.sum(),to_be_deposited.sum(),Vdep_inact.sum(),Vdep_overburden.sum())
                        Vdep_end = np.concatenate([Vdep_inact_new, to_be_deposited, Vdep_act, ], axis=0)
                        #PUT THE GRAVEL where we did before, edge of active. SAND distributed above. 
                        #Vdep_end = np.concatenate([Vdep_end, to_be_deposited], axis=0)
                    
                if np.any(Vdep_end < 0):
                    # debugging
                    print("Negative values found in Vdep_end!")
                    
                # ..and store Vdep for next time step
                #put 'overburden' back
                if self.hypsolayers:
                    #Vdep_end = np.flipud(Vdep_end) #Flipping back to original direction
                    SedimSys.Qbi_dep_0[n] = np.concatenate([Vdep_overburden, np.float32(Vdep_end)], axis=0)
                    
                    #ccJR hardcoded checks, these negatives are coming from somewhere but I can't seem to explore CASCADE objects
                    SedimSys.Qbi_dep_0[n][SedimSys.Qbi_dep_0[n]<0] = 0
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
                if n==6 and int(t) % 450==0 and t>1:  #should hit some low and the max flows
                   

                #for some reason, there is material over at the far edge. Look for it and pause.
                    #searchvl = 1/SedimSys.n_layers * sum(sum(SedimSys.Qbi_dep_0[n]))
                    #_,Vdep_edgesearch,Vdep_other, Fi_es = SedimSys.layer_search(np.flipud(SedimSys.Qbi_dep_0[n]), searchvl, roundpar)    
                    #if sum(Vdep_edgesearch[:,7]) > 10:  #there is mass 
                    #    print('EDGE SAND WHY HERE')
                        # vfoo = SedimSys.Qbi_dep_0[n]
                    print('Delta_V n=',n,' t=',t)
                    print(Delta_V)
                    print(np.sum(SedimSys.Qbi_dep_0[n][:,1:], axis=0), )  
                    print( np.sum(Qbi_dep_old[n][:,1:], axis=0))
                    print(np.sum(SedimSys.Qbi_dep_0[n][:,1:], axis=0) - np.sum(Qbi_dep_old[n][:,1:], axis=0))
                        
                    
                # self.Delta_V_all[t,n] = Delta_V 
                # # And Delta_V per class
                self.Delta_V_class = np.sum(SedimSys.Qbi_dep_0[n][:,1:], axis=0) - np.sum(Qbi_dep_old[n][:,1:], axis=0)
                SedimSys.Delta_V_class_all[t,n,:] = self.Delta_V_class            
                
                # Update slope, if required.
                if self.update_slope == True:
                    self.sedim_sys.node_el[t+1][n]= self.sedim_sys.node_el[t,n] + Delta_V/( np.sum(self.reach_data.wac[np.append(n, self.network['upstream_node'][n])] * self.reach_data.length[np.append(n, self.network['upstream_node'][n])]) * (1-self.sedim_sys.phi) )
                
              
                #update roughness within the time loop.ccJR - assuming uniform width GSD for now. 
                if self.vary_roughness:
                    #need updated D84 at least
                    #reach_data.D16[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 16, psi)
                    #reach_data.D50[n] = D_finder(extended_output['Fi_r_ac'][timescale-2][n], 50, psi)
                    SedimSys.reach_data.D84[n] = D_finder(SedimSys.Fi_al[t,n,:], 84, SedimSys.psi).astype(np.float32)
                    
                    Hbar = h[n]  #JR updated C84 to be a reachdata variable
                    s8f_keul = (1 / 0.41) * np.log((12.14 * Hbar) / (SedimSys.reach_data.C84_fac[n]*SedimSys.reach_data.D84[n])) #make the 3 a passable parameters? ccJR
                    C_keul = s8f_keul * np.sqrt(9.81)
                    n_keul = Hbar**(1/6) / C_keul
                    SedimSys.reach_data.n[n] = n_keul   
    
            """End of the reach loop"""
            

            
            
            # Save Qbi_dep according to saving frequency
            # Determine whether to perform mass balance calculation
            domassbalcalc = False
            t_y = None  # Initialize t_y
            
            if self.save_dep_layer == 'always':
                domassbalcalc = True
                t_y = t + 1  # Always save every timestep
            
            elif self.save_dep_layer == 'yearly' and int(t + 2) % 365 == 0 and t != 0:
                domassbalcalc = True
                t_y = int((t + 2) / 365)
            
            elif self.save_dep_layer == 'monthhour' and int(t + 2) % 720 == 0 and t != 0:
                domassbalcalc = True
                t_y = int((t + 2) / 720)
            
            # Perform mass balance calculation and saving if needed
            if domassbalcalc:
                SedimSys.Qbi_dep[t_y] = copy.deepcopy(SedimSys.Qbi_dep_0)
            
                if self.hypsolayers:
                    for n in self.network['n_hier']:   
                        mass_after = 0
                        V_dep_remaining = np.flipud(SedimSys.Qbi_dep_0[n])  #work from 
                        mass_before = V_dep_remaining[:, 1:].sum()
                        singleslicevol = mass_before / SedimSys.n_layers
            
                        for nl in range(SedimSys.n_layers):
                            _, V_dep_slice, V_dep_remaining, Fi_slice = SedimSys.layer_search(
                                V_dep_remaining, singleslicevol, roundpar, None, False
                            )
                            mass_after += V_dep_slice[:, 1:].sum()
            
                            if np.any(np.isnan(V_dep_remaining)):
                                V_dep_remaining[np.isinf(V_dep_remaining) | np.isnan(V_dep_remaining)] = 0
            
                            SedimSys.Qbi_FiLayers[t_y, n, nl, :] = Fi_slice
                    qfoo = SedimSys.Qbi_FiLayers[t_y,4,:,:]
                    if sum(qfoo[0:10, 6]) > 0:
                        print('bottomsand \r\n',sum(qfoo[0:10, :]))
                    
                else:
                    for n in self.network['n_hier']:
                        _, _, _, Fi_slice = SedimSys.layer_search(
                            SedimSys.Qbi_dep_0[n], SedimSys.al_vol[t, n], roundpar
                        )
                        SedimSys.Qbi_FiLayers[t_y, n, 0, :] = Fi_slice        
                        
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
                           'Fi_r_ac': SedimSys.Fi_al,  #this is the active Fi on the FINAL t of the simulation, at that t's Q. 
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
                            'X_save': SedimSys.Xgrid,
                           }
        
        
        return data_output, extended_output
