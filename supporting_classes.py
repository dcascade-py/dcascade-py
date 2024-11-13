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

from transport_capacity_computation import TransportCapacityCalculator 
from supporting_functions import D_finder, tr_cap_deposit, deposit_from_passing_sediments

# Supporting functions
# from supporting_functions import D_finder
# from supporting_functions import D_finder, sortdistance, matrix_compact, change_slope
# from supporting_functions import layer_search, tr_cap_deposit
# from supporting_functions import cascades_end_time_or_not
# from supporting_functions import deposit_from_passing_sediments
# from supporting_functions import compute_time_lag

# from transport_capacity_computation import compute_transport_capacity, compute_cascades_velocities

# from flow_depth_calc import choose_flow_depth
# from slope_reduction import choose_slopeRed




class Cascade:
    def __init__(self, provenance, elapsed_time, volume):
        self.provenance = provenance
        self.elapsed_time = elapsed_time
        self.volume = volume
        # To be filled during the time step
        self.velocities = np.nan


           
        
class ReachData:
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
        
        self.rugosity = self.compute_rugosity()
        
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
    
    def compute_rugosity(self):
        # DD: idea of function for the rugosity. 
        # to test and see if it is what we want in terms of physics
        if 'rugosity' in self.geodf:
            rugosity = self.geodf['rugosity'].astype(float).values
        elif 'D90' in self.geodf:
            rugosity = self.geodf['D90'].astype(float).values
        else:
            rugosity = self.geodf['D84'].astype(float).values
        return rugosity




class SedimentarySystem:
    ''' Class for managing sediment exchanges, reassembling, and storing
    '''
    #TODO: (DD) tests must be made on eros max, al, and velocity_height 
    # to make sure that they don't contain Nans
    #TODO: Better think on how the matrices should be initialised.
    # For example the vel_section_height should it not be directly set as a 2D zero
    # matrice in the constructor ? (I did it)
    
    def __init__(self, reach_data, network, timescale, ts_length, save_dep_layer, 
                 update_slope, psi, phi = 0.4, minvel = 0.0000001):
        
        self.reach_data = reach_data
        self.network = network
        self.timescale = timescale
        self.ts_length = ts_length
        self.save_dep_layer = save_dep_layer
        self.update_slope = update_slope
        self.n_classes = len(psi)
        self.n_reaches = reach_data.n_reaches
        self.psi = psi
        self.phi = phi                          # sediment porosity
        self.minvel = minvel
        self.outlet = int(network['outlet'])    # outlet reach ID identification
        
        # Setted variables      
  
        
        # Storing matrices
        self.Qbi_dep = None
        self.Qbi_tr = None
        self.Qbi_mob = None
        self.V_sed = None
        self.Q_out = None
        self.slope = None
        self.node_el = None
        self.flow_depth = None
        self.tr_cap = None 
        self.tr_cap_before_tlag = None
        self.tr_cap_sum = None  
        self.Fi_al = None  
        self.Fi_al_before_tlag = None
        self.D50_al = None  
        self.D50_al_before_tlag = None
        self.eros_max_vol = None
        self.eros_max_depth = None
        self.al_vol = None
        self.al_depth = None 
        self.vl_height = self.create_2d_zero_array()
        
        
        # temporary ?
        self.Qbi_dep_0 = None 
        
        self.Qc_class_all = None        # DD: can it be optional ?
        # self.Delta_V_class_all = None   # DD: To be removed        
        # self.Delta_V_all = self.create_2d_zero_array()  # reach mass balance (volumes eroded or deposited)

        
    def create_3d_zero_array(self):
        return np.zeros((self.timescale, self.n_reaches, self.n_classes))
    
    def create_2d_zero_array(self):
        return np.zeros((self.timescale, self.n_reaches))
        
    def initialize_slopes(self):
        self.min_slope = min(self.reach_data.slope)  # put a minimum value to guarantee movement 
        self.slope = self.create_2d_zero_array()
        self.slope[0,:] = np.maximum(self.reach_data.slope, self.min_slope)
        self.slope[1,:] = np.maximum(self.reach_data.slope, self.min_slope)
        
        # In case of constant slope
        if self.update_slope == False:
            self.slope[:,:] = self.slope[0,:]
        
    def initialize_elevations(self):    
        # Initialize node elevation (for each reach the matrix reports the fromN elevation)
        # The last column reports the outlet ToNode elevation (last node of the network), 
        # which can never change elevation.
        self.node_el = np.zeros((self.timescale, self.n_reaches + 1))
        self.node_el[0,:] = np.append(self.reach_data.el_fn, self.reach_data.el_tn[self.outlet])
        self.node_el[1,:] = np.append(self.reach_data.el_fn, self.reach_data.el_tn[self.outlet])
        self.node_el[:,-1] = self.node_el[1,-1]
        
        # In case of constant slope
        if self.update_slope == False:
            self.node_el[:,: ] = self.node_el[0,:]
        
       
    def initialize_storing_matrices(self): 
    
        # Create Qbi dep matrix with size size depending on how often we want to save it:
        if self.save_dep_layer=='never':
            dep_save_number = 1
        if self.save_dep_layer=='yearly':
            dep_save_number = int(self.timescale / 365) + 1  # +1 because we also keep t0.
        if self.save_dep_layer=='always':
            dep_save_number = self.timescale
        self.Qbi_dep = [[np.expand_dims(np.zeros(self.n_classes + 1, dtype=np.float32), axis = 0) for _ in range(self.n_reaches)] for _ in range(dep_save_number)]
        
        self.Qbi_tr = [np.zeros((self.n_reaches, self.n_reaches, self.n_classes), dtype=np.float32) for _ in range(self.timescale)] # sediment within the reach AFTER transfer, which also gives the provenance 
        self.Qbi_mob = [np.zeros((self.n_reaches, self.n_reaches, self.n_classes), dtype=np.float32) for _ in range(self.timescale)] # sediment within the reach BEFORE transfer, which also gives the provenance
        # Note Qbi_tr and Qbi_mob are 3D matrices, if we add the time as a 4th dimension, we can not look at the matrix in spyder. 
               
        self.Qbi_dep_0 = [np.expand_dims(np.zeros(self.n_classes + 1, dtype=np.float32), axis = 0) for _ in range(self.n_reaches)] # Initialise sediment deposit in the reaches  
       
        # 3D arrays
        self.Q_out = self.create_3d_zero_array()  # amount of material delivered outside the network in each timestep
        self.V_sed = self.create_3d_zero_array()  # velocities
        
        self.tr_cap = self.create_3d_zero_array()  # transport capacity per each sediment class
        self.tr_cap_before_tlag = self.create_3d_zero_array()
        
        self.Fi_al = self.create_3d_zero_array() # contains grain size distribution of the active layer
        self.Fi_al[:,0] = np.nan #DD: why ?        
        self.Fi_al_before_tlag = self.create_3d_zero_array()
        self.Fi_al_before_tlag[:,0] = np.nan #DD: why ?
        self.Qc_class_all = self.create_3d_zero_array()
        # self.Delta_V_class_all = self.create_3d_zero_array()
        
        # 2D arrays
        self.D50_al = self.create_2d_zero_array()  # D50 of the active layer in each reach in each timestep
        self.D50_al_before_tlag = self.create_2d_zero_array()
        self.tr_cap_sum = self.create_2d_zero_array()  # total transport capacity 
        self.flow_depth = self.create_2d_zero_array()
        # self.Delta_V_all = self.create_2d_zero_array()  # reach mass balance (volumes eroded or deposited)
 
    def set_sediment_initial_deposit(self, Qbi_dep_in):
        #TODO: (DD) better way to store Qbi_dep, Qbi_dep_0 etc ? 
        for n in self.network['n_hier']:  
            # if no inputs are defined, initialize deposit layer with a single cascade with no volume and GSD equal to 0
            q_bin = np.array(Qbi_dep_in[n])
            if not q_bin.any(): #if all zeros 
                self.Qbi_dep_0[n] = np.hstack((n, np.zeros(self.n_classes))).reshape(1,-1)
            else:           
                self.Qbi_dep_0[n] = np.float32(np.hstack((np.ones(q_bin.shape[0]) * n, Qbi_dep_in[n, 0]))).reshape(1,-1)
                self.Fi_al[0,n,:] = np.sum(q_bin, axis=0) / np.sum(q_bin)
                self.D50_al[0,n] = D_finder(self.Fi_al[0,n,:], 50, self.psi)
    
        self.Qbi_dep[0] = copy.deepcopy(self.Qbi_dep_0)  # store init condition of dep layer
        
    def set_erosion_maximum(self, eros_max_depth_, roundpar):
        # Set maximum volume in meters that can be eroded for each reach, for each time step.
        self.eros_max_depth = np.ones((1, self.n_reaches)) * eros_max_depth_
        self.eros_max_vol = np.round(self.eros_max_depth * self.reach_data.wac * self.reach_data.length, roundpar)
        
    def set_active_layer(self):       
        # Set active layer volume, i.e. the one used for calculating the tranport 
        # capacity in [m3/s]. Corresponds to the depth that the river can see 
        # every second (more like a continuum carpet ...) defined here as 2.D90 
        # [Parker 2008].
        
        self.al_vol = self.create_2d_zero_array()  
        self.al_depth = self.create_2d_zero_array()  
        
        # We take the input D90, or if not provided, the D84:
        if ~np.isnan(self.reach_data.D90):
            reference_d = self.reach_data.D90
        else:
            reference_d = self.reach_data.D84
        for n in self.network['n_hier']:
            al_depth_t = np.maximum(2 * reference_d[n], 0.01)
            al_vol_t = al_depth_t * self.reach_data.wac[n] * self.reach_data.length[n]
            self.al_vol[:,n] = np.repeat(al_vol_t, self.timescale, axis=0)
            self.al_depth[:,n] = np.repeat(al_depth_t, self.timescale, axis=0)        
        
    
    def compute_cascades_velocities(self, cascades_list, Vdep, 
                                    Q_reach, v, h, roundpar, t, n, 
                                    indx_velocity, indx_vel_partition,
                                    indx_tr_cap, indx_tr_partition):
        
        # Compute the velocity of the cascades in reach_cascade_list.
        # The velocity must be assessed by re-calculating the transport capacity 
        # in the present reach, considering the effect of the arriving cascade(s).
        # Two methods are proposed to re-evaluated the transport capacity, chosen 
        # by the indx_velocity. 
        # First method: the simplest, we re-calculate the transport capacity on each cascade itself.
        # Second method: we consider the active layer volume, to complete, if needed, 
        # the list of cascade by some reach material. If the cascade volume is more 
        # than the active layer, we consider all the cascade volume.
        

        if indx_velocity == 1:
            velocities_list = []
            for cascade in cascades_list:
                cascade.velocities = self.volume_velocities(cascade.volume,
                                                       Q_reach, v, h, t, n,     
                                                       indx_vel_partition, 
                                                       indx_tr_cap, indx_tr_partition)
                
                velocities_list.append(cascade.velocities)
            # In this case, we store the averaged velocities obtained among all the cascades
            velocities = np.mean(np.array(velocities_list), axis = 0)
                
        if indx_velocity == 2:   
            # concatenate cascades in one volume, and compact it by original provenance
            # DD: should the cascade volume be in [m3/s] ?
            volume_all_cascades = np.concatenate([cascade.volume for cascade in cascades_list], axis=0) 
            volume_all_cascades = self.matrix_compact(volume_all_cascades)
            
            volume_total = np.sum(volume_all_cascades[:,1:])            
            if volume_total < self.al_vol[t, n]:
                _, Vdep_active, _, _ = self.layer_search(Vdep, self.al_vol[t, n],
                                        roundpar, Qpass_volume = volume_all_cascades)
                volume_all_cascades = np.concatenate([volume_all_cascades, Vdep_active], axis=0) 

            velocities = self.volume_velocities(volume_all_cascades, 
                                                Q_reach, v, h, t, n,
                                                indx_vel_partition,  
                                                indx_tr_cap, indx_tr_partition)
            
            for cascade in cascades_list:
                cascade.velocities = velocities
                
        # Store velocities
        self.V_sed[t, n, :] = velocities * self.ts_length


    def volume_velocities(self, volume, Q_reach, v, h, t, n, 
                          indx_vel_partition,
                          indx_tr_cap, indx_tr_partition):
        
        ''' Compute the velocity of the volume of sediments. The transport capacity [m3/s]
        is calculated on this volume, and the velocity is calculated by dividing the 
        transport capacity by a section (hVel x width x (1 - porosity)). 
        For partionning the section among the different sediment class in the volume, 
        two methods are proposed. 
        The first one put the same velocity to all classes.
        The second divides the section equally among the classes with non-zero transport 
        capacity, so the velocity stays proportional to the transport capacity of that class.
        
        '''
        # Find volume sediment class fractions and D50
        volume_total = np.sum(volume[:,1:])
        volume_total_per_class = np.sum(volume[:,1:], axis = 0)
        sed_class_fraction = volume_total_per_class / volume_total
        D50 = float(D_finder(sed_class_fraction, 50, self.psi))
        
        # Compute the transport capacity
        calculator = TransportCapacityCalculator(sed_class_fraction, D50, 
                                                 self.slope[t, n], Q_reach, 
                                                 self.reach_data.wac[n],
                                                 v, h, self.psi)
        [ tr_cap_per_s, pci ] = calculator.tr_cap_function(indx_tr_cap, indx_tr_partition)
        
        
        Svel = self.vl_height[t, n] * self.reach_data.wac[n] * (1 - self.phi)  # the global section where all sediments pass through
        if Svel == 0 or numpy.isnan(Svel) == True:
            raise ValueError("The section to compute velocities can not be 0 or NaN.")
    
        if indx_vel_partition == 1:
            velocity_same = np.sum(tr_cap_per_s) / Svel     # same velocity for each class
            velocity_same = np.maximum(velocity_same, self.minvel)    # apply the min vel threshold
            velocities = np.full(len(tr_cap_per_s), velocity_same) # put the same value for all classes
            
        elif indx_vel_partition == 2:
            # Get the number of classes that are non 0 in the transport capacity flux:
            number_with_flux = np.count_nonzero(tr_cap_per_s)
            if number_with_flux != 0:
                Si = Svel / number_with_flux             # same section for all sediments
                velocities = np.maximum(tr_cap_per_s/Si, self.minvel)
            else:
                velocities = np.zeros(len(tr_cap_per_s)) # if transport capacity is all 0, velocity is all 0
        return velocities        
    
    
    
    def cascades_end_time_or_not(self, cascade_list, n):
        ''' Fonction to decide if the traveling cascades in cascade list stop in 
        the reach or not, due to the end of the time step.
        Inputs:
            cascade_list:       list of traveling cascades
            n:                  reach index
            
        Return:
            cascade_list_new:       same cascade list updated. Stopping cascades or 
                                    partial volumes have been removed
                                                              
            depositing_volume:      the volume to be deposited in this reach. 
                                    They are ordered according to their arrival time
                                    at the inlet, so that volume arriving first 
                                    deposit first.
        '''   
        # Order cascades according to their arrival time, so that first arriving 
        # cascade are first in the loop and are deposited first 
        # Note: in the deposit layer matrix, first rows are the bottom layers
        cascade_list = sorted(cascade_list, key=lambda x: np.mean(x.elapsed_time))
        
        depositing_volume_list = []
        cascades_to_be_completely_removed = []
        
        for cascade in cascade_list:
            # Time in, time travel, and time out in time step unit (not seconds)
            t_in = cascade.elapsed_time
            t_travel_n = self.reach_data.length[n] / (cascade.velocities * self.ts_length)
            t_out = t_in + t_travel_n
            # Vm_stop is the stopping part of the cascade volume
            # Vm_continue is the continuing part
            Vm_stop, Vm_continue = self.stop_or_not(t_out, cascade.volume)
            
            if Vm_stop is not None:
                depositing_volume_list.append(Vm_stop)
                
                if Vm_continue is None: 
                    # no part of the volume continues, we remove the entire cascade
                    cascades_to_be_completely_removed.append(cascade)
                else: 
                    # some part of the volume continues, we update the volume 
                    cascade.volume = Vm_continue
                                    
            if Vm_continue is not None:
                # update time for continuing cascades
                cascade.elapsed_time = t_out 
                # put to 0 the elapsed time of the empty sediment classes
                # i.e. the classes that have deposited, while other did not
                # (Necessary for the time lag calculation later in the code)
                cond_0 = np.all(cascade.volume[:,1:] == 0, axis = 0)
                cascade.elapsed_time[cond_0] = 0
                
        
        # If they are, remove complete cascades:
        cascade_list_new = [casc for casc in cascade_list if casc not in cascades_to_be_completely_removed]   
        
        # If they are, concatenate the deposited volumes 
        if depositing_volume_list != []:
            depositing_volume = np.concatenate(depositing_volume_list, axis=0)
            if np.all(depositing_volume[:,1:] == 0):
                raise ValueError("DD check: we have an empty layer stopping ?")
        else:
            depositing_volume = None
        
        return cascade_list_new, depositing_volume #DD: to be stored somewhere
    
    
    def stop_or_not(self, t_new, Vm):
        ''' 
        Function that decides if a volume of sediments will stop in this 
        reach or not, depending on time. Part of the volume can stop or continue. 
        
        t_new: elapsed time since beginning of time step for Vm, for each sed class
        Vm: traveling volume of sediments
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
    

    def compute_time_lag(self, cascade_list):#, passing_cascade_in_trcap, time_lag_for_mobilised):
        
        # The time lag is the time we use to mobilise from the reach, 
        # before cascades from upstream reaches arrive at the outlet of the present reach.
        # We take it as the time for the first cascade to arrive at the outet. 
        
        # cascade_list            : the list of passing cascade objects. Can be empty.

        if cascade_list == []:
            time_lag = np.ones(self.n_classes) # the time lag is the entire time step as no other cascade reach the outlet
        else:
            time_arrays = np.array([cascade.elapsed_time for cascade in cascade_list])
            time_lag = np.min(time_arrays, axis=0)                   
        
        return time_lag     
    
    
    def compute_transport_capacity(self, Vdep, roundpar, t, n, Q, v, h,
                                   indx_tr_cap, indx_tr_partition,
                                   passing_cascades = None, per_second = True):
        # Compute the transport capacity in m3/s using the active layer (2D90)
        # on the deposit layer (Vdep) and eventual passing cascades.
        # The option "per second" put the passing cascades in m3/s instead of m3/ts_length
        
        if passing_cascades == None or passing_cascades == []:
            passing_volume = None
        else:
            # Makes a single volume out of the pssing cascade list:
            passing_volume = np.concatenate([cascade.volume for cascade in passing_cascades], axis=0)
            passing_volume = self.matrix_compact(passing_volume) #compact by original provenance
            
            if per_second == True:
                passing_volume = copy.deepcopy(passing_volume)
                passing_volume[:,1:] = passing_volume[:,1:] / self.ts_length
            
                
        # Compute fraction and D50 in the active layer
        _,_,_, Fi_al_ = self.layer_search(Vdep, self.al_vol[t,n], roundpar, Qpass_volume = passing_volume)                   
        # In case the active layer is empty, I use the GSD of the previous timestep
        if np.sum(Fi_al_) == 0:
           Fi_al_ = self.Fi_al[t-1,n,:] 
        D50_al_ = float(D_finder(Fi_al_, 50, self.psi))
           
        # Transport capacity in m3/s
        calculator = TransportCapacityCalculator(Fi_al_ , D50_al_, self.slope[t,n], 
                                               Q[t,n], self.reach_data.wac[n], v[n],
                                               h[n], self.psi)
        tr_cap_per_s, Qc = calculator.tr_cap_function(indx_tr_cap, indx_tr_partition)
        
        return tr_cap_per_s, Fi_al_, D50_al_, Qc
    
    
    def compute_mobilised_volume(self, Vdep, tr_cap_per_s, n, roundpar,
                                 passing_cascades = None, time_fraction = None):
        
        # Case where we don't consider a time lag, the time for mobilising is the complete time step:
        if time_fraction is None: 
            time_fraction = np.ones(self.n_classes)
            
        # Real time to mobilise: 
        time_to_mobilise = time_fraction * self.ts_length
        # Mobilisable volume:
        volume_mobilisable = tr_cap_per_s * time_to_mobilise        
        # Erosion maximum during the time lag 
        # (we take the mean time lag among the classes)
        e_max_vol_ = self.eros_max_vol[0,n] * np.mean(time_fraction)       
        # Eventual total volume arriving
        if passing_cascades == None or passing_cascades == []:
            sum_pass = 0
        else:
            passing_volume = np.concatenate([cascade.volume for cascade in passing_cascades], axis=0)
            sum_pass = np.sum(passing_volume[:,1:], axis=0)
            
                      
        # Compare sum of passing cascade to the mobilisable volume (for each sediment class)               
        diff_with_capacity = volume_mobilisable - sum_pass

        # Sediment classes with positive values in diff_with_capacity are mobilised from the reach n
        diff_pos = np.where(diff_with_capacity < 0, 0, diff_with_capacity)     
        if np.any(diff_pos): 
            # Search for layers to be put in the erosion max (e_max_vol_)
            V_inc_el, V_dep_el, V_dep_not_el, _ = self.layer_search(Vdep, e_max_vol_, roundpar)
            [V_mob, Vdep_new] = tr_cap_deposit(V_inc_el, V_dep_el, V_dep_not_el, diff_pos, roundpar)
            if np.all(V_mob[:,1:] == 0):
                V_mob = None
        else:
            Vdep_new  = Vdep
            V_mob = None
                          
        # Sediment classes with negative values in diff_with_capacity
        # are over capacity
        # They are deposited, i.e. directly added to Vdep
        diff_neg = -np.where(diff_with_capacity > 0, 0, diff_with_capacity)     
        if np.any(diff_neg):  
            Vm_removed, passing_cascades, residual = deposit_from_passing_sediments(np.copy(diff_neg), passing_cascades, roundpar)
            # Deposit the Vm_removed:
            Vdep_new = np.concatenate([Vdep_new, Vm_removed], axis=0)
            
        return V_mob, passing_cascades, Vdep_new
    
        
    
    def layer_search(self, V_dep_old, V_lim, roundpar, Qpass_volume = None):
        # This function searches uppermost layers from a volume of layers, 
        # to correspond to a maximum volume. Passing cascade can be integrated
        # to the top of the volume.
        # The maximum volume can represent for example the active layer, 
        # i.e. what we consider as active during the transport process,
        # or a maximum to be eroded per time step.
    
        # INPUTS:    
        # V_dep_old :             the reach deposit layer
        # V_lim  :            is the total maximum volume to be mobilised
        # Qpass_volume :      is the traveling volume to be added in the layers
        
        # RETURN:
        # V_inc2act    :      Layers of the incoming volume to be put in the active layer
        # V_dep2act    :      layers of the deposit volume to be put in the active layer
        # V_dep_new        :      remaining deposit layer
        # Fi_r_reach   :      fraction of sediment in the active layer
        
        if Qpass_volume is None:
            # Put an empty layer (for computation)
            empty_incoming_volume = np.hstack((0, np.zeros(self.n_classes))) 
            empty_incoming_volume = np.expand_dims(empty_incoming_volume, axis = 0) 
            Qpass_volume = empty_incoming_volume

        # If, considering the incoming volume, I am still under the threshold of the active layer volume,
        # I put sediment from the deposit layer into the active layer.
        if (V_lim - np.sum(Qpass_volume[:, 1:])) > 0:
                        
            V_inc2act = Qpass_volume # All the passing volume is in the active layer

            # Find what needs to be added from V_dep_old in the active layer :            
            V_lim_dep = V_lim - np.sum(Qpass_volume[:, 1:]) # Remaining active layer volume after considering incoming sediment cascades
            
            csum = np.flipud(np.cumsum(np.flipud(np.sum(V_dep_old[:, 1:], axis=1)), axis = 0))  # Cumulative sum of V_dep_old
            
            # If the volume available in V_dep_old is less than V_lim_dep:
            #  (i've reached the bottom)
            # and I put all the deposit into the active layer           
            if (np.argwhere(csum > V_lim_dep)).size == 0 :  # the vector is empty 
                print(' reach the bottom ....')

                V_dep2act = V_dep_old  
                V_dep = np.c_[V_dep_old[0,0], np.zeros((1, Qpass_volume.shape[1]-1))]
            
            # Otherwise I must select the uppermost layers from V_dep_old to be 
            # put in the active layer
            else:               
                index = np.max(np.argwhere(csum >= V_lim_dep)) # index (x nclasses) to shows the lowermost layer to be put in the active layer

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
            perc_dep = V_lim / np.sum(Qpass_volume[:, 1:])

            if ~np.isnan(roundpar):
                Qpass_dep = np.around(Qpass_volume[:, 1:]*(1-perc_dep), decimals=roundpar)
            else:
                # this contains the fraction of the incoming volume to be deposited
                Qpass_dep = Qpass_volume[:, 1:]*(1-perc_dep)

            V_inc2act = np.hstack((Qpass_volume[:, 0][:,None], Qpass_volume[:, 1:] - Qpass_dep))
            V_dep2act = np.append(V_dep_old[0, 0], np.zeros((1, Qpass_volume.shape[1]-1)))
            
            if V_dep2act.ndim == 1: 
                V_dep2act = V_dep2act[None, :]

            # if, given the round, the deposited volume of the incoming cascades is not 0...
            if any(np.sum(Qpass_volume[:, 1:]*(1-perc_dep), axis = 0)):
                V_dep = np.vstack((V_dep_old, np.hstack((Qpass_volume[:, 0][:,None], Qpass_dep))))
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
    
    
    def matrix_compact(self, V_layer):
        # Function that groups layers (rows) in V_layers
        # according to the original provenance (first column)
        
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
    
    
    
