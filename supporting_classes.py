"""
Created on Tue Oct 29 10:58:54 2024

@author: diane
"""
import copy
import os
import sys
from itertools import groupby

# General imports
import numpy as np
import numpy.matlib
import pandas as pd
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')

from constants import GRAV, R_VAR, RHO_S, RHO_W
from supporting_functions import D_finder
from transport_capacity_computation import TransportCapacityCalculator


class Cascade:
    def __init__(self, provenance, elapsed_time, volume):
        """
        Initialyse a cascade.

        @param provenance (...)
            Provenance
        @param elapsed_time (...)
            Elapsed time
        @param volume (...)
            Volume
        """
        self.provenance = provenance
        self.elapsed_time = elapsed_time # can contain nans, in case a class has 0 volume
        self.volume = volume # size = n_classes + 1, to include the original provenance in a first column
        # To be filled during the time step
        self.velocities = np.nan # in m/s
        # Flag to know if the cascade is from external source (default False)
        self.is_external = False



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

    def compute_roughness(self):
        # to test and see if it is what we want in terms of physics
        if 'roughness' in self.geodf:
            roughness = self.geodf['roughness'].astype(float).values
        elif 'D90' in self.geodf:
            roughness = self.geodf['D90'].astype(float).values
        else:
            roughness = self.geodf['D84'].astype(float).values
        return roughness




class SedimentarySystem:
    ''' Class for managing sediment exchanges, reassembling, and storing
    '''
    #TODO: (DD) tests must be made on eros max, al, and velocity_height
    # to make sure that they don't contain Nans


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
        
        self.n_metadata = 1

        # Setted variables


        # Storing matrices
        self.Qbi_dep = None
        self.external_inputs = None
        self.force_pass_external_inputs = None
        self.Qbi_tr = None
        self.Qbi_mob = None
        self.Qbi_mob_from_r = None
        self.V_sed = None
        self.Q_out = None
        self.slope = None
        self.width = None
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
        self.mass_balance = self.create_3d_zero_array()


        # temporary ?
        self.Qbi_dep_0 = None

        self.Qc_class_all = None        # DD: can it be optional ?


    def sediments(self, matrix):
        '''
        Access the sediment columns of the matrix.
        @warning: Use self.sediments(matrix) for access, but use self.sediments(matrix)[:] for assignment!!!
        '''
        if matrix.ndim == 1:
            return matrix[self.n_metadata:]
        elif matrix.ndim == 2:
            return matrix[:, self.n_metadata:]


    def provenance(self, matrix):
        '''
        Access the provenance column of the matrix.
        @warning: Use self.provenance(matrix) for access, but use self.provenance(matrix)[:] for assignment!!!
        '''
        if matrix.ndim == 1:
            return matrix[0]
        elif matrix.ndim == 2:
            return matrix[:, 0]

    def create_4d_zero_array(self):
        ''' This type of matrice is made for including provenance (axis 0)
        Note: we add the time as a list, otherwise we can not look at the 4d matrix in spyder.
        '''
        return [np.zeros((self.n_reaches, self.n_reaches, self.n_classes)) for _ in range(self.timescale)]

    def create_3d_zero_array(self):
        return np.zeros((self.timescale, self.n_reaches, self.n_classes))

    def create_2d_zero_array(self):
        return np.zeros((self.timescale, self.n_reaches))

    def initialize_slopes(self):
        self.min_slope = min(self.reach_data.slope)  # put a minimum value to guarantee movement  (DD: why min of the reach slopes ?)
        self.slope = self.create_2d_zero_array()
        self.slope[0,:] = np.maximum(self.reach_data.slope, self.min_slope)
        self.slope[1,:] = np.maximum(self.reach_data.slope, self.min_slope)

        # In case of constant slope
        if self.update_slope == False:
            self.slope[:,:] = self.slope[0,:]

    def initialize_widths(self, indx_width_calc):
        self.width = self.create_2d_zero_array()

        if indx_width_calc == 1:
            # Static width
            self.width[0,:] = self.reach_data.wac

        if indx_width_calc == 2:
            # Varying width, if a bankfull width is specified, we take this one
            # Otherwise, we vary the width based on the Wac column in reachdata
            # It will be reduced later within the time loop
            if self.reach_data.wac_bf is not None:
                self.width[0,:] = self.reach_data.wac_bf
            else:
                self.width[0,:] = self.reach_data.wac

        # Put the same initial width for all time steps
        self.width[:,:] = self.width[0,:]

    def initialize_elevations(self):
        # Initialize node elevation (for each reach the matrix reports the fromN elevation)
        # The last column reports the outlet ToNode elevation (last node of the network),
        # which can never change elevation.
        self.node_el = np.zeros((self.timescale, self.n_reaches + 1))
        self.node_el[0,:] = np.append(self.reach_data.el_fn, self.reach_data.el_tn[self.outlet])
        self.node_el[1,:] = np.append(self.reach_data.el_fn, self.reach_data.el_tn[self.outlet])
        # Fix last node elevation for all time steps:
        self.node_el[:,-1] = self.node_el[1,-1]

        # In case of constant slope:
        if self.update_slope == False:
            self.node_el[:,: ] = self.node_el[0,:]


    def initialize_storing_matrices(self, t_track):

        # Create Qbi dep matrix with size size depending on how often we want to save it:
        if self.save_dep_layer=='never':
            dep_save_number = 1
        if self.save_dep_layer=='yearly':
            dep_save_number = int(self.timescale / 365) + 1  # +1 because we also keep t0.
        if self.save_dep_layer=='always':
            dep_save_number = self.timescale
        self.Qbi_dep = [[np.expand_dims(np.zeros(self.n_metadata + self.n_classes), axis = 0) for _ in range(self.n_reaches)] for _ in range(dep_save_number)]
        
        # For t_track
        if t_track is not None:
            # n_time_track = t_track[1]-t_track[0]+1
            self.Qbi_dep_track = [[np.expand_dims(np.zeros(self.n_metadata + self.n_classes), axis = 0) for _ in range(self.n_reaches)] for _ in range(self.timescale)]
            self.Qbi_dep_track2 = self.create_4d_zero_array()
        
        # Initial Qbi_dep:
        self.Qbi_dep_0 = [np.expand_dims(np.zeros(self.n_metadata + self.n_classes), axis = 0) for _ in range(self.n_reaches)] # Initialise sediment deposit in the reaches

        # Moving sediments storing matrice
        self.Qbi_mob = self.create_4d_zero_array() # Volume leaving the reach (gives also original provenance)
        self.Qbi_mob_from_r = self.create_4d_zero_array() # Volume mobilised from reach (gives also original provenance)
        # TODO: DD see if we keep Qbi_tr
        self.Qbi_tr = self.create_4d_zero_array() # Volume entering the reach (gives also original provenance)
        # Direct connectivity matrice (an extra reach column is added to consider sediment leaving the system)
        self.direct_connectivity = [np.zeros((self.n_reaches, self.n_reaches + 1, self.n_classes)) for _ in range(self.timescale)]


        # 3D arrays
        self.Q_out = self.create_3d_zero_array()  # amount of material delivered outside the network in each timestep
        self.V_sed = self.create_3d_zero_array()  # velocities
        self.sediment_budget = self.create_3d_zero_array()

        self.tr_cap = self.create_3d_zero_array()  # transport capacity per each sediment class
        self.tr_cap_before_tlag = self.create_3d_zero_array()

        self.Fi_al = self.create_3d_zero_array() # contains grain size distribution of the active layer
        self.Fi_al[:,0] = np.nan #DD: why ?
        self.Fi_al_before_tlag = self.create_3d_zero_array()
        self.Fi_al_before_tlag[:,0] = np.nan #DD: why ?
        self.Qc_class_all = self.create_3d_zero_array()

        # 2D arrays
        self.D50_al = self.create_2d_zero_array()  # D50 of the active layer in each reach in each timestep
        self.D50_al_before_tlag = self.create_2d_zero_array()
        self.tr_cap_sum = self.create_2d_zero_array()  # total transport capacity
        self.flow_depth = self.create_2d_zero_array()



    def set_sediment_initial_deposit(self, Qbi_dep_in):
        #TODO: (DD) better way to store Qbi_dep, Qbi_dep_0 etc ?
        for n in self.network['n_hier']:
            # if no inputs are defined, initialize deposit layer with a single cascade with no volume and GSD equal to 0
            q_bin = np.array(Qbi_dep_in[n])
            if not q_bin.any(): #if all zeros
                self.Qbi_dep_0[n] = np.hstack((n, np.zeros(self.n_classes))).reshape(1,-1)
            else:
                self.Qbi_dep_0[n] = np.float64(np.hstack((np.ones(q_bin.shape[0]) * n, Qbi_dep_in[n, 0]))).reshape(1,-1)
                self.Fi_al[0,n,:] = np.sum(q_bin, axis=0) / np.sum(q_bin)
                self.D50_al[0,n] = D_finder(self.Fi_al[0,n,:], 50, self.psi)

        self.Qbi_dep[0] = copy.deepcopy(self.Qbi_dep_0)  # store init condition of dep layer

    def set_erosion_maximum(self, eros_max_depth_, roundpar):
        # Set maximum volume in meters that can be eroded for each reach, for each time step.
        self.eros_max_depth = np.ones(self.n_reaches) * eros_max_depth_
        self.eros_max_vol = np.round(self.eros_max_depth * self.reach_data.wac * self.reach_data.length, roundpar)

    def set_active_layer(self, al_depth):
        # Set active layer volume, i.e. the one used for calculating the tranport
        # capacity in [m3/s]. Corresponds to the depth that the river can see
        # every second (more like a continuum carpet ...)
        # By default it is defined as 2.D90 [Parker 2008]
        # But this is maybe not adapted for sandy rivers.

        self.al_vol = self.create_2d_zero_array()
        self.al_depth = self.create_2d_zero_array()

        if al_depth == '2D90':
            # We take the input D90, or if not provided, the D84:
            if self.reach_data.D90 is not None:
                reference_d = self.reach_data.D90
            else:
                reference_d = self.reach_data.D84
            # Multiply by two, + apply a minimum threshold
            al_depth_t = np.maximum(2 * reference_d, 0.01)
        elif isinstance(al_depth, (int, float)):
            # Apply the input AL depth
            al_depth_t = al_depth * np.ones(self.n_reaches)
        else:
            raise ValueError('As options for the AL depth, you can choose "2D90" or a fixed number')

        # Compute the AL volumes (all reaches)
        al_vol_t = al_depth_t * self.reach_data.wac * self.reach_data.length
        # Store it for all time steps:
        self.al_vol = np.tile(al_vol_t, (self.timescale, 1))
        self.al_depth = np.tile(al_depth_t, (self.timescale, 1))

    def set_velocity_section_height(self, vel_section_height, h, t):
        '''Set section height, for estimating velocity [m/s] from sediment flux [m3/s]
        Possibilities:  '2D90': twice the input D90.
                        '0.1_hw': 10% of the water column.
                        Or a fixed value.
        '''
        if vel_section_height == '2D90':
            # We take the input D90, or if not provided, the D84:
            if self.reach_data.D90 is not None:
                reference_d = self.reach_data.D90
            else:
                reference_d = self.reach_data.D84
            # Multiply by two, + apply a minimum threshold of 1 cm
            vl_height_t = np.maximum(2 * reference_d, 0.01)

        elif vel_section_height == '0.1_hw':
            vl_height_t = 0.1 * h
        elif isinstance(vel_section_height, (int, float)):
            vl_height_t = vel_section_height * np.ones(self.n_reaches)
        else:
            raise ValueError('Velocity height options are 2D90, 0.1_hw, or a number')

        # Store:
        self.vl_height[t, :] = vl_height_t

    def set_external_input(self, external_inputs, force_pass_external_inputs, roundpar):
        # define Qbi_input in this sed_system
        self.external_inputs = external_inputs
        self.force_pass_external_inputs = force_pass_external_inputs


    def extract_external_inputs(self, cascade_list, t, n):
        # Create a new cascade in reach n at time step t, to be added to the cascade list
        if numpy.any(self.external_inputs[t, n, :] > 0):
            provenance = n
            elapsed_time = np.zeros(self.n_classes)
            volume = np.expand_dims(np.append(n, self.external_inputs[t, n, :]), axis = 0)
            ext_cascade = Cascade(provenance, elapsed_time, volume)
            # We specify that the cascade is external:
            ext_cascade.is_external = True
            cascade_list.append(ext_cascade)

        return cascade_list


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

            volume_total = np.sum(self.sediments(volume_all_cascades))
            if volume_total < self.al_vol[t, n]:
                _, Vdep_active, _, _ = self.layer_search(Vdep, self.al_vol[t, n],
                                        Qpass_volume = volume_all_cascades, roundpar = roundpar)
                volume_all_cascades = np.concatenate([volume_all_cascades, Vdep_active], axis=0)

            velocities = self.volume_velocities(volume_all_cascades,
                                                Q_reach, v, h, t, n,
                                                indx_vel_partition,
                                                indx_tr_cap, indx_tr_partition)

            for cascade in cascades_list:
                cascade.velocities = velocities

        # Store velocities in m/s
        self.V_sed[t, n, :] = velocities


    def volume_velocities(self, volume, Q_reach, v, h, t, n,
                          indx_vel_partition,
                          indx_tr_cap, indx_tr_partition):

        ''' Compute the velocity of the volume of sediments in m/s.
        The transport capacity [m3/s] is calculated on this volume,
        and the velocity is calculated by dividing the
        transport capacity by a section (hVel x width x (1 - porosity)).
        For partionning the section among the different sediment class in the volume,
        two methods are proposed.
        The first one put the same velocity to all classes.
        The second divides the section equally among the classes with non-zero transport
        capacity, so the velocity stays proportional to the transport capacity of that class.

        '''
        # Find volume sediment class fractions and D50
        volume_total = np.sum(self.sediments(volume))
        volume_total_per_class = np.sum(self.sediments(volume), axis = 0)
        sed_class_fraction = volume_total_per_class / volume_total
        D50 = float(D_finder(sed_class_fraction, 50, self.psi))

        # Compute the transport capacity
        calculator = TransportCapacityCalculator(sed_class_fraction, D50,
                                                 self.slope[t, n], Q_reach,
                                                 self.width[t, n],
                                                 v, h, self.psi, self.reach_data.roughness[n])

        [ tr_cap_per_s, pci ] = calculator.tr_cap_function(indx_tr_cap, indx_tr_partition)


        Svel = self.vl_height[t, n] * self.width[t, n] * (1 - self.phi)  # the global section where all sediments pass through
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



    def cascades_end_time_or_not(self, cascade_list, n, t):
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
        cascade_list = sorted(cascade_list, key=lambda x: np.nanmean(x.elapsed_time))

        depositing_volume_list = []
        cascades_to_be_completely_removed = []


        for cascade in cascade_list:
            # Time in, time travel, and time out in time step unit (not seconds)
            t_in = cascade.elapsed_time
            if cascade.is_external == True and cascade.provenance == n:
                # Particular case where external cascades are passed to the next reach and excluded of the calculation
                if self.force_pass_external_inputs == True:
                    continue
                else:
                    # External sources coming from current reach,
                    # are starting halfway of the reach.
                    distance_to_reach_outlet = 0 * self.reach_data.length[n]
            else:
                distance_to_reach_outlet = self.reach_data.length[n]
            t_travel_n = distance_to_reach_outlet / (cascade.velocities * self.ts_length)
            t_out = t_in + t_travel_n
            # Vm_stop is the stopping part of the cascade volume
            # Vm_continue is the continuing part
            Vm_stop, Vm_continue = self.stop_or_not(t_out, cascade.volume)

            if Vm_stop is not None:
                depositing_volume_list.append(Vm_stop)
                # Fill connectivity matrice:
                self.direct_connectivity[t][cascade.provenance, n, :] += np.sum(self.sediments(Vm_stop), axis = 0)

                if Vm_continue is None:
                    # no part of the volume continues, we remove the entire cascade
                    cascades_to_be_completely_removed.append(cascade)
                else:
                    # some part of the volume continues, we update the volume
                    cascade.volume = Vm_continue

            if Vm_continue is not None:
                # update time for continuing cascades
                cascade.elapsed_time = t_out
                # put to np.nan the elapsed time of the empty sediment classes
                # (Necessary for the time lag calculation later in the code)
                cond_0 = np.all(self.sediments(cascade.volume) == 0, axis = 0)
                cascade.elapsed_time[cond_0] = np.nan


        # If they are, remove complete cascades:
        cascade_list_new = [casc for casc in cascade_list if casc not in cascades_to_be_completely_removed]

        # If they are, concatenate the deposited volumes
        if depositing_volume_list != []:
            depositing_volume = np.concatenate(depositing_volume_list, axis=0)
            if np.all(self.sediments(depositing_volume) == 0):
                raise ValueError("DD check: we have an empty layer stopping ?")
        else:
            depositing_volume = None

        return cascade_list_new, depositing_volume


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

        if np.all(self.sediments(Vm_stop) == 0) == True:
            Vm_stop = None
        if np.all(self.sediments(Vm_continue) == 0) == True:
            Vm_continue = None

        return Vm_stop, Vm_continue


    def compute_time_lag(self, cascade_list):

        # The time lag is the time we use to mobilise from the reach,
        # before cascades from upstream reaches arrive at the outlet of the present reach.
        # We take it as the time for the first cascade to arrive at the outet.

        # cascade_list            : the list of passing cascade objects. Can be empty.

        if cascade_list == []:
            time_lag = np.ones(self.n_classes) # the time lag is the entire time step as no other cascade reach the outlet
        else:
            casc_mean_elaps_time = np.array([np.nanmean(cascade.elapsed_time) for cascade in cascade_list])
            if np.isnan(casc_mean_elaps_time).any() == True:
                raise ValueError("Strange case, one passing cascades has only nan in her elapsed times")
            time_lag = np.min(casc_mean_elaps_time)

        return time_lag


    def compute_transport_capacity(self, Vdep, roundpar, t, n, Q, v, h,
                                   indx_tr_cap, indx_tr_partition,
                                   passing_cascades = None, per_second = False):
        # Compute the transport capacity in m3/s using the active layer
        # on the deposit layer (Vdep) and passing cascades (if they are).

        # The option "per second" put the passing cascades in m3/s instead of m3/ts_length
        # This option is by default False. Putting it True can create some
        # strange behaviour (on and off mobilisation).

        if passing_cascades == None or passing_cascades == []:
            passing_volume = None

        else:
            # Particular case where external cascades are passed to the next reach and excluded of the calculation
            if self.force_pass_external_inputs == True:
                passing_cascades = [cascade for cascade in passing_cascades
                                      if not (cascade.is_external == True and cascade.provenance == n)]
            if passing_cascades == []:
                passing_volume = None
            else:
                # Makes a single volume out of the passing cascade list:
                passing_volume = np.concatenate([cascade.volume for cascade in passing_cascades], axis=0)
                passing_volume = self.matrix_compact(passing_volume) #compact by original provenance
                if per_second == True:
                    passing_volume = copy.deepcopy(passing_volume)
                    self.sediments(passing_volume)[:] = self.sediments(passing_volume) / self.ts_length

        # Compute fraction and D50 in the active layer
        # TODO: warning when the AL is very small, we will have Fi_r is 0 due to roundpar
        _,_,_, Fi_al_ = self.layer_search(Vdep, self.al_vol[t,n], Qpass_volume = passing_volume, roundpar = roundpar)
        # In case the active layer is empty, I use the GSD of the previous timestep
        if np.sum(Fi_al_) == 0:
           Fi_al_ = self.Fi_al[t-1, n, :]
        D50_al_ = float(D_finder(Fi_al_, 50, self.psi))

        # Transport capacity in m3/s
        calculator = TransportCapacityCalculator(Fi_al_ , D50_al_, self.slope[t,n],
                                               Q[t,n], self.width[t,n], v[n],
                                               h[n], self.psi, self.reach_data.roughness[n])

        tr_cap_per_s, Qc = calculator.tr_cap_function(indx_tr_cap, indx_tr_partition)

        return tr_cap_per_s, Fi_al_, D50_al_, Qc


    def compute_mobilised_volume(self, Vdep, tr_cap_per_s, n, t, roundpar,
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
        e_max_vol_ = self.eros_max_vol[n] * np.mean(time_fraction)

        # Eventual total volume arriving
        if passing_cascades == None or passing_cascades == []:
            sum_pass = 0
            passing_cascades_excluded = []
        else:
            # Particular case where external cascades are passed to the next reach and excluded of the calculation
            if self.force_pass_external_inputs == True:
                passing_cascades_excluded = [cascade for cascade in passing_cascades
                                      if (cascade.is_external == True and cascade.provenance == n)]
                passing_cascades = [cascade for cascade in passing_cascades if cascade not in passing_cascades_excluded]

            if  passing_cascades == []:
                sum_pass = 0
            else:
                passing_volume = np.concatenate([cascade.volume for cascade in passing_cascades], axis=0)
                sum_pass = np.sum(self.sediments(passing_volume), axis=0)


        # Compare sum of passing cascade to the mobilisable volume (for each sediment class)
        diff_with_capacity = volume_mobilisable - sum_pass

        # Sediment classes with positive values in diff_with_capacity are mobilised from the reach n
        diff_pos = np.where(diff_with_capacity < 0, 0, diff_with_capacity)
        if np.any(diff_pos):
            # Search for layers to be put in the erosion max (e_max_vol_)
            V_inc_el, V_dep_el, V_dep_not_el, _ = self.layer_search(Vdep, e_max_vol_, roundpar = roundpar)
            [V_mob, Vdep_new] = self.tr_cap_deposit(V_inc_el, V_dep_el, V_dep_not_el, diff_pos, roundpar)
            if np.all(self.sediments(V_mob) == 0):
                V_mob = None
        else:
            Vdep_new  = Vdep
            V_mob = None

        # Sediment classes with negative values in diff_with_capacity
        # are over capacity
        # They are deposited, i.e. directly added to Vdep
        diff_neg = -np.where(diff_with_capacity > 0, 0, diff_with_capacity)
        if np.any(diff_neg):
            Vm_removed, passing_cascades, residual = self.deposit_from_passing_sediments(np.copy(diff_neg), passing_cascades, roundpar, n, t)
            # Deposit the Vm_removed:
            Vdep_new = np.concatenate([Vdep_new, Vm_removed], axis=0)

        # Re-add the external cascades, that were excluded from calculation
        if self.force_pass_external_inputs == True and passing_cascades_excluded != []:
            passing_cascades.extend(passing_cascades_excluded)

        # If the new vdep is empty, put an empty layer for next steps
        if Vdep_new.size == 0:
            Vdep_new = np.hstack((n, np.zeros(self.n_classes)))
            Vdep_new = np.expand_dims(Vdep_new, axis = 0)

        return V_mob, passing_cascades, Vdep_new



    def layer_search(self, V_dep_old, V_lim, Qpass_volume = None, roundpar = None):
        # This function searches uppermost layers from a volume of layers,
        # to correspond to a maximum volume. Passing cascade can be integrated
        # to the top of the volume.
        # The maximum volume can represent for example the active layer,
        # i.e. what we consider as active during the transport process,
        # or a maximum to be eroded per time step.

        # INPUTS:
        # V_dep_old :         the reach deposit layer
        # V_lim  :            is the total maximum volume to be selected
        # Qpass_volume :      is the traveling volume to be added at the top of the layers
        # roundpar     :      number of decimals for rounding volumes

        # RETURN:
        # V_inc2act    :      Layers of the incoming volume to be put in the maximum volume
        # V_dep2act    :      layers of the deposit volume to be put in the maximum volume
        # V_dep_new    :      remaining deposit layer
        # Fi_r_reach   :      fraction of sediment in the maximum volume

        reach_provenance_idx = V_dep_old[0, 0]

        if Qpass_volume is None:
            # Put an empty layer (for computation)
            empty_incoming_volume = np.hstack((0, np.zeros(self.n_classes)))
            empty_incoming_volume = np.expand_dims(empty_incoming_volume, axis = 0)
            Qpass_volume = empty_incoming_volume

        # 1) If, considering the incoming volume, I am still under the threshold of the maximum volume,
        # I put sediment from the deposit layer into the maximum volume.
        if (V_lim - np.sum(self.sediments(Qpass_volume))) > 0:

            V_inc2act = Qpass_volume # All the passing volume is in the active layer (maximum volume)

            # Find what needs to be added from V_dep_old in the active layer :
            V_lim_dep = V_lim - np.sum(self.sediments(Qpass_volume)) # Remaining active layer volume after considering incoming sediment cascades

            # Sum classes in V_dep_old
            sum_class = np.sum(self.sediments(V_dep_old), axis=1)
            # Cumulate volumes from last to first row
            # Reminder: Last row is the top layer
            csum_Vdep = np.cumsum(sum_class[::-1])[::-1]

            # If the volume available in V_dep_old is less than V_lim_dep:
            #  (i've reached the bottom)
            # and I put all the deposit into the active layer
            if (np.argwhere(csum_Vdep > V_lim_dep)).size == 0 :  # the vector is empty
                print(' reach the bottom ....')
                V_dep2act = V_dep_old
                # Leave an empty layer in Vdep
                V_dep = np.c_[reach_provenance_idx, np.zeros((1, self.n_classes))]

            # Otherwise I must select the uppermost layers from V_dep_old to be
            # put in the active layer
            else:
                # Index (x nclasses) to shows the lowermost layer to be put in the active layer:
                index = np.max(np.argwhere(csum_Vdep >= V_lim_dep))
                # Layers above the targeted layer
                Vdep_above_index = V_dep_old[csum_Vdep < V_lim_dep, :]
                # Layers under the targeted layer
                Vdep_under_index = V_dep_old[0:index, :]
                # Targeted layer
                threshold_layer = V_dep_old[index, :]

                # (DD ?) if i have multiple deposit layers, put the upper layers into the active layer until i reach the threshold.

                # The layer on the threshold (defined by index) gets divided according to perc_layer:
                sum_above_threshold_layer = np.sum(self.sediments(Vdep_above_index))
                threshold_layer_residual = V_lim_dep - sum_above_threshold_layer
                threshold_layer_sum = sum(self.sediments(threshold_layer))
                perc_layer = threshold_layer_residual/threshold_layer_sum

                # remove small negative values that can arise from the difference being very close to 0
                perc_layer = np.maximum(0, perc_layer)

                # Multiply the threshold layer by perc_layer
                if roundpar is not None: # round
                    threshold_layer_included = np.around(self.sediments(threshold_layer) * perc_layer, decimals = roundpar)
                    threshold_layer_excluded = self.sediments(threshold_layer) - threshold_layer_included
                else:
                    threshold_layer_included = self.sediments(threshold_layer) * perc_layer
                    threshold_layer_excluded = self.sediments(threshold_layer) - threshold_layer_included

                # Re-add the provenance column:
                threshold_layer_included = np.hstack((self.provenance(threshold_layer), threshold_layer_included)).reshape(1, -1)
                threshold_layer_excluded = np.hstack((self.provenance(threshold_layer), threshold_layer_excluded)).reshape(1, -1)
                # Stack vertically the threshold layer included to the above layers (V_dep2act):
                V_dep2act = np.vstack((threshold_layer_included, Vdep_above_index))
                # Stack vertically the threshold layer excluded to the below layers (V_dep):
                V_dep = np.vstack((Vdep_under_index, threshold_layer_excluded))


        # 2) If the incoming sediment volume is enough to completely fill the active layer,
        # deposit part of the incoming cascades and put the rest into the active layer
        # NB: we deposit a same percentage of all cascades
        else:
            # Percentage of the volume to put in the active layer for all the cascades
            perc_dep = V_lim / np.sum(self.sediments(Qpass_volume))

            # Volume from the incoming volume to be deposited:
            if roundpar is not None:
                Qpass_dep = np.around(self.sediments(Qpass_volume) * (1 - perc_dep), decimals = roundpar)
            else:
                Qpass_dep = self.sediments(Qpass_volume) * (1 - perc_dep)

            # Volume from the incoming volume to be kept in the active layer:
            Qpass_act = self.sediments(Qpass_volume) - Qpass_dep
            # Re add the provenance column:
            V_inc2act = np.hstack((self.provenance(Qpass_volume)[:,None], Qpass_act))
            V_inc2dep = np.hstack((self.provenance(Qpass_volume)[:,None], Qpass_dep))

            # Add V_inc2dep to Vdep:
            # If, given the round, the deposited volume of the incoming cascades is not 0:
            if any(np.sum(self.sediments(Qpass_volume) * (1 - perc_dep), axis = 0)):
                V_dep = np.vstack((V_dep_old, V_inc2dep))
            else:
                # Otherwise, I leave the deposit volume as it was.
                V_dep = V_dep_old

            # Create an empty layer for the deposit volume to be put in the active layer:
            V_dep2act = np.append(reach_provenance_idx, np.zeros((1, self.n_classes)))
            if V_dep2act.ndim == 1:
                V_dep2act = V_dep2act[None, :]

        # Remove empty rows (if the matrix is not already empty)
        if (np.sum(self.sediments(V_dep2act), axis = 1) != 0).any():
            V_dep2act = V_dep2act[np.sum(self.sediments(V_dep2act), axis = 1) != 0, :]

        # Find the GSD of the active layer, for the transport capacity calculation:
        total_sum = np.sum(self.sediments(V_dep2act)) + np.sum(self.sediments(V_inc2act))
        sum_per_class = np.sum(self.sediments(V_dep2act), axis=0) + np.sum(self.sediments(V_inc2act), axis=0)
        Fi_r_reach = sum_per_class / total_sum
        # If V_act is empty, I put Fi_r equal to 0 for all classes
        Fi_r_reach[np.isinf(Fi_r_reach) | np.isnan(Fi_r_reach)] = 0

        return V_inc2act, V_dep2act, V_dep, Fi_r_reach


    def tr_cap_deposit(self, V_inc2act, V_dep2act, V_dep_not_act, tr_cap, roundpar):
        '''
        INPUTS:
        V_inc2act :  incoming volume that is in the maximum mobilisable volume (active layer)
                    (n_layers x n_classes + 1)
        V_dep2act :  deposit volume that is in the maximum mobilisable volume (active layer)
                    (n_layers x n_classes + 1)
        V_dep_not_act     :  remaining deposit volume
                    (n_layers x n_classes + 1)
        tr_cap    :  volume that can be mobilised during the time step according to transport capacity
                    (x n_classes)
        roundpar  : number of decimals to round the volumes
        '''

        # Identify classes for which the incoming volume in the active layer
        # is under the transport capacity:
        under_capacity_classes = tr_cap > np.sum(self.sediments(V_inc2act), axis=0)

        # If they are any, sediment will have to be mobilised from V_dep2act,
        # taking into consideration the sediment stratigraphy
        # (upper layers get mobilized first)
        if np.any(under_capacity_classes):
            # sum of under capacity classes in the incoming volume:
            sum_classes = np.sum(V_inc2act[:, np.append(False, under_capacity_classes)], axis=0)
            # remaining active layer volume per class after considering V_inc2act
            # (for under capacity classes only)
            tr_cap_remaining = tr_cap[under_capacity_classes] - sum_classes
            # select columns in V_dep2act corresponding to under_capacity_classes
            V_dep2act_class = V_dep2act[:, np.append(False, under_capacity_classes)]
            # Cumulate volumes from last to first row
            # Reminder: Last row is the top layer
            csum = np.cumsum(V_dep2act_class[::-1], axis = 0)[::-1]

            # Find the indexes of the lowermost layer falling within tr_cap_remaining, for each class.
            # In case tr cap remaining is higher than the total volume in Vdep_2_act, for a given class,
            # we take the bottom class (first row):
            mapp = csum >= tr_cap_remaining
            mapp[0, np.any(~mapp, axis = 0)] = True # force bottom layer to be true (first row)
            firstoverthresh = (mapp * 1).argmin(axis = 0)
            firstoverthresh = firstoverthresh - 1
            firstoverthresh[firstoverthresh == -1] = csum.shape[0] - 1
            # Finally, we obtain a binary matrix indicating the threshold layers:
            # (DD: maybe there is a more elegant way to find this matrix ?)
            mapfirst = np.zeros(mapp.shape)
            mapfirst[firstoverthresh, np.arange(np.sum(under_capacity_classes*1))] = 1
            # Now compute the percentage to be lifted from the layer "on the threshold":
            sum_layers_above_threshold = np.sum(np.where(mapp == False, V_dep2act_class, 0), axis=0)
            remaining_for_split_threshold = tr_cap_remaining - sum_layers_above_threshold
            perc_threshold = remaining_for_split_threshold/V_dep2act_class[firstoverthresh, np.arange(np.sum(under_capacity_classes*1))]
            # limit to a maximum of 1, in case the tr_cap_remaining is higher than the total volume (see above),
            # and we had taken the bottom layer:
            perc_dep = np.minimum(perc_threshold, 1)

            # Final matrix indicating the percentage we take from V_dep2act_class:
            # (To multiply to V_dep2act_class)
            map_perc = mapfirst * perc_dep + ~mapp*1

            # The matrix V_dep2act_new contains the mobilized cascades from
            # the deposit layer, now corrected according to the tr_cap:
            V_dep2act_new = np.zeros(V_dep2act.shape)
            self.provenance(V_dep2act_new)[:] = self.provenance(V_dep2act)
            V_dep2act_new[:,np.append(False, under_capacity_classes)== True] = map_perc * V_dep2act_class
            # Round the volume:
            if ~np.isnan(roundpar):
                self.sediments(V_dep2act_new)[:]  = np.around(self.sediments(V_dep2act_new), decimals=roundpar)

            # The matrix V_2dep contains the cascades that will be deposited into the deposit layer.
            # (the new volumes for the classes in under_capacity_classes and all the volumes in the remaining classes)
            V_2dep = np.zeros(V_dep2act.shape)
            # add all volume in other (over capacity) classes
            V_2dep[: , np.append(True, ~under_capacity_classes) == True] = V_dep2act[: , np.append(True, ~under_capacity_classes) == True]
            # add remaining volume in uncer capacity classe
            V_2dep_class = V_dep2act_class - V_dep2act_new[:,np.append(False, under_capacity_classes)== True]
            V_2dep[: , np.append(False, under_capacity_classes) == True] = V_2dep_class
            #V_2dep[: , np.append(False, under_capacity_classes) == True] = (1 - map_perc) * V_dep2act_class

            # Round the volume:
            if ~np.isnan(roundpar):
                self.sediments(V_2dep)[:]  = np.around(self.sediments(V_2dep), decimals=roundpar)

        # If they are no sediment to be mobilised from V_dep2act,
        # I re-deposit all the matrix V_dep2act into the deposit layer:
        else:
            V_2dep = V_dep2act
            # V_dep2act_new is empty:
            V_dep2act_new = np.zeros(V_dep2act.shape)
            V_dep2act_new[0] = 0 # EB:0 because it should be the row index (check whether should be 1)

        # For the classes where V_inc2act is enough, I deposit the cascades
        # proportionally
        # TODO : DD, now in this new version, there is never volume incoming in this function -> to be adapted
        sum_classes_above_capacity = np.sum(V_inc2act[: , np.append(False, ~under_capacity_classes) == True], axis = 0)
        # percentage to mobilise from the above_capacity classes:
        perc_inc = tr_cap[~under_capacity_classes] / sum_classes_above_capacity
        perc_inc[np.isnan(perc_inc)] = 0 # change NaN to 0 (naN appears when both tr_cap and sum(V_inc2act) are 0)
        class_perc_inc = np.zeros(under_capacity_classes.shape)
        class_perc_inc[under_capacity_classes == False] = perc_inc
        # Incomimg volume that is effectively mobilised, according to tr_cap:
        V_inc2act_new = V_inc2act*(np.append(True,under_capacity_classes)) + V_inc2act*np.append(False, class_perc_inc)

        # Mobilised volume :
        V_mob = np.vstack((V_dep2act_new, V_inc2act_new))
        V_mob = self.matrix_compact(V_mob)
        # Round:
        if ~np.isnan(roundpar):
            self.sediments(V_mob)[:] = np.around(self.sediments(V_mob), decimals = roundpar)

        # Compute what is to be added to V_dep from Q_incomimg:
        # DD: again in V2, there in no Q_incoming in this function anymore -> to be adapted
        class_residual = np.zeros(under_capacity_classes.shape);
        class_residual[under_capacity_classes==False] = 1 - perc_inc
        V_inc2dep = V_inc2act*np.hstack((1, class_residual))

        # Final volume from active layer to be put in Vdep:
        V_2dep = np.vstack((V_2dep, V_inc2dep))
        # Round:
        if ~np.isnan( roundpar ):
            self.sediments(V_2dep)[:] = np.around(self.sediments(V_2dep), decimals=roundpar)

        # Put the volume exceeding the transport capacity (V_2dep) back in the deposit:
        # (If the upper layer in the deposit and the lower layer in the volume to be
        #deposited are from the same reach, i sum them)
        V_dep = np.copy(V_dep_not_act)
        if (V_dep[-1,0] == V_2dep[0,0]):
            V_dep[-1,1:] = V_dep[-1,1:] + V_2dep[0,1:]
            V_dep = np.vstack((V_dep, V_2dep[1:,:]))
        else:
            V_dep = np.vstack((V_dep, V_2dep))

        # Remove empty rows:
        if not np.sum(self.sediments(V_dep2act)) == 0:
            V_dep = V_dep[np.sum(self.sediments(V_dep), axis = 1) != 0]

        return V_mob, V_dep



    def deposit_from_passing_sediments(self, V_remove, cascade_list, roundpar, n, t):
        ''' This function remove the quantity V_remove from the list of cascades.
        The order in which we take the cascade is from largest times (arriving later)
        to shortest times (arriving first). Hypotheticaly, cascade arriving first
        are passing in priority, in turn, cascades arriving later are deposited in priority.
        (DD: can be discussed)
        If two cascades have the same time, they are processed as one same cascade.

        INPUTS:
        V_remove : quantity to remove, per sediment class (array of size number of sediment classes).
        cascade_list : list of cascades. Reminder, a cascade is a Cascade class with attributes:
                        direct provenance, elapsed time, and the volume
        roundpar : number of decimals to round the cascade volumes (Vm)
        RETURN:
        r_Vmob : removed volume from cascade list
        cascade_list : the new cascade list, after removing the volumes
        V_remove : residual volume to remove
        '''
        removed_Vm_all = []

        # Order cascades according to the inverse of their elapsed time
        # and put cascade with same time in a sublist, in order to treat them together
        sorted_cascade_list = sorted(cascade_list, key=lambda x: np.nanmean(x.elapsed_time), reverse=True)
        sorted_and_grouped_cascade_list = [list(group) for _, group in groupby(sorted_cascade_list, key=lambda x: np.nanmean(x.elapsed_time))]

        # Loop over the sorted and grouped cascades
        for cascades in sorted_and_grouped_cascade_list:
            Vm_same_time = np.concatenate([casc.volume for casc in cascades], axis=0)
            if np.any(self.sediments(Vm_same_time)) == False: #In case Vm_same_time is full of 0
                del cascades
                continue
            # Storing matrix for removed volumes
            removed_Vm = np.zeros_like(Vm_same_time)
            self.provenance(removed_Vm)[:] = self.provenance(Vm_same_time) # same first col with initial provenance
            for col_idx in range(self.sediments(Vm_same_time).shape[1]):  # Loop over sediment classes
                if V_remove[col_idx] > 0:
                    col_sum = np.sum(Vm_same_time[:, col_idx+1])
                    if col_sum > 0:
                        fraction_to_remove = min(V_remove[col_idx] / col_sum, 1.0)
                        # Subtract the fraction_to_remove from the input cascades objects (to modify them directly)
                        for casc in cascades:
                            Vm = casc.volume
                            removed_quantities = Vm[:, col_idx+1] * fraction_to_remove
                            Vm[:, col_idx+1] -= removed_quantities
                            # Round Vm
                            Vm[:, col_idx+1] = np.round(Vm[:, col_idx+1], decimals = roundpar)
                            # Ensure no negative values
                            if np.any(Vm[:, col_idx+1] < -10**(-roundpar)) == True:
                                raise ValueError("Negative value in VM is strange")
                            # Store removed volume in direct connectivity matrix:
                            self.direct_connectivity[t][casc.provenance, n, col_idx] += np.sum(removed_quantities)

                        # Store the removed quantities in the removed volumes matrix
                        removed_Vm[:, col_idx+1] = Vm_same_time[:, col_idx+1] * fraction_to_remove
                        # Update V_remove by subtracting the total removed quantity
                        V_remove[col_idx] -= col_sum * fraction_to_remove
                        # Ensure V_remove doesn't go under the number fixed by roundpar
                        if np.any(V_remove[col_idx] < -10**(-roundpar)) == True:
                            raise ValueError("Negative value in V_remove is strange")
            # Round and store removed volumes
            self.sediments(removed_Vm)[:] = np.round(self.sediments(removed_Vm), decimals=roundpar)
            removed_Vm_all.append(removed_Vm)
        # Concatenate all removed quantities into a single matrix
        r_Vmob = np.vstack(removed_Vm_all) if removed_Vm_all else np.array([])
        # Gather layers of same original provenance in r_Vmob
        r_Vmob = self.matrix_compact(r_Vmob)

        # Delete cascades that are now only 0 in input cascade list
        cascade_list = [cascade for cascade in cascade_list if not np.all(self.sediments(cascade.volume) == 0)]

        # The returned cascade_list is directly modified by the operations on Vm
        return r_Vmob, cascade_list, V_remove


    def check_mass_balance(self, t, n, delta_volume_reach):
        ''' Definition to check the mass balance at time step t in reach n
        '''
        tot_out = np.sum(self.Qbi_mob[t][:, n, :], axis = 0)
        tot_in = np.sum(self.Qbi_tr[t][:, n, :], axis = 0)
        mass_balance_ = tot_in - tot_out - delta_volume_reach
        if np.any(mass_balance_ != 0) == True:
            self.mass_balance[t, n, :] = mass_balance_
        if np.abs(np.sum(mass_balance_)) >= 100:
            # DD: 100 is what I consider as a big volume loss
            print('Warning, the mass balance loss is higher than 100 m^3')

    def change_slope(self, Node_el_t, Lngt, Network , **kwargs):
        """"CHANGE_SLOPE modify the Slope vector according to the changing elevation of
        the nodes: It also guarantees that the slope is not negative or lower then
        the min_slope value by changing the node elevation bofore findin the SLlpe"""

        #define minimum reach slope


        #initialization
        if len(kwargs) != 0:
            min_slope = kwargs['s']
        else:
            min_slope = 0

        outlet = Network['n_hier'][-1]
        down_node = Network['downstream_node']
        down_node = np.array([int(n) for n in down_node])
        down_node[int(outlet)] = (len(Node_el_t)-1)

        Slope_t = np.zeros(Lngt.shape)

        #loop for all reaches
        for n in range(len(Lngt)):
            #find the minimum node elevation to guarantee Slope > min_slope
            min_node_el = min_slope * Lngt[n] + Node_el_t[down_node[n]]

            #change the noide elevation if lower to min_node_el
            Node_el_t[n] = np.maximum(min_node_el, Node_el_t[n] )

            #find the new slope
            Slope_t[n] = (Node_el_t[n] - Node_el_t[down_node[n]]) / Lngt[n]


        return Slope_t, Node_el_t


    def matrix_compact(self, volume):
        # Function that groups layers (rows) in volume
        # according to the original provenance (first column)

        # INPUT:
        # volume: sediment volume or cascade (n_layers x n_classe + 1).
        #           The first column is the original provenance.
        # RETURN:
        # volume_compacted: volume where layers have been summed by provenance


        idx = np.unique(self.provenance(volume)) # Provenance reach indexes
        volume_compacted = np.empty((len(idx), volume.shape[1]))
        # sum elements with same ID
        for ind, i in enumerate(idx):
            vect = volume[self.provenance(volume) == i,:]
            volume_compacted[ind,:] = np.append(idx[ind], np.sum(self.sediments(vect), axis = 0))

        if volume_compacted.shape[0]>1:
            volume_compacted = volume_compacted[np.sum(self.sediments(volume_compacted), axis = 1) != 0]

        if volume_compacted.size == 0:
            volume_compacted = (np.hstack((idx[0], np.zeros(self.sediments(volume).shape[1])))).reshape(1,-1)

        return volume_compacted



