"""
Created on Tue Oct 29 10:58:54 2024

@author: diane
"""
import copy
from itertools import groupby

import numpy as np

from cascade import Cascade
from d_finder import D_finder
from transport_capacity_calculator import TransportCapacityCalculator

np.seterr(divide='ignore', invalid='ignore')


class SedimentarySystem:
    '''
    @brief Network scale sedimentary system object.

    Contains all informations regarding the sedimentary system, i.e. its initialisation and its
    dynamicity through the simulation.

    Globaly contains:
        - The reach initial morphological informations
        - The sediment transfer functions
        - The sediment storing matrices

    @param reach_data
        The reach data inputs (ReachData object)
    @param network
        Network structure (graph)
    @param timescale
        Time step number (int)
    @param ts_length
        Time step length (float)
    @param save_dep_layer
        Option for saving deposit layer
    @param psi
        Sediment size class vector in psi scale (1d array)
    @param phi
        Porosity (default: 0.4)
    @param minvel
        minimum velocity (default: 0.0000001)
    @param n_metadata
        number of metadata column (default: 1, for storing initial provenance)
    '''


    def __init__(self, reach_data, network, timescale, ts_length, save_dep_layer,
                 psi, phi = 0.4, minvel = 0.0000001, n_metadata = 1):

        self.reach_data = reach_data
        self.network = network
        self.timescale = timescale
        self.ts_length = ts_length
        self.save_dep_layer = save_dep_layer
        self.n_metadata = n_metadata
        self.n_classes = len(psi)
        self.n_reaches = reach_data.n_reaches
        self.psi = psi
        self.phi = phi                          # sediment porosity
        self.minvel = minvel
        self.outlet = int(network['outlet'])    # outlet reach ID identification


        # Storing matrices (and related options)
        self.Qbi_dep = None
        self.external_inputs = None
        self.force_pass_external_inputs = None
        self.Qbi_tr = None
        self.Qbi_mob = None
        self.Qbi_mob_from_r = None
        self.V_sed = None
        self.Q_out = None
        self.update_slope = None
        self.indx_slope_red = None
        self.indx_width_calc = None
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
        self.al_depth_method = None
        self.vl_height = self.create_2d_zero_array()
        self.mass_balance = self.create_3d_zero_array()


        # temporary ?
        self.Qbi_dep_0 = None
        self.Qc_class_all = None        # DD: can it be optional ?


    def sediments(self, matrix):
        '''
        Access the sediment columns of the matrix.
        @warning Use self.sediments(matrix) for access, but use self.sediments(matrix)[:] for assignment!
        '''
        if matrix.ndim == 1:
            return matrix[self.n_metadata:]
        elif matrix.ndim == 2:
            return matrix[:, self.n_metadata:]

    def metadata(self, matrix):
        '''
        Access the metadata columns of the matrix.
        @warning Use self.metadata(matrix) for access, but use self.metadata(matrix)[:] for assignment!
        '''
        if matrix.ndim == 1:
            return matrix[:self.n_metadata]
        elif matrix.ndim == 2:
            return matrix[:, :self.n_metadata]

    def provenance(self, matrix):
        '''
        Access the provenance column of the matrix.
        @warning Use self.provenance(matrix) for access, but use self.provenance(matrix)[:] for assignment!
        '''
        if matrix.ndim == 1:
            return matrix[0]
        elif matrix.ndim == 2:
            return matrix[:, 0]

    def create_volume(self, provenance=None, etime=None, metadata=None, gsd=None):
        '''
        Create a volume of sediment with a set provenance, a possible erosion time and a grain size distribution.
        '''
        # Setting the metadata
        if provenance is not None:
            metadata_list = [provenance]
            if self.n_metadata > 1:
                if etime is not None:
                    metadata_list.append(etime)
                else:
                    metadata_list.append(np.nan)
        if metadata is not None:
            assert provenance is None
            assert etime is None
            metadata_list = metadata

        # Grain size distribution data
        if gsd is not None:
            gsd = gsd
        else:
            gsd = np.zeros(self.n_classes)

        volume = np.hstack((metadata_list, gsd))
        is_single = gsd.ndim == 1
        if is_single:
            volume = volume.reshape(1,-1)

        return volume

    def create_4d_zero_array(self):
        '''
        Initialise a 4d storing matrice (time step x initial reach x current reach x sediment size class)
        @note we add the time as a list, otherwise we can not look at the 4d matrix in spyder.
        '''
        return [np.zeros((self.n_reaches, self.n_reaches, self.n_classes)) for _ in range(self.timescale)]

    def create_3d_zero_array(self):
        """
        Initialise a 3d storing matrice (time step x reach x sediment size class)
        """
        return np.zeros((self.timescale, self.n_reaches, self.n_classes))

    def create_2d_zero_array(self):
        """
        Initialise a 2d storing matrice (time step x reach)
        """
        return np.zeros((self.timescale, self.n_reaches))

    def initialize_slopes(self, update_slope, indx_slope_red):
        '''
        Initialise slopes options and storing matrices.

        @param update_slope
            Option for updating slope with deposit or not (bool)
        @param indx_slope_red
            Option for the slope reduction - see Pitscheider et al. (int)
        '''
        self.update_slope = update_slope
        self.indx_slope_red = indx_slope_red

        # Initialise slope matrices
        self.min_slope = min(self.reach_data.slope)  # put a minimum value to guarantee movement  (DD: why min of the reach slopes ?)
        self.slope = self.create_2d_zero_array()
        self.slope[0,:] = np.maximum(self.reach_data.slope, self.min_slope)
        self.slope[1,:] = np.maximum(self.reach_data.slope, self.min_slope)

        # In case of constant slope
        if update_slope == False:
            self.slope[:,:] = self.slope[0,:]


    def initialize_widths(self, indx_width_calc):
        '''
        Initialise width options and storing matrices.

        @param indx_width_calc
            Option for the width variations - e.g. with hydrology (int)
        '''
        self.indx_width_calc = indx_width_calc

        # Initialise width matrices
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

    def initialize_elevations(self, update_slope):
        """
        Initialize node elevations.
        @note For each reach the matrix reports the fromN elevation
        @note The last column reports the outlet ToNode elevation (last node of the network)
        """

        self.node_el = np.zeros((self.timescale, self.n_reaches + 1))
        self.node_el[0,:] = np.append(self.reach_data.el_fn, self.reach_data.el_tn[self.outlet])
        self.node_el[1,:] = np.append(self.reach_data.el_fn, self.reach_data.el_tn[self.outlet])
        # Fix last node elevation for all time steps:
        self.node_el[:,-1] = self.node_el[1,-1]

        # In case of constant slope:
        if update_slope == False:
            self.node_el[:,: ] = self.node_el[0,:]


    def initialize_storing_matrices(self):
        """ Initialize all matrices used for storing sediment transport during the simulations
        """
        # Create Qbi dep matrix with size size depending on how often we want to save it:
        # Note: Qbi_dep stores the state of the deposit layer Vdep at the beginning of the time step 
        # Note: t = 0 is the initial deposit layer 
        # Note: an extra time column (+1) is used to represent the state of Vdep at the end of the last time step
        
        if self.save_dep_layer=='never':
            dep_save_number = 1
        if self.save_dep_layer=='yearly':
            dep_save_number = int(self.timescale / 365) + 1  
        if self.save_dep_layer=='always':
            dep_save_number = self.timescale + 1
        self.Qbi_dep = [[np.expand_dims(np.zeros(self.n_metadata + self.n_classes), axis = 0) for _ in range(self.n_reaches)] for _ in range(dep_save_number)]

        # Initial Qbi_dep:
        self.Qbi_dep_0 = [np.expand_dims(np.zeros(self.n_metadata + self.n_classes), axis = 0) for _ in range(self.n_reaches)] # Initialise sediment deposit in the reaches

        # Moving sediments storing matrice
        self.Qbi_mob = self.create_4d_zero_array() # Volume leaving the reach (gives also original provenance)
        self.Qbi_mob_from_r = self.create_4d_zero_array() # Volume mobilised from reach (gives also original provenance)
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
        """ Initialize initial sediment deposit layer in each reach

        @ param Qbi_dep_in
            Matrice with user-defined initial deposit volumes (3d array, size: n_reaches, 1, n_classes)
        """

        for n in self.network['n_hier']:
            # if no inputs are defined, initialize deposit layer with a single cascade with no volume and GSD equal to 0
            q_bin = np.array(Qbi_dep_in[n])
            if not q_bin.any(): #if all zeros
                self.Qbi_dep_0[n] = self.create_volume(provenance=n)
            else:
                assert q_bin.shape[0] == 1
                self.Qbi_dep_0[n] = self.create_volume(provenance=n, gsd=Qbi_dep_in[n, 0])
                self.Fi_al[0,n,:] = np.sum(q_bin, axis=0) / np.sum(q_bin)
                self.D50_al[0,n] = D_finder(self.Fi_al[0,n,:], 50, self.psi)

        self.Qbi_dep[0] = copy.deepcopy(self.Qbi_dep_0)  # store init condition of dep layer


    def set_erosion_maximum(self, eros_max_depth_, roundpar):
        """
        Set maximum volume in cubic meters that can be eroded for each reach, for each time step.
        @note By default the active layer (see below) and the erosion maximum are the same

        @param eros_max_depth_
            Depth of the erodible layer per time step (float or string, e.g. '2D90')
        """
        # Set maximum volume in meters that can be eroded for each reach, for each time step.
        self.eros_max_vol = self.create_2d_zero_array()
        self.eros_max_depth = self.create_2d_zero_array()

        if eros_max_depth_ == '2D90':
            # We take the input D90, or if not provided, the D84:
            if self.reach_data.D90 is not None:
                reference_d = self.reach_data.D90
            else:
                reference_d = self.reach_data.D84
            # Multiply by two, + apply a minimum threshold
            eros_max_t = np.maximum(2 * reference_d, 0.01)
        elif isinstance(eros_max_depth_, (int, float)):
            # Apply the input eros depth
            eros_max_t = eros_max_depth_ * np.ones(self.n_reaches)
        else:
            raise ValueError('As options for the eros max depth, you can choose "2D90" or a fixed number')

        # Compute the erodible volumes (all reaches)
        eros_vol_t = eros_max_t * self.reach_data.wac * self.reach_data.length
        # Store it for all time steps:
        self.eros_max_vol = np.tile(eros_vol_t, (self.timescale, 1))
        self.eros_max_depth = np.tile(eros_max_t, (self.timescale, 1))


    def set_active_layer(self, al_depth_, al_depth_method):
        """
        Set active layer volume in cubic meters.
        It is the layer used for computing the GSD for the transport capacity calculation.

        @note By default it is defined as 2.D90 [Parker 2008] but this is maybe not adapted for sandy rivers.
        @note By default the active layer and the erosion maximum are the same

        @param al_depth_
            Depth of the active layer (float or string, e.g. '2D90')
        @param al_depth_method
            Option to select the active layers,
            1: from the reach deposit layer top, the possible passing through cascade are then added at the top
            2: from the top, including possible passing cascades. In this case,
            al_depth and eros_max, even if they are equal do not include the same layers

        """
        self.al_vol = self.create_2d_zero_array()
        self.al_depth = self.create_2d_zero_array()

        if al_depth_ == '2D90':
            # We take the input D90, or if not provided, the D84:
            if self.reach_data.D90 is not None:
                reference_d = self.reach_data.D90
            else:
                reference_d = self.reach_data.D84
            # Multiply by two, + apply a minimum threshold
            al_depth_t = np.maximum(2 * reference_d, 0.01)
        elif isinstance(al_depth_, (int, float)):
            # Apply the input AL depth
            al_depth_t = al_depth_ * np.ones(self.n_reaches)
        else:
            raise ValueError('As options for the AL depth, you can choose "2D90" or a fixed number')

        # Compute the AL volumes (all reaches)
        al_vol_t = al_depth_t * self.reach_data.wac * self.reach_data.length
        # Store it for all time steps:
        self.al_vol = np.tile(al_vol_t, (self.timescale, 1))
        self.al_depth = np.tile(al_depth_t, (self.timescale, 1))

        # Set option for calculating the al depth (from top of the possible passing through cascades or from their bottom)
        self.al_depth_method = al_depth_method


    def set_velocity_section_height(self, vel_section_height, h, t):
        '''
        Set characteristic traveling height, for estimating velocity [m/s] from sediment flux [m3/s]

        @param vel_section_height
            The characteristic traveling height, hv.
            Possibilities:  '2D90': twice the input D90;'0.1_hw': 10% of the water column; Or a fixed value.

        @note the velocity is calculated as: v = Qs/S, with S the traveling section, S = W x hv x (1-phi)
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
        """
        Set external inputs (lateral sources).

        @param external_inputs
            The external input matrix
        @force_pass_external_inputs
            Option to for the external input to be passed entirely to the downstream reach (bool)
        """

        self.external_inputs = external_inputs
        self.force_pass_external_inputs = force_pass_external_inputs


    def extract_external_inputs(self, cascade_list, t, n):
        """
        Create a new cascade object in reach n at time step t, from external input,
        to be added to the cascade list.

        @param cascade_list
            list of cascades currently in the reach
        @param t
            current time step
        @param n
            current reach index

        @return
            the cascade list updated with the external input
        """
        if np.any(self.external_inputs[t, n, :] > 0):
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

        """
        Compute the velocity of all the cascade objects in cascade_list.

        @param cascade_list
            List of cascade objects
        @param Vdep
            Reach deposit layer
        @param Q_reach
            Reach water discharge [m3/s] at this time step
        @param v
            Reach water velocity [m/s]
        @param h
            Reach water height
        @param t
            Time step
        @param n
            Reach index
        @param indx_velocity
            Index for velocity calculation method
        @param indx_vel_partition
            Index for velocity calculation method among the different size classes
        @param indx_tr_cap
            Index for transport capacity formula
        @param indx_tr_partition
            Index for transport capacity paritionning

        @note
            The velocity must be assessed by re-calculating the transport capacity [m3/s]
            in the present reach considering all arriving cascade(s).
        @note
            Two methods are proposed to evaluated this transport capacity, chosen
            by the indx_velocity parameter: 1) the simplest, we re-calculate the transport
            capacity on each cascade independantly.
            2) we consider the active layer volume. This will potentially add the influence of
            some reach deposited material.
        @note
            The velocity can be either the same among the sediment size classes
            or set proportionnaly to the fluxes Qi
        """


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

        '''
        Compute the velocity of one sediment volume.

        @param volume
            one sediment volume
        @param Q_reach
            Reach water discharge [m3/s] at this time step
        @param v
            Reach water velocity [m/s]
        @param h
            Reach water height
        @param t
            Time step
        @param n
            Reach index
        @param indx_vel_partition
            Index for velocity calculation method among the different size classes
        @param indx_tr_cap
            Index for transport capacity formula
        @param indx_tr_partition
            Index for transport capacity paritionning

        @return
            the computed velocities (1d array, size: n_classes)

        @note
            The transport capacity [m3/s] is calculated on this volume,
            and the velocity is calculated by dividing the transport capacity
            by a section (hVel x width x (1 - porosity)).

        @note
            For partionning the section among the different sediment size class in the volume,
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
        if Svel == 0 or np.isnan(Svel) == True:
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
        '''
        Fonction to decide if the traveling cascades in cascade list stop in
        the reach or not, due to the end of the time step.


        @param cascade_list
            List of cascades objects
        @param n
            Reach index
        @param t
            Time step

        @return cascade_list_new
            List of cascades objects updated. Stopping cascades or partial volumes have been removed
        @return depositing_volume
            The volume to be deposited in this reach. Stopped by time.

        @note
            Cascades are ordered according to their arrival time at the inlet,
            so that volume arriving first deposit first.


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
        Decides if a volume of sediments (Vm) will stop in this
        reach at the end of the time step or not.

        @note Part of the volume can stop or continue.

        @param t_new
            Elapsed time since beginning of time step for Vm, for each sed class
        @param Vm
            Volume of sediments

        @return Vm_stop
            Stopping volume
        @return Vm_continue
            Continuing volume

        '''

        cond_stop = np.append([True]*self.n_metadata, [t_new>1])
        Vm_stop = np.zeros_like(Vm)
        Vm_stop[:, cond_stop] = Vm[:, cond_stop]

        cond_continue = np.append([True]*self.n_metadata, [t_new<=1])
        Vm_continue = np.zeros_like(Vm)
        Vm_continue[:, cond_continue] = Vm[:, cond_continue]

        if np.all(self.sediments(Vm_stop) == 0) == True:
            Vm_stop = None
        if np.all(self.sediments(Vm_continue) == 0) == True:
            Vm_continue = None

        return Vm_stop, Vm_continue


    def compute_time_lag(self, cascade_list):
        """
        @details
            This function is used only when the time_lag option is activated in the simulation.
            The time lag is the time before the first continuing cascade arrives
            at the outlet of the reach. It leave time to mobilise from the reach
            deposit layer in the meantime.

        @param
            list of continuing cascade objects. Can be empty.
        @return time_lag
            the time lag
        """

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
                                   passing_cascades = None):
        """
        Compute the transport capacity in reach n at time step t.

        @details
            The transport capacity [m3/s] is computed on the active layer, made of
            the deposit layer (Vdep) and eventually continuing cascades (passing_cascades)

        @param Vdep
            Reach deposit layer
        @param passing_cascades
            List of cascade (the ones that did not end the time step in this reach = continuing cascades)

        @return tr_cap_per_s
            Transport capacity in m3/s (array, size: n_classes)
        @return Fi_al_
            Sediment fraction per size class in the active layer
        @return D50_al_
            Active layer D50
        @return Qc
            Critical discharge (only for Rickenmann transport formula)
        """

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

        # Compute fraction and D50 in the active layer
        # TODO: warning when the AL is very small, we can have Fi_r is 0 due to roundpar

        if passing_volume is None:
            AL_volume = self.al_vol[t,n]
        else:
            if self.al_depth_method == 1:
                # Method 1: (default) if there are passing cascades, their total volume is added to the user-defined active volume
                sum_pass = np.sum(self.sediments(passing_volume))
                AL_volume = self.al_vol[t,n] + sum_pass
            elif self.al_depth_method == 2:
                # Method 2: the active depth is measured from the top of the passing cascades
                AL_volume = self.al_vol[t,n]


        _,_,_, Fi_al_ = self.layer_search(Vdep, AL_volume, Qpass_volume = passing_volume, roundpar = roundpar)


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


    def compute_mobilised_volumes(self, Vdep, tr_cap_per_s, n, t, roundpar,
                                 passing_cascades = None, time_fraction = None):

        """
        Compute the mobilised volumes in reach n at time step t.

        @details
            This function compute the mobilised volumes from a reach, considering
            the eventual continuig cascades (cascade coming from upstream but not
            stopped by time). It can mobilised a volume from the reach (Vmob) and/or
            deposit some of the continuing cascade (that are now stopped by energy).

        @note
            The order in which we select the continuing cascade to be deposited
            is from largest times (arriving later) to shortest times (arriving first).
            Hypotheticaly, cascade arriving first are passing in priority, in turn,
            cascades arriving later are deposited in priority.


        @param Vdep
            Reach deposit layer
        @param tr_cap_per_s
            Transport capacity
        @param passing_cascades
            List of cascade (the ones that did not end the time step in this reach = continuing cascades)
        @param time_fraction
            Fraction of the time step we are actually mobilising (created to consider the time lag option)

        @return Vmob
            New volume mobilised from the reach deposit layer
        @return passing_cascades
            Liste of cascades (continuing cascade that actually manage to pass the outlet)
        @return Vdep_new
            Updated reach deposit layer

        """

        # Case where we don't consider a time lag, the time for mobilising is the complete time step:
        if time_fraction is None:
            time_fraction = np.ones(self.n_classes)

        # Real time to mobilise:
        time_to_mobilise = time_fraction * self.ts_length
        # Mobilisable volume:
        volume_mobilisable = tr_cap_per_s * time_to_mobilise
        # Erosion maximum during the time lag
        # (we take the mean time lag among the classes)
        e_max_vol_ = self.eros_max_vol[t,n] * np.mean(time_fraction)

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

        # Sediment classes with negative values in diff_with_capacity are over capacity
        # They are deposited, i.e. directly added to Vdep
        diff_neg = -np.where(diff_with_capacity > 0, 0, diff_with_capacity)
        if np.any(diff_neg):
            Vm_removed, passing_cascades, residual = self.deposit_from_passing_sediments(np.copy(diff_neg), passing_cascades, roundpar, n, t)
            # Deposit the Vm_removed:
            Vdep_new = np.concatenate([Vdep_new, Vm_removed], axis=0)

        # Re-add the external cascades, that were excluded from calculation
        if self.force_pass_external_inputs == True and passing_cascades_excluded != []:
            passing_cascades.extend(passing_cascades_excluded)

        # If the new Vdep is empty, put an empty layer for next steps
        if Vdep_new.size == 0:
            Vdep_new = self.create_volume(provenance=n)

        return V_mob, passing_cascades, Vdep_new



    def layer_search(self, V_dep_old, V_lim, Qpass_volume = None, roundpar = None):
        """
        Select uppermost layers available in a reach.

        @details
            This function searches uppermost layers from a volume of layers,
            to correspond to a maximum volume (V_lim). Passing cascade can be integrated
            to the top of the volume (Qpass_volume).
            The maximum volume can represent for example the active layer,
            i.e. what we consider as active during the transport process,
            or a maximum to be eroded per time step.

        @param V_dep_old
            Reach deposit layer
        @param V_lim
            Volume limit (float)
        @param Qpass_volume
            Sediment volume to be added to the top of the reach deposit layer

        @return V_inc2act
            Layers of the traveling volume (Qpass_volume) to be put in the maximum volume
        @return V_dep2act
            Layers of the deposit volume (V_dep_old) to be put in the maximum volume
        @return V_dep_new
            Remaining deposit layer
        @return Fi_r_reach
            Fraction of sediment per sizes in the maximum volume
        """

        reach_metadata = V_dep_old[0, :self.n_metadata]

        if Qpass_volume is None:
            # Put an empty layer (for computation)
            Qpass_volume = self.create_volume(provenance=0)

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
                V_dep = np.c_[reach_metadata, np.zeros((1, self.n_classes))]

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

                # Re-add the metadata columns:
                threshold_layer_included = self.create_volume(metadata=self.metadata(threshold_layer), gsd=threshold_layer_included)
                threshold_layer_excluded = self.create_volume(metadata=self.metadata(threshold_layer), gsd=threshold_layer_excluded)
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
            # Re add the metadata columns:
            V_inc2act = self.create_volume(metadata=self.metadata(Qpass_volume), gsd=Qpass_act)
            V_inc2dep = self.create_volume(metadata=self.metadata(Qpass_volume), gsd=Qpass_dep)

            # Add V_inc2dep to Vdep:
            # If, given the round, the deposited volume of the incoming cascades is not 0:
            if any(np.sum(self.sediments(Qpass_volume) * (1 - perc_dep), axis = 0)):
                V_dep = np.vstack((V_dep_old, V_inc2dep))
            else:
                # Otherwise, I leave the deposit volume as it was.
                V_dep = V_dep_old

            # Create an empty layer for the deposit volume to be put in the active layer:
            V_dep2act = self.create_volume(metadata=reach_metadata)
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
            mask = np.append([False]*self.n_metadata, under_capacity_classes)
            sum_classes = np.sum(V_inc2act[:, mask], axis=0)
            # remaining active layer volume per class after considering V_inc2act
            # (for under capacity classes only)
            tr_cap_remaining = tr_cap[under_capacity_classes] - sum_classes
            # select columns in V_dep2act corresponding to under_capacity_classes
            V_dep2act_class = V_dep2act[:, mask]
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
            V_dep2act_new[:, mask] = map_perc * V_dep2act_class
            # Round the volume:
            if ~np.isnan(roundpar):
                self.sediments(V_dep2act_new)[:]  = np.around(self.sediments(V_dep2act_new), decimals=roundpar)

            # The matrix V_2dep contains the cascades that will be deposited into the deposit layer.
            # (the new volumes for the classes in under_capacity_classes and all the volumes in the remaining classes)
            V_2dep = np.zeros(V_dep2act.shape)
            # add all volume in other (over capacity) classes
            V_2dep[:, ~mask] = V_dep2act[:, ~mask]
            # add remaining volume in uncer capacity classe
            V_2dep_class = V_dep2act_class - V_dep2act_new[:, mask]
            V_2dep[:, mask] = V_2dep_class

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
        sum_classes_above_capacity = np.sum(V_inc2act[: , np.append([False]*self.n_metadata, ~under_capacity_classes) == True], axis = 0)
        # percentage to mobilise from the above_capacity classes:
        perc_inc = tr_cap[~under_capacity_classes] / sum_classes_above_capacity
        perc_inc[np.isnan(perc_inc)] = 0 # change NaN to 0 (naN appears when both tr_cap and sum(V_inc2act) are 0)
        class_perc_inc = np.zeros(under_capacity_classes.shape)
        class_perc_inc[under_capacity_classes == False] = perc_inc
        # Incomimg volume that is effectively mobilised, according to tr_cap:
        mask_above_capacity = np.append([True]*self.n_metadata, under_capacity_classes)#np.append([False]*self.n_metadata, ~under_capacity_classes)
        mask_under_capacity = np.append([False]*self.n_metadata, class_perc_inc)

        V_inc2act_new = V_inc2act * mask_above_capacity + V_inc2act * mask_under_capacity

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
        V_inc2dep = V_inc2act * np.hstack(([1]*self.n_metadata, class_residual))

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
        '''
        This function remove the quantity V_remove from the list of cascades.

        @note
            The order in which we take the cascade is from largest times (arriving later)
            to shortest times (arriving first). Hypotheticaly, cascade arriving first
            are passing in priority, in turn, cascades arriving later are deposited in priority.
            (DD: can be discussed)
            If two cascades have the same time, they are processed as one same cascade.

        @param V_remove
            Quantity to remove, per sediment class (1d array, size: n_classes).
        @param cascade_list
            List of cascades (continuing cascades)

        @return r_Vmob
            Removed volume from cascade list
        @return cascade_list
            The new cascade list
        @return V_remove
            Residual volume to remove

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
            for psi_idx in range(len(self.psi)):  # Loop over sediment classes
                col_idx = self.n_metadata + psi_idx
                if V_remove[psi_idx] > 0:
                    col_sum = np.sum(Vm_same_time[:, col_idx])
                    if col_sum > 0:
                        fraction_to_remove = min(V_remove[psi_idx] / col_sum, 1.0)
                        # Subtract the fraction_to_remove from the input cascades objects (to modify them directly)
                        for casc in cascades:
                            Vm = casc.volume
                            removed_quantities = Vm[:, col_idx] * fraction_to_remove
                            Vm[:, col_idx] -= removed_quantities
                            # Round Vm
                            Vm[:, col_idx] = np.round(Vm[:, col_idx], decimals = roundpar)
                            # Ensure no negative values
                            if np.any(Vm[:, col_idx] < -10**(-roundpar)) == True:
                                raise ValueError("Negative value in VM is strange")
                            # Store removed volume in direct connectivity matrix:
                            self.direct_connectivity[t][casc.provenance, n, psi_idx] += np.sum(removed_quantities)

                        # Store the removed quantities in the removed volumes matrix
                        removed_Vm[:, col_idx] = Vm_same_time[:, col_idx] * fraction_to_remove
                        # Update V_remove by subtracting the total removed quantity
                        V_remove[psi_idx] -= col_sum * fraction_to_remove
                        # Ensure V_remove doesn't go under the number fixed by roundpar
                        if np.any(V_remove[psi_idx] < -10**(-roundpar)) == True:
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
        '''
        Function used to check the mass balance at time step t in reach n
        '''
        tot_out = np.sum(self.Qbi_mob[t][:, n, :], axis=0)
        tot_in = np.sum(self.Qbi_tr[t][:, n, :], axis=0)
        mass_balance_ = tot_in - tot_out - delta_volume_reach
        if np.any(mass_balance_ != 0) == True:
            self.mass_balance[t, n, :] = mass_balance_
        if np.abs(np.sum(mass_balance_)) >= 100:
            # DD: 100 is what I consider as a big volume loss
            print('Warning, the mass balance loss is higher than 100 m^3')


    def update_node_elevation_with_deposit(self, t, n):
        """
        Update node elevation at t+1 according to deposits at t in the reach just downstream of the node.
        """
        # Total sediment budget (deposited or eroded) of reach n at time step t
        sed_budg_t_n = np.sum(self.sediment_budget[t,n,:])

        # Update the elevation of the upstream node of the reach, after Czuba (2017)
        up_reaches = self.network['upstream_node'][n] # upstream reaches
        sum_all_areas = np.sum(self.reach_data.wac[np.append(n, up_reaches)] * self.reach_data.length[np.append(n, up_reaches)]) # sum of areas upstream and downstream of the node
        delta_h = 2 * sed_budg_t_n / (sum_all_areas * (1 - self.phi))

        self.node_el[t+1, n] = self.node_el[t,n] + delta_h


    def change_slope(self, t, **kwargs):

        """
        Modifies the slope vector according to the changing elevation of the nodes.

        @param t
             the time step

        @note
            It also guarantees that the slope is not negative or lower than
            the min_slope value by changing the node elevation bofore findin the SLlpe
        """
        #Define minimum reach slope
        if len(kwargs) != 0:
            min_slope = kwargs['s']
        else:
            min_slope = 0

        # Loop for all reaches
        for n in range(self.n_reaches):
            r_length = self.reach_data.length[n]                    # reach length
            down_node = int(self.network['downstream_node'][n])     # reach downstream node

            if n == self.outlet:
                down_node = -1 #because in self.node_el, the outlet node it stored at the last column

            # Find the minimum elevation of the upstream node to guarantee Slope > min_slope
            min_node_el = min_slope * r_length + self.node_el[t+1, down_node]
            # Change the node elevation if lower to min_node_el
            self.node_el[t+1, n] = np.maximum(min_node_el, self.node_el[t+1, n])

            #Find the new slope and store it for the next time step
            self.slope[t+1,n] = (self.node_el[t+1, n] - self.node_el[t+1, down_node]) / r_length



    def matrix_compact(self, volume):
        """
        Function that groups layers (rows) in volume according to the original provenance (first column)

        @param volume
            the volume to be merged (n_layers x n_classe + 1)

        @return
            the volume compacted, where layers have been summed by provenance

        """

        provenance_ids = np.unique(self.provenance(volume)) # Provenance reach indexes
        volume_compacted = np.empty((len(provenance_ids), volume.shape[1]))
        # sum elements with same ID
        for ind, i in enumerate(provenance_ids):
            vect = volume[self.provenance(volume) == i,:]
            volume_compacted[ind,:] = self.create_volume(provenance=provenance_ids[ind], gsd=np.sum(self.sediments(vect), axis=0))

        if volume_compacted.shape[0] > 1:
            volume_compacted = volume_compacted[np.sum(self.sediments(volume_compacted), axis = 1) != 0]

        if volume_compacted.size == 0:
            volume_compacted = self.create_volume(provenance=provenance_ids[0])

        return volume_compacted


    def sort_by_init_provenance(self, volume, n):
        '''
        Function to sort the volume layers according to initial provenance.
        @note
            It was used in v1 to sort the incoming volume layers according to the distance to
            their initial provenance reach.
            DD: We keep it in v2. Removing it seems to substancially change the results (+/-10%)
        '''
        distancelist = self.network['upstream_distance_list'][n]

        idx = np.argwhere(volume[:, 0][:,None] == distancelist[~(np.isnan(distancelist))])[:,1]

        if idx.size != 0 and len(idx) != 1:  # if there is a match #EB check
            volume_sort = np.array(volume[(idx-1).argsort(), :])
        else:
            volume_sort = volume

        return volume_sort

