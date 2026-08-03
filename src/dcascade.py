"""
Created on Tue Oct 29 10:58:54 2024

@author: Diane Doolaeghe, Elisa Bozzolan, Anne Laure Argentin
"""
import copy
import os
import sys

# General imports
import numpy as np
import numpy.matlib
import pandas as pd
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')

from cascade import Cascade
from d_finder import D_finder
from flow_depth import choose_flow_depth
from sedimentary_system import SedimentarySystem
from slope_reduction import choose_slope_reduction
from width_variation import choose_width_variation
from hypsometry import update_hypsometric_hydraulics, hypso_transport_capacity


class DCASCADE:
    """
    @brief Main class of the D-CASCADE code. Used to run the algorithm and transfer sediments.

    @param sedim_sys Sedimentary system class
    """

    def __init__(self, sedim_sys: SedimentarySystem):

        self.sedim_sys = sedim_sys
        self.reach_data = sedim_sys.reach_data
        self.network = sedim_sys.network
        self.n_reaches = sedim_sys.n_reaches
        self.n_classes = sedim_sys.n_classes
        self.n_metadata = sedim_sys.n_metadata
        self.t_track = sedim_sys.t_track 

        # Simulation attributes
        self.timescale = sedim_sys.timescale   # time step number
        self.ts_length = sedim_sys.ts_length                  # time step length
        self.save_dep_layer = sedim_sys.save_dep_layer        # option for saving the deposition layer or not
        self.update_slope = sedim_sys.update_slope            # option for updating slope
        self.indx_slope_red = sedim_sys.indx_slope_red
        self.indx_width_calc = sedim_sys.indx_width_calc

        # Indexes
        self.indx_flo_depth = None
        self.indx_tr_cap = None
        self.indx_tr_partition = None
        self.indx_velocity = None
        self.indx_vel_partition = None


    def set_hydraulic_options(self, indx_flo_depth):
        self.indx_flo_depth = indx_flo_depth

    def set_transport_indexes(self, indx_tr_cap, indx_tr_partition):
        self.indx_tr_cap = indx_tr_cap
        self.indx_tr_partition = indx_tr_partition

    def set_velocity_options(self, indx_velocity, indx_vel_partition, vel_height_option):
        self.indx_velocity = indx_velocity
        self.indx_vel_partition = indx_vel_partition
        self.vel_height_option = vel_height_option


    def run(self, Q, roundpar):

        SedimSys = self.sedim_sys
        
        # Save for Pinzano
        Q_pinz = Q[:self.timescale, 37]
        h_0_pinz = np.zeros(self.timescale)
        h_1_pinz = np.zeros(self.timescale)
        w_0_pinz = np.zeros(self.timescale)
        w_1_pinz = np.zeros(self.timescale)
        
        tr_cap_1_pinz = np.zeros(self.timescale)
        tr_cap_2_pinz = np.zeros(self.timescale)
                        
        # start waiting bar
        for t in tqdm(range(self.timescale)):
            
            # Channel width calculation
            SedimSys.width = choose_width_variation(self.reach_data, SedimSys, Q, t, self.indx_width_calc)
            
            w_0_pinz[t] = SedimSys.width[t, 37]
            
            # Define flow depth and flow velocity for all reaches at this time step:
            h, v = choose_flow_depth(self.reach_data, SedimSys, Q, t, self.indx_flo_depth)
            SedimSys.flow_depth[t] = h
            SedimSys.water_velocity[t] = v
            
            ### save for Pinzano
            h_0_pinz[t] = h[37]

            
            if self.sedim_sys.hypso_code > 0:
                # Compute hypsometric flow for hypso reaches only. It will overwrite width and water height previously defined
                # Returns hypsometric water height, to be used for tr_cap if hypso code == 2                
                hypso_hw = update_hypsometric_hydraulics(self.reach_data, SedimSys, Q, t, self.indx_flo_depth)
                # h, and v need to be replaced to be used below (DD: maybe think of a more elegant way)
                h = SedimSys.flow_depth[t]
                v = SedimSys.water_velocity[t] 
                
                h_1_pinz[t] = hypso_hw[37]['h_mean']            
                w_1_pinz[t] = hypso_hw[37]['width']
                

            # Compute velocity section height (may be dependant on the water depth)
            SedimSys.set_velocity_section_height(self.vel_height_option, h, t)

            # Slope reduction functions
            SedimSys.slope = choose_slope_reduction(self.reach_data, SedimSys, Q, t, h, self.indx_slope_red)

            # Deposit layer from previous timestep
            Qbi_dep_old = copy.deepcopy(self.sedim_sys.Qbi_dep_0)


            # Matrix to store volumes of sediment passing through a reach
            # in this timestep, ready to go to the next reach in the same time step.
            # For each reach, stores list of Cascade objects.
            Qbi_pass = [[] for n in range(self.n_reaches)]

            # loop for all reaches:
            for n in self.network['n_hier']:

                # Extracts the deposit layer left in previous time step
                Vdep_init = Qbi_dep_old[n] # extract the deposit layer of the reach

                # Extract external cascade (if they are)
                if SedimSys.external_inputs is not None:
                    Qbi_pass[n] = SedimSys.extract_external_inputs(Qbi_pass[n], t, n)

                ###------Step 1 : Cascades generated from the reaches upstream during
                # the present time step, are passing the inlet of the reach
                # (stored in Qbi_pass[n]).
                # This computational step make them pass to the outlet of the reach
                # or stop in the reach, depending if their velocity make them
                # arrive at the outlet before the end of the time step or not.

                # Store the arriving cascades in the transported matrix (Qbi_tr)
                # Note: we store the volume by original provenance
                for cascade in Qbi_pass[n]:
                    SedimSys.Qbi_tr[t][[SedimSys.provenance(cascade.volume).astype(int)], n, :] += SedimSys.sediments(cascade.volume)

                # If specified, store also the erosion times of these cascades entering the reach
                # (can be a decimal number since we average the time of same provenance)
                if self.t_track == True: 
                    if Qbi_pass[n] != []:
                        concat_volume = np.concatenate([cascade.volume for cascade in Qbi_pass[n]], axis=0)
                        concat_volume = SedimSys.matrix_compact(concat_volume)
                        SedimSys.Qbi_tr_eros_times[t, SedimSys.provenance(concat_volume).astype(int), n] = SedimSys.metadata(concat_volume)[:, 1]

                # Compute the velocity of the cascades in this reach [m/s]
                if Qbi_pass[n] != []:
                    SedimSys.compute_cascades_velocities(Qbi_pass[n], Vdep_init,
                                               Q[t,n], v[n], h[n], roundpar, t, n,
                                               self.indx_velocity, self.indx_vel_partition,
                                               self.indx_tr_cap, self.indx_tr_partition)
                else:
                    SedimSys.V_sed[t, n, :] = np.nan

                # Decides weather cascades, or parts of cascades,
                # finish the time step here or not.
                # After this step, Qbi_pass[n] contains volume that do not finish
                # the time step in this reach.
                if Qbi_pass[n] != []:
                    Qbi_pass[n], to_be_deposited = SedimSys.cascades_end_time_or_not(Qbi_pass[n], n, t)
                else:
                    to_be_deposited = None

                # After this step, Qbi_pass[n] contains volume that do not finish
                # the time step in this reach, i.e the continuing cascades

                ###------Step 2 : Mobilise volumes from the reach considering the
                # eventual continuing cascades.

                # Compute transport capacity
                tr_cap_per_s, Fi_al, D50_al, Qc = SedimSys.compute_transport_capacity(Vdep_init, roundpar, t, n, Q, v, h,
                                                                                  self.indx_tr_cap, self.indx_tr_partition,
                                                                                  passing_cascades = Qbi_pass[n])
                # Save Pinz
                if n == 37:
                    tr_cap_1_pinz[t] = np.sum(tr_cap_per_s)

                # Hypsometric transport capacity calculation for hypso reaches.
                if SedimSys.hypso_code >=2 and n in hypso_hw.keys():                        
                    tr_cap_per_s, Fi_al_, D50_al_, Qc, hypso_tr_cap_per_s  = hypso_transport_capacity(Vdep_init, roundpar, t, n, Q, hypso_hw,                                                                                                                             
                                                                                              self.indx_tr_cap, self.indx_tr_partition,
                                                                                              SedimSys,
                                                                                              passing_cascades = Qbi_pass[n])                                                                                        
                    
                    # Save Pinz
                    if n == 37:
                        tr_cap_2_pinz[t] = np.sum(tr_cap_per_s)
                    
                    
                    # Save hypso tr_cap                        
                    hypso_hw[n]['hypso_tr_cap'] = hypso_tr_cap_per_s
                                   

                # Store transport capacity and active layer informations:
                SedimSys.Fi_al[t, n, :] = Fi_al
                SedimSys.D50_al[t, n] = D50_al
                SedimSys.Qc_class_all[t, n] = Qc
                SedimSys.tr_cap[t, n, :] = tr_cap_per_s * self.ts_length

                # Mobilise:
                Vmob, Qbi_pass[n], Vdep_end = SedimSys.compute_mobilised_volumes(Vdep_init, tr_cap_per_s,
                                                                                     n, t, roundpar,
                                                                                     passing_cascades = Qbi_pass[n])

                ###-----Step 3: Finalisation.
                # Add the cascades that were mobilised from this reach to Qbi_pass[n]:
                if Vmob is not None:
                    elapsed_time = np.zeros(self.n_classes)
                    provenance = n
                    Qbi_pass[n].append(Cascade(provenance, elapsed_time, Vmob))

                # Deposit the stopping cascades in Vdep
                if to_be_deposited is not None:
                    to_be_deposited = SedimSys.sort_by_init_provenance(to_be_deposited, n)
                    #DD: see if we remove line above, because theoretically to_be_deposited is already sorted
                    Vdep_end = np.concatenate([Vdep_end, to_be_deposited], axis=0)

                # Store Vdep for next time step
                SedimSys.Qbi_dep_0[n] = np.copy(Vdep_end)

                # Store cascades in the mobilised volume.
                # All cascades (passing + mobilised from reach):
                for cascade in Qbi_pass[n]:
                    SedimSys.Qbi_mob[t][[SedimSys.provenance(cascade.volume).astype(int)], n, :] += SedimSys.sediments(cascade.volume)
                # Cascades from reach only:
                if Vmob is not None:
                    SedimSys.Qbi_mob_from_r[t][[SedimSys.provenance(Vmob).astype(int)], n, :] += SedimSys.sediments(Vmob)

                # Finally, pass these cascades to the next reach (if we are not at the outlet)
                if n != SedimSys.outlet:
                    n_down = np.squeeze(self.network['downstream_node'][n], axis = 1)
                    n_down = int(n_down) # Note: This is wrong if there is more than 1 reach downstream (to consider later)
                    Qbi_pass[n_down].extend(copy.deepcopy(Qbi_pass[n]))
                else:
                    n_down = None
                    # If it is the outlet, we add the cascades to Qout and to the last column of the connectivity matrix
                    for cascade in Qbi_pass[n]:
                        SedimSys.Q_out[t, [SedimSys.provenance(cascade.volume).astype(int)], :] += SedimSys.sediments(cascade.volume)
                        SedimSys.direct_connectivity[t][cascade.provenance, -1, :] += np.sum(SedimSys.sediments(cascade.volume), axis = 0)

                # Store sediment budget:
                vol_out = np.sum(SedimSys.Qbi_mob[t][:, n, :], axis = 0) # sum over provenance
                vol_in = np.sum(SedimSys.Qbi_tr[t][:, n, :], axis = 0)
                SedimSys.sediment_budget[t, n, :] = vol_in - vol_out

                # Check sediment volume mass balance:
                delta_volume_reach = np.sum(SedimSys.sediments(SedimSys.Qbi_dep_0[n]), axis = 0) - np.sum(SedimSys.sediments(Qbi_dep_old[n]), axis = 0)
                SedimSys.check_mass_balance(t, n, delta_volume_reach)

                # Optional: Update the changes in bed elevation, due to deposition (+) or erosion (-)
                # Note: sediment budget at t, will update the node elevation at t+1
                if self.update_slope == True and t != self.timescale - 1:
                    SedimSys.update_node_elevation_with_deposit(t, n)
                    

            """End of the reach loop"""

            # Save Qbi_dep according to saving frequency
            if self.save_dep_layer == 'always':
                SedimSys.Qbi_dep[t+1] = copy.deepcopy(SedimSys.Qbi_dep_0)
            if self.save_dep_layer == 'yearly':
                if int(t+2) % 365 == 0 and t != 0:
                    t_y = int((t+2)/365)
                    SedimSys.Qbi_dep[t_y] = copy.deepcopy(SedimSys.Qbi_dep_0)

            # In case of changing slope, change the slope accordingly to the bed elevation (at t+1)
            if self.update_slope == True and t != self.timescale - 1:
                # DD: see what min slope value should be
                SedimSys.change_slope(t)

        # How many time the bottom was reached during the simulation
        if SedimSys.reach_bottom_count != 0:
            print("\n The deposit layer bottom was reached " + str(SedimSys.reach_bottom_count) + " times. \n")
            
            
        # For Pinzano plots
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(figsize=(7, 4))
        # ax.plot(Q_pinz, h_0_pinz, "o", label="h0")
        # ax.plot(Q_pinz, h_1_pinz, "v", label="h1")
        # ax.set_xlabel("Discharge Q [m3/s]")
        # ax.set_ylabel("Water height")
        # ax.grid(True, alpha=0.3)
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
        
        # fig, ax = plt.subplots(figsize=(7, 4))
        # ax.plot(Q_pinz, w_0_pinz, "o", label="w0")
        # ax.plot(Q_pinz, w_1_pinz, "v", label="w1")
        # ax.set_xlabel("Discharge Q [m3/s]")
        # ax.set_ylabel("Width [m]")
        # ax.grid(True, alpha=0.3)
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
        
        
        # #
        # mobilised = SedimSys.create_2d_zero_array()
        # for t in range(self.timescale):
        #     # Sum over provenances (axe 0) and sediment classes (axe 2)
        #     mobilised[t,:] = np.sum(SedimSys.Qbi_mob[t], axis = (0,2))
        
        # Vout_pinz = mobilised[:, 37]
        
        # fig, ax = plt.subplots(figsize=(7, 4))
        # ax.plot(Q_pinz, Vout_pinz, "o", label="V out")
        # ax.set_xlabel("Discharge Q [m3/s]")
        # ax.set_ylabel("V out [m3]")
        # ax.grid(True, alpha=0.3)
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
        
        # fig, ax = plt.subplots(figsize=(7, 4))
        # ax.plot(Q_pinz, tr_cap_1_pinz*self.ts_length, "o", label="tr_cap_1")
        # ax.plot(Q_pinz, tr_cap_2_pinz*self.ts_length, "v", label="tr_cap_2")
        # ax.set_xlabel("Discharge Q [m3/s]")
        # ax.set_ylabel("Tr cap total")
        # ax.grid(True, alpha=0.3)
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
        
        # fig, ax = plt.subplots(figsize=(7, 4))
        # ax.plot(tr_cap_2_pinz*self.ts_length, tr_cap_1_pinz*self.ts_length, "o", markersize = 3)
        # ax.plot([0,20000], [0, 20000], "--")
        # ax.set_xlabel("Tr cap total 2")
        # ax.set_ylabel("Tr cap total 1")
        # ax.grid(True, alpha=0.3)
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
        
        
        """End of the time loop"""


    def output_processing(self, Q):
        SedimSys = self.sedim_sys

        # Simulation parameters : dictionary to store the parameters used for the simulation
        # Volume out            : total volume [m^3] leaving the reach per time step, including passing cascades
        # Volume in             : total volume [m^3] entering the reach per time step, including passing cascades
        # Sediment budget       : budget between the total leaving the reach and entering the reach
        # Mobilised from reach  : total volume [m^3] mobilised from the reach per time step, excluding passing cascades
        # Deposited             : volume that deposits in the reach [m^3] (includes cascades finishing the time step + over-capacity cascades)
        # Volume outlet         : total volume of sediment leaving the network
        # D50 volume out        : D50 in the volume out
        # D50 active layer      : D50 in the active layer, used to compute the transport capacity
        # Direct connectivity   : volume connectivity per time step (axis 0). For a given cascade produce by a reach (axis 1), we see where it deposits (axis 2).
        # Transport capacity    : total transport capacity [m^3] per reach and per time step
        # Touch erosion max     : binary matrice indicating when the erosion maximum is reached --> ToDo

        # Create dictionary of the simulation parameters
        simulation_param = {'psi': SedimSys.psi, 'ts length': self.ts_length, 'update slope': self.update_slope,
                            'idx flow': self.indx_flo_depth, 'idx slope red': self.indx_slope_red,
                            'idx width calc': self.indx_width_calc, 'idx tr cap': self.indx_tr_cap,
                            'idx tr partition': self.indx_tr_partition, 'idx velocity': self.indx_velocity,
                            'idx vel partition': self.indx_vel_partition
                            }

        # Sum quantities
        mobilised = SedimSys.create_2d_zero_array()
        transported = SedimSys.create_2d_zero_array()
        mobilised_from_reach = SedimSys.create_2d_zero_array()
        direct_connectivity = np.zeros((self.timescale, self.n_reaches, self.n_reaches + 1)) # + 1 to consider sediment going to the outlet
        deposited = SedimSys.create_2d_zero_array()

        for t in range(self.timescale):
            # Sum over provenances (axe 0) and sediment classes (axe 2)
            mobilised[t,:] = np.sum(SedimSys.Qbi_mob[t], axis = (0,2))
            transported[t,:] = np.sum(SedimSys.Qbi_tr[t], axis = (0,2))
            mobilised_from_reach[t,:] = np.sum(SedimSys.Qbi_mob_from_r[t], axis = (0,2))
            # Sum direct connectivity over sediment classes (axe 2)
            direct_connectivity[t,:,:] = np.sum(SedimSys.direct_connectivity[t], axis = 2)
            # Deposited is the connectivity volumes summed by provenance (axe 0) and classes (axe 2) (excluding outlet)
            deposited[t,:] = np.sum(SedimSys.direct_connectivity[t][:, :-1, :], axis = (0,2))

        # Compute D50 mobilised (over sediment classes and provenance):
        D50_mob = SedimSys.create_2d_zero_array()
        for t in range(self.timescale):
            sum_by_provenance = np.sum(SedimSys.Qbi_mob[t], axis = 0)
            Fi_mob_t  = sum_by_provenance / mobilised[t, :][:, np.newaxis]
            D50_mob[t,:] = D_finder(Fi_mob_t, 50, SedimSys.psi)

        # Total sediment budget, summed over sediment classes (axe 2):
        volume_budget = np.sum(SedimSys.sediment_budget, axis = 2)

        # Total transport capacity, summed over sediment classes (axe 2):
        transport_capacity = np.sum(SedimSys.tr_cap, axis = 2)

        data_output = {'Simulation parameters': simulation_param,
                       'Volume out [m^3]': mobilised.astype(np.float32),
                       'Volume in [m^3]': transported.astype(np.float32),
                       'Sediment budget [m^3]': volume_budget.astype(np.float32),
                       'Mobilised from reach [m^3]': mobilised_from_reach.astype(np.float32),
                       'Deposited [m^3]': deposited.astype(np.float32),
                       'Volume outlet [m^3]': mobilised[:, SedimSys.outlet].astype(np.float32),
                       'D50 volume out [m]': D50_mob.astype(np.float32),
                       'D50 active layer [m]': SedimSys.D50_al.astype(np.float32),
                       'Direct connectivity [m^3]': direct_connectivity.astype(np.float32),
                       'Transport capacity [m^3]': transport_capacity.astype(np.float32),
                       'Fraction taken from AL': SedimSys.fr_mob_in_al,   # Active layer fraction metric
                       }

        if self.t_track == True:
            data_output['Eros_times'] = SedimSys.Qbi_tr_eros_times

        # Sum quantities by provenance
        mobilised_per_class = np.zeros((self.timescale, self.n_reaches, self.n_classes))
        transported_per_class = np.zeros((self.timescale, self.n_reaches, self.n_classes))
        deposited_per_class = np.zeros((self.timescale, self.n_reaches, self.n_classes))


        for t in range(self.timescale - 1):
            # Sum over provenances (axe 0)
            mobilised_per_class[t,:,:] = np.sum(SedimSys.Qbi_mob[t], axis = (0))
            transported_per_class[t,:,:] = np.sum(SedimSys.Qbi_tr[t], axis = (0))
            deposited_per_class[t,:,:] = np.sum(SedimSys.direct_connectivity[t][:, :-1, :], axis = (0)) # - 1 to exclude outlet

        # Complete matrices:
        extended_output = {'Volume out per grain sizes [m^3]': mobilised_per_class,
                           'Volume in per grain sizes [m^3]': transported_per_class,
                           'Deposited per grain sizes [m^3]': deposited_per_class,


                           'Qbi_mob [m^3]': SedimSys.Qbi_mob,
                           'Qbi_tr [m^3]': SedimSys.Qbi_tr,
                           'Qbi_mob_from_reach [m^3]': SedimSys.Qbi_mob_from_r,
                           'Qbi_dep [m^3]': SedimSys.Qbi_dep,
                           'Qout per class [m^3]': SedimSys.Q_out.astype(np.float32),
                           'Sediment budget per class [m^3]': SedimSys.sediment_budget.astype(np.float32),
                           'Tr_cap per class [m^3]': SedimSys.tr_cap.astype(np.float32),
                           'Node_el [m]': SedimSys.node_el,
                           'Fi_al': SedimSys.Fi_al.astype(np.float32),
                           'AL depth [m]': SedimSys.al_depth.astype(np.float32),
                           'Velocity section height [m]': SedimSys.vl_height.astype(np.float32),
                           'Velocities [m/s]': SedimSys.V_sed.astype(np.float32),
                           'Widths [m]': SedimSys.width.astype(np.float32),
                           'Slopes': SedimSys.slope.astype(np.float32),
                           'Mass balance [m^3]' : SedimSys.mass_balance.astype(np.float32)
                           }


        return data_output, extended_output
