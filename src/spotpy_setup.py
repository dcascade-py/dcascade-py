import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add source (src) folder in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from GSD_curvefit import GSDcurvefit
from main import DCASCADE_main
from plot_function import dynamic_plot
from preprocessing import (check_sediment_sizes, extract_Q,
                        graph_preprocessing, read_network)
from reach_data import ReachData



import spotpy

class spotpy_setup(object):
    def __init__(self, dim=1):

        # This should be your observed data, e.g., sediment transport in m3/s, for the same time period as your simulation. Here we just use random data for demonstration purposes.
        #---------------------Path to the pickle output
        path = "../cascade_results/"
        name_simu = 'Vjosa_test'
        data_output = pd.read_pickle(open( path + name_simu + '.p' , "rb"))
        self.observations = data_output['Volume out [m^3]'][:, 1] 

        self.dim = dim
        self.params = []
        for i in range(self.dim):
            parname = "base_" + str(i)  # Used for D16, D50, D84
            self.params.append(spotpy.parameter.Uniform(parname, 0, 0.01)) # A step and a guess can also be given.
            parname = "delta1_" + str(i)  # Used for D50, 84
            self.params.append(spotpy.parameter.Uniform(parname, 0, 0.01)) # A step and a guess can also be given.
            parname = "delta2_" + str(i)  # Used for D84
            self.params.append(spotpy.parameter.Uniform(parname, 0, 0.01)) # A step and a guess can also be given.
            # The D16 < D50 < D84 constraint is implemented in the simulation function

    def parameters(self):           
        return spotpy.parameter.generate(self.params)

    def simulation(self, params):
        n = len(params)
        base = params[0:n:3]
        delta1 = params[1:n:3]
        delta2 = params[2:n:3]

        gsd16 = base
        gsd50 = base + delta1
        gsd84 = base + delta1 + delta2

        print("base, delta1, delta2:", base, delta1, delta2)
        print("gsd16, gsd50, gsd84:", gsd16, gsd50, gsd84)

        simulation = self.main_function(gsd16, gsd50, gsd84) # This is the function that runs the model and returns the simulated sediment transport (or any other variable you want to compare to observations)
        return simulation

    def evaluation(self):
        observations = self.observations
        return observations

    def objectivefunction(self,simulation,evaluation):
        objectivefunction= spotpy.objectivefunctions.rmse(evaluation, simulation)
        return objectivefunction
    
    def main_function(self, gsd16, gsd50, gsd84):
        """
        See the 00-DACASCADE_user_script_example_Vjosa.py file.
        """

        #--------------------1) Pathes

        #---River shape files
        path_river_network = Path('../inputs/input_trial/')
        # Reach data file (shp, but can also be a csv)
        name_river_network = 'River_Network.shp'
        filename_river_network = path_river_network / name_river_network

        #---Discharge files
        path_q = Path('../inputs/input_trial/')
        # csv file that specifies the water flows in m3/s as a (nxm) matrix, where n = number of time steps; m = number of reaches (equal to the one specified in the river network)
        name_q = 'Q_Vjosa.csv'
        filename_q = path_q / name_q

        #---Nome of the output
        name_output = 'Vjosa_test'

        #-------------------2) User-defined main parameters of the simulation

        #---Sediment classes definition
        # defines the sediment sizes considered in the simulation
        #(must be compatible with D16, D50, D84 defined for the reach - i.e. max sed class cannot be lower than D16)
        sed_range = [-8, 5]  # range of sediment sizes - in Krumbein phi (φ) scale (classes from coarse to fine – e.g., -9.5, -8.5, -7.5 … 5.5, 6.5).
        n_classes = 6        # number of classes

        #---Timescale
        timescale = 20 # days
        ts_length = 60 * 60 * 24 # length of timestep in seconds - 60*60*24 = daily; 60*60 = hourly

        #---Transport capacity formula and partitioning
        indx_tr_cap = 2                 # 2: Wilkock and Crowe;
                                        # 3: Engelund and Hansen;
                                        # 6: Ackers and White;

        indx_tr_partition = 4           # 1: Direct calculation summing fractionnal load;
                                        # 2: BMF: "Bed Material" Fraction weighting of fractionnal loads;
                                        # 3: Molinas rates: weighting on total load
                                        # 4: Shear stress correction (only for Formula already partitionned, e.g., W&C)

        #---Initial layer sizes
        deposit_layer = 100000      # Initial deposit layer [m]. Warning: will overwrite the deposit column in the reach_data file
        al_depth = 0.3              # Active layer depth [m] (Possibilities: '2D90', or any fixed value)

        #---Storing Deposit layer
        save_dep_layer = 'never' # options: 'yearly', 'always', 'never'.  Choose when to save the deposit layer matrix


        #---Option to save extended outputs or not
        # Note: saving the extended outputs can require memory, but allow you to access more outputs (see README file)
        save_extended = True


        ################ PREPROCESSING ###############

        # Read the network
        reach_data_df = read_network(filename_river_network)
        reach_data = ReachData(reach_data_df)

        # Define the initial deposit layer per each reach in [m3/m]
        reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)

        # Read/define the water discharge
        Q = extract_Q(filename_q)

        # Sort reach_data according to the from_n, and organise the Q file accordingly
        sorted_indices = reach_data.sort_values_by(reach_data.from_n)
        Q_new = np.zeros(Q.shape)
        for i, idx in enumerate(sorted_indices):
            Q_new[:,i] = Q.iloc[:,idx]
        Q = Q_new

        # Extract network properties
        network = graph_preprocessing(reach_data)

        # Sediment classes defined in Krumbein phi (φ) scale
        psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)

        # Sediment classes in mm
        dmi = 2**(-psi).reshape(-1,1)
        check_sediment_sizes(reach_data, dmi)


        # Define input sediment load in the deposit layer
        deposit = reach_data.deposit * reach_data.length * reach_data.wac

        # Define initial sediment fractions per class in each reaches, using a Rosin distribution
        N = len(gsd16)
        reach_data.D16[:N] = gsd16
        reach_data.D50[:N] = gsd50
        reach_data.D84[:N] = gsd84

        Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi)

        # Initialise deposit layer
        Qbi_dep_in = np.zeros((reach_data.n_reaches, 1, n_classes))
        for n in range(reach_data.n_reaches):
            Qbi_dep_in[n] = deposit[n] * Fi_r[n,:]


        # Prepare optionnal paramaters (possibly not given by the user) for calling the DCASCADE_main function
        kwargs = {}

        if 'eros_max' in globals():
            kwargs['eros_max'] = globals().get('eros_max')

        if 'al_depth_method' in globals():
            kwargs['al_depth_method'] = globals().get('al_depth_method')

        if 'vel_height' in globals():
            kwargs['vel_height'] = globals().get('vel_height')

        if 'indx_flo_depth' in globals():
            kwargs['indx_flo_depth'] = globals().get('indx_flo_depth')

        if 'indx_velocity' in globals():
            kwargs['indx_velocity'] = globals().get('indx_velocity')

        if 'indx_vel_partition' in globals():
            kwargs['indx_vel_partition'] = globals().get('indx_vel_partition')

        if 'indx_slope_red' in globals():
            kwargs['indx_slope_red'] = globals().get('indx_slope_red')

        if 'indx_width_calc' in globals():
            kwargs['indx_width_calc'] = globals().get('indx_width_calc')

        if 'update_slope' in globals():
            kwargs['update_slope'] = globals().get('update_slope')

        if 'roundpar' in globals():
            kwargs['roundpar'] = globals().get('roundpar')

        if 'save_dep_layer' in globals():
            kwargs['save_dep_layer'] = globals().get('save_dep_layer')

        if 'external_inputs' in globals():
            kwargs['external_inputs'] = globals().get('external_inputs')

        if 'force_pass_external_inputs' in globals():
            kwargs['force_pass_external_inputs'] = globals().get('force_pass_external_inputs')

        if 'dam_trap_efficiency' in globals():
            kwargs['dam_trap_efficiency'] = globals().get('dam_trap_efficiency')


        ################ CALL MAIN ###############
        data_output, extended_output = DCASCADE_main(reach_data, network, Q, psi, timescale, ts_length, al_depth,
                                                    indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                    **kwargs)

        ################ SAVE OUTPUTS ###############
        import pickle

        # path_results = Path("../cascade_results/")
        # if not os.path.exists(path_results):
        #     os.makedirs(path_results)

        # name_file = path_results / Path(str(name_output) + '.p')
        # pickle.dump(data_output, open(name_file , "wb"))  # save it into a file named save.p

        # if save_extended:
        #     name_file_ext = path_results / Path(str(name_output) + '_ext.p')
        #     pickle.dump(extended_output , open(name_file_ext , "wb"))  # save it into a file named save.p

        return data_output['Volume out [m^3]'][:,1]

