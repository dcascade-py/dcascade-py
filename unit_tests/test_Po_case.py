"""
Created on Thu Sep  5 13:50:28 2024

@author: Diane Doolaeghe
"""

import sys, os
# Add source (src) folder in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))



from pathlib import Path
import geopandas as gpd
import numpy as np
import time

from main import DCASCADE_main
from GSD_curvefit import GSDcurvefit
from preprocessing import extract_Q, graph_preprocessing, read_network
from reach_data import ReachData

# get the current folder (where is this script)
# for github actions 
BASE_DIR = Path(__file__).resolve().parent



# Temporary test for us. The input files are not versionned, but must be asked to Diane D..

''' List of tests performed here:

        test_Po_Engelund_all_new_options_false
        test_Po_Wilcock_all_new_options_false
        (reproducing algorithme of the version 1 of dcascade)

        test_Po_Engelund_all_new_options_true
        test_Po_Wilcock_all_new_options_true

'''

#Pathes
path_river_network = BASE_DIR / Path('../inputs/Input_Po_untracked/shp/')
name_river_network = 'Po_river_network.shp'
filename_river_network = path_river_network / name_river_network

path_q = BASE_DIR / Path('../inputs/Input_Po_untracked/')
name_q = 'Po_Qdaily_3y.csv'
filename_q = path_q / name_q




def test_Po_Engelund_all_true_no_tlag():
    '''150 days are simulated.
    We use Engelund. With the "Bed Material Fraction" partitioning.
    '''

    # User defined parameters:
    deposit_layer = 100000
    eros_max = 1
    al_depth = '2D90'
    al_depth_method = 2
    update_slope = False
    timescale = 150
    ts_length = 60 * 60 * 24
    sed_range = [-8, 3]
    n_classes = 6
    save_dep_layer = 'never'
    roundpar = 0


    # reach data
    network = read_network(filename_river_network)
    reach_data = ReachData(network)
    reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
    sorted_indices = reach_data.sort_values_by(reach_data.from_n)
    Network = graph_preprocessing(reach_data)

    # Q file
    Q = extract_Q(filename_q)
    Q_new = np.zeros(Q.shape) #reorganise Q file according to reachdata sorting
    for i, idx in enumerate(sorted_indices):
        Q_new[:,i] = Q.iloc[:,idx]
    Q = Q_new

    # Sediment classes
    psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)
    dmi = 2**(-psi).reshape(-1,1)
    print(min(reach_data.D16) * 1000, ' must be greater than ', np.percentile(dmi, 10, method='midpoint'))
    print(max(reach_data.D84) * 1000, ' must be lower than ',  np.percentile(dmi, 90, method='midpoint'))
    Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi)

     # External sediment
    Qbi_input = np.zeros((timescale, reach_data.n_reaches, n_classes))

    # Input sediment load in deposit layer
    deposit = reach_data.deposit * reach_data.length
    Qbi_dep_in = np.zeros((reach_data.n_reaches, 1, n_classes))
    for n in range(reach_data.n_reaches):
        Qbi_dep_in[n] = deposit[n] * Fi_r[n,:]

    # indexes
    indx_tr_cap = 3         # Engelund and Hansen
    indx_tr_partition = 2   # BMF


    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length,
                                                 al_depth, indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 al_depth_method = al_depth_method,
                                                 eros_max = eros_max)



    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Volume out [m^3]'], axis = 0)
    expected_result = np.array([8.429000e+03, 4.732400e+04, 7.178500e+04, 6.271300e+04,
                                5.997500e+04, 8.072900e+04, 1.494300e+05, 3.735750e+05,
                                3.437590e+05, 3.120860e+05, 4.581190e+05, 3.086930e+05,
                                1.814020e+05, 1.761820e+05, 3.592480e+05, 2.814300e+05,
                                2.536910e+05, 1.326270e+05, 9.417650e+05, 7.853850e+05,
                                7.016850e+05, 6.002290e+05, 6.181120e+05, 5.468450e+05,
                                1.779030e+05, 3.221400e+05, 4.826300e+05, 4.266530e+05,
                                4.911440e+05, 6.046450e+05, 5.908870e+05, 7.808130e+05,
                                1.082178e+06, 1.103561e+06, 9.882430e+05, 1.111007e+06,
                                9.960880e+05, 1.352414e+06, 9.511350e+05, 9.502310e+05,
                                9.447720e+05, 8.346200e+05, 5.650000e+05, 2.114200e+05,
                                1.820000e+02, 2.921000e+03, 6.428000e+03, 5.665000e+03,
                                5.502000e+03, 1.557000e+03, 2.002643e+06, 3.527080e+05,
                                8.180000e+02, 2.567000e+03, 5.900740e+05, 5.211900e+04,
                                8.695200e+04, 2.357000e+04, 3.896610e+05, 2.141030e+05,
                                5.488730e+05, 2.539100e+04, 5.984700e+05, 1.611100e+05])


    np.testing.assert_array_equal(test_result, expected_result)

    # Test the total transported volume per reach
    test_result = np.sum(data_output['Volume in [m^3]'], axis = 0)
    expected_result = np.array([0.000000e+00, 8.429000e+03, 4.732400e+04, 7.178500e+04,
                               6.289500e+04, 6.289600e+04, 8.072900e+04, 1.558580e+05,
                               3.735750e+05, 3.437590e+05, 3.120860e+05, 4.581190e+05,
                               3.143580e+05, 1.814020e+05, 1.816840e+05, 3.592480e+05,
                               2.829870e+05, 2.536910e+05, 2.135270e+06, 9.417650e+05,
                               7.853850e+05, 7.025030e+05, 9.529370e+05, 6.206790e+05,
                               1.136919e+06, 1.779030e+05, 3.742590e+05, 4.826300e+05,
                               5.136050e+05, 4.911440e+05, 6.282150e+05, 5.908870e+05,
                               1.170474e+06, 1.296281e+06, 1.103561e+06, 1.537116e+06,
                               1.111007e+06, 1.021479e+06, 1.950884e+06, 9.511350e+05,
                               9.502310e+05, 1.105882e+06, 8.346200e+05, 5.650000e+05,
                               0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                               0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                               0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                               0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                               0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00])

    # the absolute tolerance is fixed to 1e6, because the expected results
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, atol = 1e06)

    # # D50 active layer: DD: TO DO
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([2.47770230e-02, 1.51755507e-02, 1.51655911e-02, 2.02552886e-02,
    #        2.03466590e-02, 1.77876247e-02, 1.65305320e-02, 8.71357324e-03,
    #        2.22509619e-02, 1.02616880e-02, 2.92620602e-02, 1.42418506e-02,
    #        1.58089862e-02, 1.83735086e-02, 4.62385848e-03, 1.58265987e-02,
    #        1.01485931e-02, 2.08804488e-03, 8.22115101e-05, 6.77232744e-04,
    #        9.22278655e-04, 2.08774248e-03, 1.55653800e-03, 8.36720622e-04,
    #        5.65198322e-05, 1.84719781e-03, 1.46731157e-03, 2.54021205e-03,
    #        4.89826506e-04, 5.33831955e-04, 4.38677070e-04, 2.74783902e-04,
    #        2.48700526e-04, 2.39310130e-04, 2.21651510e-04, 1.24331445e-04,
    #        1.43998111e-04, 2.23117944e-04, 2.04964090e-04, 2.62114246e-04,
    #        2.96271429e-04, 2.61212012e-04, 2.48002917e-04, 2.41762730e-04,
    #        3.19105188e-02, 2.51241504e-02, 4.20755494e-02, 1.80465901e-02,
    #        2.02931896e-02, 1.92805646e-02, 6.48783109e-05, 2.96288850e-05,
    #        2.55396815e-02, 2.50317658e-02, 1.64401892e-04, 6.00410774e-03,
    #        6.25982204e-04, 1.69362805e-04, 1.54117240e-04, 3.50491022e-04,
    #        2.86550815e-05, 3.09206758e-04, 3.09270737e-04, 2.65200462e-04
    # ])

    # # the relative tolerance is fixed to 1e-05, because the expected results
    # # were displayed by spyder, and have 6 significative numbers
    # np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)


def test_Po_Wilcock_all_true_no_tlag():
    '''150 days are simulated.
    We use Wilcock and Crowes.
    '''

    # User defined parameters:
    deposit_layer = 100000
    eros_max = 1
    al_depth = '2D90'
    al_depth_method = 2
    update_slope = False
    timescale = 150
    ts_length = 60 * 60 * 24
    sed_range = [-8, 3]
    n_classes = 6
    save_dep_layer = 'never'
    roundpar = 0


    # reach data
    network = read_network(filename_river_network)
    reach_data = ReachData(network)
    reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
    sorted_indices = reach_data.sort_values_by(reach_data.from_n)
    Network = graph_preprocessing(reach_data)

    # Q file
    Q = extract_Q(filename_q)
    Q_new = np.zeros(Q.shape) #reorganise Q file according to reachdata sorting
    for i, idx in enumerate(sorted_indices):
        Q_new[:,i] = Q.iloc[:,idx]
    Q = Q_new

    # Sediment classes
    psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)
    dmi = 2**(-psi).reshape(-1,1)
    print(min(reach_data.D16) * 1000, ' must be greater than ', np.percentile(dmi, 10, method='midpoint'))
    print(max(reach_data.D84) * 1000, ' must be lower than ',  np.percentile(dmi, 90, method='midpoint'))
    Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi)

     # External sediment
    Qbi_input = np.zeros((timescale, reach_data.n_reaches, n_classes))

    # Input sediment load in deposit layer
    deposit = reach_data.deposit * reach_data.length
    Qbi_dep_in = np.zeros((reach_data.n_reaches, 1, n_classes))
    for n in range(reach_data.n_reaches):
        Qbi_dep_in[n] = deposit[n] * Fi_r[n,:]

    # indexes
    indx_tr_cap = 2         # Wilcock
    indx_tr_partition = 4   # Shear stress p.

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length,
                                                 al_depth, indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 al_depth_method = al_depth_method,
                                                 eros_max = eros_max)


    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Volume out [m^3]'], axis = 0)
    expected_result = np.array([2.38000e+02, 1.89100e+03, 3.85900e+03, 2.36000e+02, 6.90000e+01,
                                2.48100e+03, 7.53900e+03, 7.41550e+04, 2.40350e+04, 3.49320e+04,
                                1.58310e+04, 1.30900e+03, 1.26800e+03, 1.73600e+03, 1.06391e+05,
                                1.74280e+04, 9.12800e+03, 2.33800e+04, 4.89967e+05, 3.73292e+05,
                                3.01059e+05, 2.00861e+05, 1.95540e+05, 1.89826e+05, 9.02780e+04,
                                1.32382e+05, 1.74785e+05, 9.62880e+04, 2.42294e+05, 2.95985e+05,
                                2.66595e+05, 4.40400e+05, 4.86735e+05, 4.75693e+05, 4.57098e+05,
                                4.93666e+05, 4.33021e+05, 5.12168e+05, 4.59035e+05, 4.67268e+05,
                                5.30005e+05, 4.31876e+05, 2.81389e+05, 1.12986e+05, 0.00000e+00,
                                2.70000e+01, 9.00000e+00, 1.82000e+02, 1.70000e+01, 1.70000e+01,
                                7.86909e+05, 2.20804e+05, 7.60000e+01, 0.00000e+00, 2.25350e+05,
                                1.80010e+04, 7.68840e+04, 3.48720e+04, 1.19919e+05, 1.20317e+05,
                                2.65827e+05, 4.30740e+04, 2.71958e+05, 9.45050e+04])


    np.testing.assert_array_equal(test_result, expected_result)

    # Test the total transported volume per reach
    test_result = np.sum(data_output['Volume in [m^3]'], axis = 0)
    expected_result = np.array([0.00000e+00, 2.38000e+02, 1.89100e+03, 3.85900e+03, 2.36000e+02,
                                9.60000e+01, 2.48100e+03, 7.54800e+03, 7.41550e+04, 2.40350e+04,
                                3.49320e+04, 1.58310e+04, 1.49100e+03, 1.26800e+03, 1.75300e+03,
                                1.06391e+05, 1.74450e+04, 9.12800e+03, 8.10289e+05, 4.89967e+05,
                                3.73292e+05, 3.01135e+05, 4.21665e+05, 1.95540e+05, 4.15176e+05,
                                9.02780e+04, 1.50383e+05, 1.74785e+05, 1.73172e+05, 2.42294e+05,
                                3.30857e+05, 2.66595e+05, 5.60319e+05, 6.07052e+05, 4.75693e+05,
                                7.22925e+05, 4.93666e+05, 4.76095e+05, 7.84126e+05, 4.59035e+05,
                                4.67268e+05, 6.24510e+05, 4.31876e+05, 2.81389e+05, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00])


    np.testing.assert_array_equal(test_result, expected_result)

    # # D50 active layer
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([2.47770230e-02, 1.51655910e-02, 1.51655917e-02, 2.10485932e-02,
    #                             2.11767354e-02, 1.78638945e-02, 1.65645256e-02, 8.60037856e-03,
    #                             2.08146269e-02, 1.02222049e-02, 2.58848529e-02, 1.82418983e-02,
    #                             1.61567309e-02, 1.84498421e-02, 4.58531174e-03, 1.51034536e-02,
    #                             1.04169866e-02, 2.17482577e-03, 1.00555667e-04, 3.29986918e-04,
    #                             7.57055283e-04, 1.59568824e-03, 1.10826816e-03, 8.71742449e-04,
    #                             2.61711002e-04, 1.32537461e-03, 1.39984758e-03, 1.82083661e-03,
    #                             6.18053576e-04, 6.08001274e-04, 5.09345500e-04, 3.29934009e-04,
    #                             2.91136973e-04, 2.94741534e-04, 2.92469881e-04, 1.93144597e-04,
    #                             1.87469088e-04, 2.44587127e-04, 2.48516943e-04, 2.66209637e-04,
    #                             2.99611719e-04, 2.74685353e-04, 2.69453010e-04, 2.57981677e-04,
    #                             3.19105188e-02, 2.51239379e-02, 4.20755494e-02, 1.80465901e-02,
    #                             2.02931896e-02, 1.92805646e-02, 6.48709343e-05, 2.96288850e-05,
    #                             2.55396815e-02, 2.50317658e-02, 1.64401892e-04, 6.00410774e-03,
    #                             6.25982204e-04, 1.69362805e-04, 1.54117240e-04, 3.50491022e-04,
    #                             2.86550815e-05, 3.09206758e-04, 3.09270737e-04, 2.65200462e-04])


    # # the relative tolerance is fixed to 1e-05, because the expected results
    # # were displayed by spyder, and have 6 significative numbers
    # np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)



if __name__ == "__main__":
    test_Po_Engelund_all_true_no_tlag()
    test_Po_Wilcock_all_true_no_tlag()

    
    print("All tests successfully run.")
