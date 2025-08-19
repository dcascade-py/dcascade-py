"""
Created on Thu Sep  5 13:50:28 2024

@author: Diane Doolaeghe
"""


import os
import sys

# Add source (src) folder in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from pathlib import Path

import geopandas as gpd
import numpy as np

from GSD_curvefit import GSDcurvefit
from main import DCASCADE_main
from preprocessing import extract_Q, graph_preprocessing, read_network
from reach_data import ReachData

# get the root of the project
# for github actions
ROOT_DIR = Path(__file__).resolve().parent.parent


''' List of tests performed here:

        test_Vjosa_Engelund_all_new_options_false
        test_Vjosa_Wilcock_all_new_options_false
        (reproducing algorithme of the version 1 of dcascade)

        test_Vjosa_Engelund_all_new_options_true
        test_Vjosa_Wilcock_all_new_options_true

'''


#Pathes
path_river_network = ROOT_DIR / Path('inputs/input_trial/')
name_river_network = 'River_Network.shp'
filename_river_network = path_river_network / name_river_network

path_q = ROOT_DIR / Path('inputs/input_trial/')
name_q = 'Q_Vjosa.csv'
filename_q = path_q / name_q

# User defined parameters:
deposit_layer = 100000
eros_max = 1
al_depth = '2D90' # if None, it is 2D90
al_depth_method = 2
timescale = 20
ts_length = 60 * 60 * 24
sed_range = [-8, 5]
n_classes = 6
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

#  # External sediment
# Qbi_input = np.zeros((timescale, reach_data.n_reaches, n_classes))

# Input sediment load in deposit layer
deposit = reach_data.deposit * reach_data.length
Qbi_dep_in = np.zeros((reach_data.n_reaches, 1, n_classes))
for n in range(reach_data.n_reaches):
    Qbi_dep_in[n] = deposit[n] * Fi_r[n,:]



def test_Vjosa_Engelund_all_new_options_false():
    '''20 days are simulated.
    We use Engelund. With the "Bed Material Fraction" partitioning.
    '''
    # indexes
    indx_tr_cap = 3         # Engelund and Hansen
    indx_tr_partition = 2   # BMF

    # options in v2
    op1 = False
    op2 = False
    op3 = False

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length,
                                                 al_depth, indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 al_depth_method = al_depth_method,
                                                 eros_max = eros_max,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)




    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Volume out [m^3]'], axis = 0)
    expected_result = np.array([431292., 174049., 124598.,  16895.,  88802.,   7794.,  13352.])
    np.testing.assert_array_equal(test_result, expected_result)

    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Volume in [m^3]'], axis = 0)
    expected_result = np.array([     0., 454923., 194467.,  34665.,      0.,      0.,      0.])
    # the absolute tolerance is fixed to 1e6, because the expected results
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, atol = 1e06)

    # #----Test D50 active layer
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([0.00235723, 0.00115333, 0.00110481,
    #                             0.00050879, 0.002357, 0.00235716, 0.00235696])
    # # the relative tolerance is fixed to 1e-05, because the expected results
    # # were displayed by spyder, and have 6 significative numbers
    # np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)

    print('\n Tuto bene with Vjosa case test using Engelund formula, all option false \n')



def test_Vjosa_Wilcock_all_new_options_false():
    '''20 days are simulated.
    We use Wilcock and Crowes.
    '''
    # indexes
    indx_tr_cap = 2         # Wilcock
    indx_tr_partition = 4   # Shear stress p

    # options in v2
    op1 = False
    op2 = False
    op3 = False

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length,
                                                 al_depth, indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 al_depth_method = al_depth_method,
                                                 eros_max = eros_max,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)

    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Volume out [m^3]'], axis = 0)
    expected_result = np.array([2.245782e+06, 5.208630e+05, 2.840860e+05, 7.198300e+04,
                                8.078890e+05, 1.185800e+05, 1.840180e+05])
    np.testing.assert_array_equal(test_result, expected_result)

    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Volume in [m^3]'], axis = 0)
    expected_result = np.array([0.000000e+00, 1.041100e+05, 3.315064e+06, 3.255060e+05,
                                0.000000e+00, 0.000000e+00, 0.000000e+00])
    np.testing.assert_array_equal(test_result, expected_result)

    # #----Test D50 active layer
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([0.00235723, 0.00235714, 0.00228797, 0.00228537, 0.002357  ,
    #                             0.00235716, 0.00235696])
    # the relative tolerance is fixed to 1e-05, because the expected results
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)

    print('\n Tuto bene with Vjosa case test using Wilcock formula, all option false  \n')


def test_Vjosa_Engelund_all_true_no_tlag():
    '''20 days are simulated.
    We use Engelund. With the "Bed Material Fraction" partitioning.
    '''
    # indexes
    indx_tr_cap = 3         # Engelund and Hansen
    indx_tr_partition = 2   # BMF

    # options in v2
    op1 = True
    op2 = True
    op3 = False

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length,
                                                 al_depth, indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 al_depth_method = al_depth_method,
                                                 eros_max = eros_max,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)


    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Volume out [m^3]'], axis = 0)
    expected_result = np.array([431292., 177463., 138996.,  71630.,  88802.,   7794.,  13352.])
    np.testing.assert_array_equal(test_result, expected_result)

    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Volume in [m^3]'], axis = 0)
    expected_result = np.array([     0., 520094., 185257., 152348.,      0.,      0.,      0.])
    # the absolute tolerance is fixed to 1e6, because the expected results
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, atol = 1e06)

    # #----Test D50 active layer
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([0.00235723, 0.00115333, 0.00110481,
    #                             0.00050879, 0.002357, 0.00235716, 0.00235696])
    # # the relative tolerance is fixed to 1e-05, because the expected results
    # # were displayed by spyder, and have 6 significative numbers
    # np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)

    print('\n Tuto bene with Vjosa case test using Engelund formula, all option true, no time lag \n')



def test_Vjosa_Wilcock_all_true_no_tlag():
    '''20 days are simulated.
    We use Wilcock and Crowes.
    '''
    # indexes
    indx_tr_cap = 2         # Wilcock
    indx_tr_partition = 4   # Shear stress p

    # options in v2
    op1 = True
    op2 = True
    op3 = False

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length,
                                                 al_depth, indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 al_depth_method = al_depth_method,
                                                 eros_max = eros_max,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)

    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Volume out [m^3]'], axis = 0)
    # expected_result = np.array([2142257.,  497025.,  271124.,   68684.,  770800.,  113202.,  175644.])
    expected_result = np.array([2.245782e+06, 5.261160e+05, 2.874930e+05, 7.204900e+04,
                                8.078890e+05, 1.185800e+05, 1.840180e+05])

    np.testing.assert_array_equal(test_result, expected_result)

    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Volume in [m^3]'], axis = 0)
    expected_result = np.array([0.000000e+00, 3.053671e+06, 6.446960e+05, 4.715110e+05,
                                0.000000e+00, 0.000000e+00, 0.000000e+00])
    np.testing.assert_array_equal(test_result, expected_result)

    # #----Test D50 active layer
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([0.00235723, 0.00235714, 0.00228797, 0.00228537, 0.002357  ,
    #                             0.00235716, 0.00235696])
    # the relative tolerance is fixed to 1e-05, because the expected results
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)

    print('\n Tuto bene with Vjosa case test using Wilcock formula, all option true, no time lag  \n')




def test_Vjosa_Engelund_all_new_options_true():
    '''20 days are simulated.
    We use Engelund. With the "Bed Material Fraction" partitioning.
    '''
    # indexes
    indx_tr_cap = 3         # Engelund and Hansen
    indx_tr_partition = 2   # BMF

    # options in v2
    op1 = True
    op2 = True
    op3 = True

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length,
                                                 al_depth, indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 al_depth_method = al_depth_method,
                                                 eros_max = eros_max,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)


    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Volume out [m^3]'], axis = 0)
    expected_result = np.array([431292., 178924., 139355.,  68516.,  88802.,   7794.,  13352.])
    np.testing.assert_array_equal(test_result, expected_result)

    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Volume in [m^3]'], axis = 0)
    expected_result = np.array([     0., 520094., 186718., 152707.,      0.,      0.,      0.])
    # the absolute tolerance is fixed to 1e6, because the expected results
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, atol = 1e06)

    # #----Test D50 active layer
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([0.00235723, 0.00115333, 0.00110481, 0.00037359,
    #                             0.002357, 0.00235716, 0.00235696])

    # # the relative tolerance is fixed to 1e-05, because the expected results
    # # were displayed by spyder, and have 6 significative numbers
    # np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)

    print('\n Tuto bene with Vjosa case test using Engelund formula, all option true \n')



def test_Vjosa_Wilcock_all_new_options_true():
    '''20 days are simulated.
    We use Wilcock and Crowes.
    '''
    # indexes
    indx_tr_cap = 2         # Wilcock
    indx_tr_partition = 4   # Shear stress p

    # options in v2
    op1 = True
    op2 = True
    op3 = True

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, psi, timescale, ts_length,
                                                 al_depth, indx_tr_cap, indx_tr_partition, Qbi_dep_in,
                                                 al_depth_method = al_depth_method,
                                                 eros_max = eros_max,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)

    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Volume out [m^3]'], axis = 0)
    expected_result = np.array([2.245782e+06, 5.265410e+05, 2.874110e+05, 7.340700e+04,
                                8.078890e+05, 1.185800e+05, 1.840180e+05])

    np.testing.assert_array_equal(test_result, expected_result)

    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Volume in [m^3]'], axis = 0)
    expected_result = np.array([0.000000e+00, 3.053671e+06, 6.451210e+05, 4.714290e+05,
                                0.000000e+00, 0.000000e+00, 0.000000e+00])
    np.testing.assert_array_equal(test_result, expected_result)

    # #----Test D50 active layer
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([0.00235723, 0.00228442, 0.0022429 , 0.00223255,
    #                             0.002357, 0.00235716, 0.00235696])

    # # the relative tolerance is fixed to 1e-05, because the expected results
    # # were displayed by spyder, and have 6 significative numbers
    # np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)

    print('\n Tuto bene with Vjosa case test using Wilcock formula, all option true  \n')





if __name__ == "__main__":
    test_Vjosa_Engelund_all_new_options_false()
    test_Vjosa_Wilcock_all_new_options_false()
    test_Vjosa_Engelund_all_true_no_tlag()
    test_Vjosa_Wilcock_all_true_no_tlag()
    test_Vjosa_Engelund_all_new_options_true()
    test_Vjosa_Wilcock_all_new_options_true()

    print("All tests successfully run.")
