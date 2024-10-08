# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:50:28 2024

@author: Diane Doolaeghe
"""

import numpy as np
from DCASCADE_loop import DCASCADE_main, ReachData
from preprocessing import graph_preprocessing, extract_Q
from pathlib import Path
import geopandas as gpd
from GSD import GSDcurvefit




# Temporary test for us. The input files are not versionned, but must be asked to Diane D.. 

''' List of tests performed here:
    
        test_Po_Engelund_all_new_options_false
        test_Po_Wilcock_all_new_options_false
        (the expected results are the same as the version 1 of dcascade)
        
        test_Po_Engelund_all_new_options_true
        test_Po_Wilcock_all_new_options_true
        
'''

#Pathes
path_river_network = Path('../Po_case_16y/Inputs/06-shp_with_tributaries_updated/')
name_river_network = 'Po_river_network.shp'
filename_river_network = path_river_network / name_river_network

path_q = Path('../Po_case_16y/Inputs/')
name_q = 'Po_Qdaily_3y.csv'
filename_q = path_q / name_q


    
    
def test_Po_Engelund_all_new_options_false():
    '''150 days are simulated. 
    We use Engelund. With the "Bed Material Fraction" partitioning. 
    '''
    
    # User defined parameters:
    deposit_layer = 100000
    eros_max = 1
    update_slope = False
    timescale = 150 
    ts_length = 60 * 60 * 24
    sed_range = [-8, 3]  
    n_classes = 6  
    save_dep_layer = 'never'  
    roundpar = 0    
    
    # indexes
    indx_tr_cap = 3      # Engelund and Hansen
    indx_partition = 2   # BMF
    indx_flo_depth = 1   # Manning
    indx_slope_red = 1   # None
    indx_velocity = 1    # same velocity for all classes
    
    # reach data
    network = gpd.GeoDataFrame.from_file(filename_river_network)  # read shapefine from shp format
    reach_data = ReachData(network)
    reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
    sorted_indices = reach_data.sort_values_by(reach_data.from_n)
    Network = graph_preprocessing(reach_data)
    
    # Q file
    Q = extract_Q(filename_q)
    Q_new = np.zeros((Q.shape)) #reorganise Q file according to reachdata sorting
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
        
    consider_overtaking_sed_in_outputs = False
    compare_with_tr_cap = False
    time_lag_for_Vmob = False
    consider_passing_sed_in_tr_cap = False
      
    # run definition
    data_output, _ = DCASCADE_main(indx_tr_cap, indx_partition, indx_flo_depth,
                                                 indx_slope_red, indx_velocity, reach_data,
                                                 Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer,
                                                 ts_length, 
                                                 consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                                 time_lag_for_Vmob, consider_passing_sed_in_tr_cap)
        
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([
    ])
    
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([
    ])  
    
    # the absolute tolerance is fixed to 1e6, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, atol = 1e06)
    
    # D50 active layer
    test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    expected_result = np.array([
    ])     
    
    # the relative tolerance is fixed to 1e-05, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
    print('Tuto bene with Po case test using Engelund formula, all new options false \n')


def test_Po_Wilcock_all_new_options_false():
    '''150 days are simulated. 
    We use Wilcock and Crowes. 
    '''
    
    # User defined parameters:
    deposit_layer = 100000
    eros_max = 1
    update_slope = False
    timescale = 150 
    ts_length = 60 * 60 * 24
    sed_range = [-8, 3]  
    n_classes = 6  
    save_dep_layer = 'never'  
    roundpar = 0    
    
    # indexes
    indx_tr_cap = 2      # Wilkock and Crowe 2003
    indx_partition = 4   # Shear stress correction
    indx_flo_depth = 1   # Manning
    indx_slope_red = 1   # None
    indx_velocity = 1    # same velocity for all classes
    
    # reach data
    network = gpd.GeoDataFrame.from_file(filename_river_network)  # read shapefine from shp format
    reach_data = ReachData(network)
    reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
    sorted_indices = reach_data.sort_values_by(reach_data.from_n)
    Network = graph_preprocessing(reach_data)
    
    # Q file
    Q = extract_Q(filename_q)
    Q_new = np.zeros((Q.shape)) #reorganise Q file according to reachdata sorting
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
        
    consider_overtaking_sed_in_outputs = False
    compare_with_tr_cap = False
    time_lag_for_Vmob = False
    consider_passing_sed_in_tr_cap = False
      
    # run definition
    data_output, _ = DCASCADE_main(indx_tr_cap, indx_partition, indx_flo_depth,
                                                 indx_slope_red, indx_velocity, reach_data,
                                                 Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer,
                                                 ts_length, 
                                                 consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                                 time_lag_for_Vmob, consider_passing_sed_in_tr_cap)
        
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([2.37000e+02, 1.72400e+03, 3.52900e+03, 1.76000e+02, 5.70000e+01,
                                2.29700e+03, 7.02900e+03, 7.02710e+04, 1.94780e+04, 3.29320e+04,
                                1.37380e+04, 1.09200e+03, 1.20000e+03, 1.63500e+03, 1.01970e+05,
                                1.38890e+04, 8.11200e+03, 2.23610e+04, 5.10629e+05, 3.88415e+05,
                                3.09542e+05, 1.96795e+05, 1.93811e+05, 1.84628e+05, 8.68200e+04,
                                1.25639e+05, 1.66510e+05, 9.07330e+04, 2.37052e+05, 2.85358e+05,
                                2.54520e+05, 4.30881e+05, 4.74544e+05, 4.63795e+05, 4.47381e+05,
                                4.87490e+05, 4.27126e+05, 4.69889e+05, 4.42632e+05, 4.52981e+05,
                                5.15129e+05, 4.20883e+05, 2.74316e+05, 1.09821e+05, 0.00000e+00,
                                2.70000e+01, 9.00000e+00, 1.82000e+02, 1.20000e+01, 1.70000e+01,
                                8.19750e+05, 2.35931e+05, 5.10000e+01, 0.00000e+00, 2.17354e+05,
                                1.73790e+04, 7.36060e+04, 3.36340e+04, 1.16413e+05, 1.16705e+05,
                                2.86696e+05, 4.22740e+04, 2.63751e+05, 9.13910e+04])

    
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([0.00000e+00, 2.34000e+02, 1.52700e+03, 3.20900e+03, 1.36000e+02,
                                7.60000e+01, 2.08200e+03, 6.51900e+03, 6.71720e+04, 1.72200e+04,
                                3.12660e+04, 1.24180e+04, 1.14200e+03, 1.13800e+03, 1.55900e+03,
                                9.81350e+04, 1.20680e+04, 7.52500e+03, 8.33920e+05, 5.00971e+05,
                                3.79607e+05, 3.01346e+05, 4.20654e+05, 1.86528e+05, 3.89685e+05,
                                8.47510e+04, 1.39409e+05, 1.63265e+05, 1.59740e+05, 2.33250e+05,
                                3.13592e+05, 2.50415e+05, 5.38484e+05, 5.81474e+05, 4.57654e+05,
                                7.23058e+05, 4.80464e+05, 2.59700e+05, 9.22855e+05, 4.36114e+05,
                                4.46357e+05, 5.95880e+05, 4.14294e+05, 2.69913e+05, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00])
      
    
    np.testing.assert_array_equal(test_result, expected_result)
    
    # D50 active layer
    test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    expected_result = np.array([2.47770230e-02, 1.51655910e-02, 1.51655917e-02, 2.10485932e-02,
                                2.11767354e-02, 1.78638945e-02, 1.65645256e-02, 8.60037856e-03,
                                2.08146269e-02, 1.02222049e-02, 2.58848529e-02, 1.82418983e-02,
                                1.61567309e-02, 1.84498421e-02, 4.58531174e-03, 1.51034536e-02,
                                1.04169866e-02, 2.17482577e-03, 1.00555667e-04, 3.29986918e-04,
                                7.57055283e-04, 1.59568824e-03, 1.10826816e-03, 8.71742449e-04,
                                2.61711002e-04, 1.32537461e-03, 1.39984758e-03, 1.82083661e-03,
                                6.18053576e-04, 6.08001274e-04, 5.09345500e-04, 3.29934009e-04,
                                2.91136973e-04, 2.94741534e-04, 2.92469881e-04, 1.93144597e-04,
                                1.87469088e-04, 2.44587127e-04, 2.48516943e-04, 2.66209637e-04,
                                2.99611719e-04, 2.74685353e-04, 2.69453010e-04, 2.57981677e-04,
                                3.19105188e-02, 2.51239379e-02, 4.20755494e-02, 1.80465901e-02,
                                2.02931896e-02, 1.92805646e-02, 6.48709343e-05, 2.96288850e-05,
                                2.55396815e-02, 2.50317658e-02, 1.64401892e-04, 6.00410774e-03,
                                6.25982204e-04, 1.69362805e-04, 1.54117240e-04, 3.50491022e-04,
                                2.86550815e-05, 3.09206758e-04, 3.09270737e-04, 2.65200462e-04])
      
    
    # the relative tolerance is fixed to 1e-05, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
    print('Tuto bene with Po case test using Wilcock formula, all new options false \n')


    
def test_Po_Engelund_all_new_options_true():
    '''150 days are simulated. 
    We use Engelund. With the "Bed Material Fraction" partitioning. 
    '''
    
    # User defined parameters:
    deposit_layer = 100000
    eros_max = 1
    update_slope = False
    timescale = 150 
    ts_length = 60 * 60 * 24
    sed_range = [-8, 3]  
    n_classes = 6  
    save_dep_layer = 'never'  
    roundpar = 0    
    
    # indexes
    indx_tr_cap = 3      # Engelund and Hansen
    indx_partition = 2   # BMF
    indx_flo_depth = 1   # Manning
    indx_slope_red = 1   # None
    indx_velocity = 1    # same velocity for all classes
    
    # reach data
    network = gpd.GeoDataFrame.from_file(filename_river_network)  # read shapefine from shp format
    reach_data = ReachData(network)
    reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
    sorted_indices = reach_data.sort_values_by(reach_data.from_n)
    Network = graph_preprocessing(reach_data)
    
    # Q file
    Q = extract_Q(filename_q)
    Q_new = np.zeros((Q.shape)) #reorganise Q file according to reachdata sorting
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
        
    consider_overtaking_sed_in_outputs = True
    compare_with_tr_cap = True
    time_lag_for_Vmob = True
    consider_passing_sed_in_tr_cap = False
      
    # run definition
    data_output, _ = DCASCADE_main(indx_tr_cap, indx_partition, indx_flo_depth,
                                                 indx_slope_red, indx_velocity, reach_data,
                                                 Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer,
                                                 ts_length, 
                                                 consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                                 time_lag_for_Vmob, consider_passing_sed_in_tr_cap)
        
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([
    4143, 23013, 28324, 21336, 19139, 28582, 65500, 116546, 85683, 103723, 
    128964, 59363, 37300, 39233, 128757, 78295, 66838, 35180, 373678, 344570, 
    304846, 259603, 288643, 252062, 85717, 107642, 135033, 118593, 159366, 
    187795, 183278, 286858, 425877, 432414, 393206, 483894, 439349, 513183, 
    374388, 376259, 383250, 359341, 248869, 107551, 76, 1413, 3082, 2777, 2568, 
    749, 993010, 170564, 334, 1196, 272444, 25357, 40559, 11105, 183819, 
    101307, 264646, 12378, 283180, 75300
    ])
    
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([
    0, 4053, 22058, 27142, 20449, 19595, 27490, 66022, 112155, 82330, 99844, 
    124055, 59313, 35931, 40360, 124335, 75693, 63882, 1.01811e+06, 373678, 
    344073, 294266, 415146, 277377, 498763, 83684, 129894, 131934, 154355, 
    156107, 194931, 183278, 456368, 527184, 416467, 657852, 481017, 451727, 
    793631, 361496, 372552, 458550, 345642, 246843, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])  
    
    # the absolute tolerance is fixed to 1e6, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, atol = 1e06)
    
    # D50 active layer
    test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    expected_result = np.array([
    0.0247771, 0.0151707, 0.0151651, 0.0203769, 0.0204802, 0.0178016, 0.0165497, 0.00866687, 
    0.0228659, 0.0102464, 0.0292624, 0.0162394, 0.0160145, 0.01844, 0.00460604, 0.0162453, 
    0.0102547, 0.00203245, 0.000366432, 0.000317929, 0.000265434, 0.00102464, 0.000659992, 
    0.00031003, 8.26324e-06, 0.00177809, 0.00141684, 0.00265591, 0.000506883, 0.000530028, 
    0.000339422, 0.000296564, 0.000263349, 0.000254487, 0.000226963, 0.000117667, 
    8.72639e-05, 0.000147065, 0.00014583, 0.00016972, 0.000188809, 0.000159561, 
    9.92066e-05, 1.82689e-05, 0.0319107, 0.025124, 0.0420758, 0.0180466, 0.0202931, 
    0.0192818, 6.48827e-05, 2.94781e-05, 0.0255411, 0.0250317, 0.000164438, 0.00600404, 
    0.000621912, 0.000169004, 0.000155712, 0.000350787, 2.85048e-05, 0.000309135, 
    0.000309159, 0.000265391
    ])     
    
    # the relative tolerance is fixed to 1e-05, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
    print('Tuto bene with Po case test using Engelund formula, all new options true \n')



def test_Po_Wilcock_all_new_options_true():
    '''150 days are simulated. 
    We use Wilcock and Crowes. 
    '''
    
    # User defined parameters:
    deposit_layer = 100000
    eros_max = 1
    update_slope = False
    timescale = 150 
    ts_length = 60 * 60 * 24
    sed_range = [-8, 3]  
    n_classes = 6  
    save_dep_layer = 'never'  
    roundpar = 0    
    
    # indexes
    indx_tr_cap = 2      # Wilkock and Crowe 2003
    indx_partition = 4   # Shear stress correction
    indx_flo_depth = 1   # Manning
    indx_slope_red = 1   # None
    indx_velocity = 1    # same velocity for all classes
    
    # reach data
    network = gpd.GeoDataFrame.from_file(filename_river_network)  # read shapefine from shp format
    reach_data = ReachData(network)
    reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
    sorted_indices = reach_data.sort_values_by(reach_data.from_n)
    Network = graph_preprocessing(reach_data)
    
    # Q file
    Q = extract_Q(filename_q)
    Q_new = np.zeros((Q.shape)) #reorganise Q file according to reachdata sorting
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
        
    consider_overtaking_sed_in_outputs = True
    compare_with_tr_cap = True
    time_lag_for_Vmob = True
    consider_passing_sed_in_tr_cap = False
      
    # run definition
    data_output, _ = DCASCADE_main(indx_tr_cap, indx_partition, indx_flo_depth,
                                                 indx_slope_red, indx_velocity, reach_data,
                                                 Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer,
                                                 ts_length, 
                                                 consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                                 time_lag_for_Vmob, consider_passing_sed_in_tr_cap)
        
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([
    237, 1724, 3530, 125, 57, 2297, 7029, 70276, 2229, 32924, 8653, 910, 1199, 
    1629, 101972, 1026, 7645, 22168, 453871, 428312, 394589, 346667, 400069, 
    338623, 129929, 166783, 195508, 85651, 229037, 286311, 261528, 417549, 
    468240, 460154, 438501, 487340, 431126, 80437, 397295, 426999, 502431, 
    413632, 269141, 108127, 0, 27, 9, 182, 12, 17, 819785, 235950, 51, 0, 
    217288, 17379, 73732, 33657, 116156, 116680, 286823, 42284, 263772, 91398
    ])
    
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([
    0, 234, 1527, 3210, 98, 77, 2082, 6519, 67177, 1658, 31263, 7783, 997, 
    1137, 1553, 98136, 747, 7168, 833781, 443983, 428312, 386692, 570296, 
    391139, 542105, 127528, 180121, 191465, 154628, 225541, 314609, 261528, 
    530512, 582412, 453770, 713823, 480184, 466561, 344209, 391889, 420822, 
    593829, 407131, 264979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0
    ])  
    
    np.testing.assert_array_equal(test_result, expected_result)
    
    # D50 active layer
    test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    expected_result = np.array([
    0.0247771, 0.0151654, 0.0151651, 0.0211763, 0.0211762, 0.0178638, 0.0165646, 
    0.00860023, 0.0260577, 0.0102222, 0.027685, 0.0183552, 0.0161581, 0.0184497, 
    0.00458518, 0.0185011, 0.010417, 0.00217651, 0.000165847, 0.000149322, 
    0.000127542, 0.000215278, 0.000171118, 0.000136044, 0.000103275, 0.000994783, 
    0.00129788, 0.00209076, 0.000616477, 0.000603861, 0.000514625, 0.000355313, 
    0.000316236, 0.000308643, 0.000307156, 0.000190073, 0.000182749, 0, 
    0.000321646, 0.000311395, 0.000306277, 0.00028365, 0.000273104, 0.0002603, 
    0.0319107, 0.025124, 0.0420758, 0.0180466, 0.0202931, 0.0192818, 6.48827e-05, 
    2.94781e-05, 0.0255411, 0.0250317, 0.000164438, 0.00600405, 0.000621912, 
    0.000169004, 0.000155712, 0.000350787, 2.85048e-05, 0.000309135, 0.000309159, 
    0.000265391
    ])  
    
    # the relative tolerance is fixed to 1e-05, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
    print('Tuto bene with Po case test using Wilcock formula, all new options true\n')
    


if __name__ == "__main__":
    # test_Po_Engelund_all_new_options_false()
    test_Po_Wilcock_all_new_options_false()   
    # test_Po_Engelund_all_new_options_true()
    # test_Po_Wilcock_all_new_options_true()