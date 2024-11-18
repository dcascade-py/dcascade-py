# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:50:28 2024

@author: Diane Doolaeghe
"""

import numpy as np
from DCASCADE_main_script import DCASCADE_main, ReachData
from preprocessing import graph_preprocessing, extract_Q
from pathlib import Path
import geopandas as gpd
from GSD import GSDcurvefit
import time



# Temporary test for us. The input files are not versionned, but must be asked to Diane D.. 

''' List of tests performed here:
    
        test_Po_Engelund_all_new_options_false
        test_Po_Wilcock_all_new_options_false
        (reproducing algorithme of the version 1 of dcascade)
        
        test_Po_Engelund_all_new_options_true
        test_Po_Wilcock_all_new_options_true
        
'''

#Pathes
path_river_network = Path('Input_Po_untracked/shp/')
name_river_network = 'Po_river_network.shp'
filename_river_network = path_river_network / name_river_network

path_q = Path('Input_Po_untracked/')
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
        
    # indexes
    indx_tr_cap = 3         # Engelund and Hansen
    indx_tr_partition = 2   # BMF
    indx_flo_depth = 1      # Manning
    
    # Options for the cascade algorithm (by default, they are all True):        
    # If all these options are False, we are reproducing the algorithme of 
    # the old version. Which means that cascades are all passing if the time step 
    # is not finished for them (= based on their velocities) + overpassing cascades are 
    # not considered in the mobilised volume nor transported

    # Option 1: If True, we consider ovepassing sediment in the output (Qbimob and Qbitr).
    # But this does not change the way sediment move.
    op1 = False

    # Option 2: If True, we now include present cascades from upstream + reach material
    # in the transport capacity calculation, to check if they should pass or not. 
    op2 = False

    # Option 3: If True, we consider a time lag between the beginning of the time step,
    # and the arrival of the first cascade to the ToN of the reach, 
    # during which we are able to mobilise from the reach itself
    op3 = False

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer, ts_length,
                                                 indx_tr_cap, indx_tr_partition, indx_flo_depth,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)
    
        
    
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([8.291000e+03, 4.564100e+04, 6.829700e+04, 5.932000e+04,
                                5.642600e+04, 7.451300e+04, 1.430240e+05, 3.254750e+05,
                                2.461390e+05, 2.819290e+05, 4.249720e+05, 2.852150e+05,
                                1.681140e+05, 1.638180e+05, 3.379010e+05, 2.653860e+05,
                                2.410600e+05, 1.233250e+05, 9.160860e+05, 7.596290e+05,
                                6.287150e+05, 5.766610e+05, 5.904780e+05, 4.730510e+05,
                                1.717550e+05, 1.994500e+05, 3.236930e+05, 3.081880e+05,
                                3.774030e+05, 3.786930e+05, 4.031020e+05, 5.382670e+05,
                                7.826430e+05, 6.997280e+05, 7.257940e+05, 9.429790e+05,
                                8.386380e+05, 4.317970e+05, 7.128180e+05, 7.924760e+05,
                                7.847530e+05, 7.148100e+05, 4.936270e+05, 1.942940e+05,
                                1.760000e+02, 2.874000e+03, 6.189000e+03, 5.594000e+03,
                                5.201000e+03, 1.551000e+03, 1.985705e+06, 3.411910e+05,
                                6.870000e+02, 2.504000e+03, 5.450060e+05, 5.068100e+04,
                                7.990200e+04, 2.219700e+04, 3.698170e+05, 2.036160e+05,
                                5.288710e+05, 2.474300e+04, 5.667540e+05, 1.511300e+05])
    

    
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([      0.,    8111.,   43773.,   64451.,   55923.,   56238.,
                                70750.,  129194.,  270683.,  255217.,  270604.,  400946.,
                                275245.,  158159.,  159241.,  321726.,  255577.,  230816.,
                                2084914.,  893739.,  684576.,  612323.,  885697.,  506277.,
                                975111.,  122746.,  247649.,  336074.,  377500.,  286744.,
                                422606.,  370780.,  833503.,  675944.,  813141., 1255814.,
                                903078.,  158613., 1545132.,  737917.,  776087.,  910439.,
                                698993.,  483032.,       0.,       0.,       0.,       0.,
                                0.,       0.,       0.,       0.,       0.,       0.,
                                0.,       0.,       0.,       0.,       0.,       0.,
                                0.,       0.,       0.,       0.])  
    
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
        
    # indexes
    indx_tr_cap = 2         # Wilcock
    indx_tr_partition = 4   # Shear stress p.
    indx_flo_depth = 1      # Manning
               
    # Options for the cascade algorithm (by default, they are all True):        
    # If all these options are False, we are reproducing the algorithme of 
    # the old version. Which means that cascades are all passing if the time step 
    # is not finished for them (= based on their velocities) + overpassing cascades are 
    # not considered in the mobilised volume nor transported

    # Option 1: If True, we consider ovepassing sediment in the output (Qbimob and Qbitr).
    # But this does not change the way sediment move.
    op1 = False

    # Option 2: If True, we now include present cascades from upstream + reach material
    # in the transport capacity calculation, to check if they should pass or not. 
    op2 = False

    # Option 3: If True, we consider a time lag between the beginning of the time step,
    # and the arrival of the first cascade to the ToN of the reach, 
    # during which we are able to mobilise from the reach itself
    op3 = False

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer, ts_length,
                                                 indx_tr_cap, indx_tr_partition, indx_flo_depth,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)
    
        
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([2.37000e+02, 1.72400e+03, 3.57100e+03, 1.98000e+02, 6.20000e+01,
                                2.29800e+03, 7.06000e+03, 7.10590e+04, 2.14120e+04, 3.32050e+04,
                                1.43800e+04, 1.15600e+03, 1.20500e+03, 1.64500e+03, 1.02042e+05,
                                1.50500e+04, 8.33600e+03, 2.23900e+04, 5.13687e+05, 3.94466e+05,
                                3.19882e+05, 2.12873e+05, 2.06835e+05, 1.95621e+05, 9.06180e+04,
                                1.31266e+05, 1.72506e+05, 9.39500e+04, 2.37467e+05, 2.90051e+05,
                                2.60656e+05, 4.32304e+05, 4.77223e+05, 4.66634e+05, 4.48657e+05,
                                4.88557e+05, 4.28901e+05, 4.73884e+05, 4.47475e+05, 4.56153e+05,
                                5.18531e+05, 4.23279e+05, 2.76209e+05, 1.11022e+05, 0.00000e+00,
                                2.70000e+01, 9.00000e+00, 1.82000e+02, 1.20000e+01, 1.70000e+01,
                                8.19750e+05, 2.35931e+05, 5.10000e+01, 0.00000e+00, 2.17354e+05,
                                1.73790e+04, 7.36060e+04, 3.36340e+04, 1.16413e+05, 1.16705e+05,
                                2.86696e+05, 4.22740e+04, 2.63751e+05, 9.13910e+04])

    
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([0.00000e+00, 2.34000e+02, 1.52700e+03, 3.24200e+03, 1.56000e+02,
                                8.10000e+01, 2.08300e+03, 6.54500e+03, 6.78770e+04, 1.88180e+04,
                                3.14820e+04, 1.29750e+04, 1.19500e+03, 1.14300e+03, 1.56900e+03,
                                9.82030e+04, 1.31110e+04, 7.71300e+03, 8.33946e+05, 5.04026e+05,
                                3.85617e+05, 3.11640e+05, 4.36617e+05, 1.99140e+05, 4.00120e+05,
                                8.84200e+04, 1.44876e+05, 1.69047e+05, 1.62849e+05, 2.33650e+05,
                                3.18214e+05, 2.56344e+05, 5.39792e+05, 5.83920e+05, 4.60174e+05,
                                7.24180e+05, 4.81575e+05, 2.48436e+05, 9.26711e+05, 4.40925e+05,
                                4.49479e+05, 5.99250e+05, 4.16695e+05, 2.71813e+05, 0.00000e+00,
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
        
    # indexes
    indx_tr_cap = 3         # Engelund and Hansen
    indx_tr_partition = 2   # BMF
    indx_flo_depth = 1      # Manning
    
    # Run definition
    start = time.time()
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer, ts_length, 
                                                 indx_tr_cap , indx_tr_partition, indx_flo_depth)
    
    end = time.time()
    print(end - start)
        
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([8.291000e+03, 4.564100e+04, 6.829700e+04, 5.932000e+04,
                                5.869100e+04, 7.451300e+04, 1.430240e+05, 3.412660e+05,
                                2.944890e+05, 2.899300e+05, 4.347970e+05, 2.912410e+05,
                                1.689230e+05, 1.639760e+05, 3.380210e+05, 2.653020e+05,
                                2.409680e+05, 1.233040e+05, 9.160850e+05, 7.596280e+05,
                                6.767570e+05, 5.745430e+05, 5.850670e+05, 5.163570e+05,
                                1.725520e+05, 2.304570e+05, 3.034530e+05, 2.712200e+05,
                                3.467360e+05, 3.913880e+05, 3.769960e+05, 5.257770e+05,
                                7.774330e+05, 7.926700e+05, 7.171210e+05, 9.245910e+05,
                                8.346850e+05, 9.135750e+05, 7.575040e+05, 7.536170e+05,
                                7.528690e+05, 6.879580e+05, 4.808960e+05, 1.923930e+05,
                                1.760000e+02, 2.874000e+03, 6.189000e+03, 5.594000e+03,
                                5.201000e+03, 1.551000e+03, 1.985705e+06, 3.411910e+05,
                                6.870000e+02, 2.504000e+03, 5.450060e+05, 5.068100e+04,
                                7.990200e+04, 2.219700e+04, 3.698170e+05, 2.036160e+05,
                                5.288710e+05, 2.474300e+04, 5.667540e+05, 1.511300e+05])
    
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([      0.,    8291.,   45641.,   68297.,   59496.,   61565.,
                                74513.,  149213.,  341266.,  294489.,  289930.,  434797.,
                                296835.,  168923.,  169177.,  338021.,  266853.,  240968.,
                                2109009.,  916085.,  759628.,  677444.,  915734.,  587571.,
                                1061363.,  172552.,  281138.,  303453.,  351122.,  346736.,
                                413585.,  376996.,  895594.,  981049.,  792670., 1245992.,
                                924591.,  859428., 1480329.,  757504.,  753617.,  903999.,
                                687958.,  480896.,       0.,       0.,       0.,       0.,
                                0.,       0.,       0.,       0.,       0.,       0.,
                                0.,       0.,       0.,       0.,       0.,       0.,
                                0.,       0.,       0.,       0.])  
    
    # the absolute tolerance is fixed to 1e6, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, atol = 1e06)
    
    # # D50 active layer
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([
    # 0.0247771, 0.0151707, 0.0151651, 0.0203769, 0.0204802, 0.0178016, 0.0165497, 0.00866687, 
    # 0.0228659, 0.0102464, 0.0292624, 0.0162394, 0.0160145, 0.01844, 0.00460604, 0.0162453, 
    # 0.0102547, 0.00203245, 0.000366432, 0.000317929, 0.000265434, 0.00102464, 0.000659992, 
    # 0.00031003, 8.26324e-06, 0.00177809, 0.00141684, 0.00265591, 0.000506883, 0.000530028, 
    # 0.000339422, 0.000296564, 0.000263349, 0.000254487, 0.000226963, 0.000117667, 
    # 8.72639e-05, 0.000147065, 0.00014583, 0.00016972, 0.000188809, 0.000159561, 
    # 9.92066e-05, 1.82689e-05, 0.0319107, 0.025124, 0.0420758, 0.0180466, 0.0202931, 
    # 0.0192818, 6.48827e-05, 2.94781e-05, 0.0255411, 0.0250317, 0.000164438, 0.00600404, 
    # 0.000621912, 0.000169004, 0.000155712, 0.000350787, 2.85048e-05, 0.000309135, 
    # 0.000309159, 0.000265391
    # ])     
    
    # # the relative tolerance is fixed to 1e-05, because the expected results 
    # # were displayed by spyder, and have 6 significative numbers
    # np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
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
        
    # indexes
    indx_tr_cap = 2         # Wilcock
    indx_tr_partition = 4   # Shear stress
    indx_flo_depth = 1      # Manning    

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer, ts_length, 
                                                 indx_tr_cap , indx_tr_partition, indx_flo_depth)
        
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([2.37000e+02, 1.72400e+03, 3.57100e+03, 1.98000e+02, 6.20000e+01,
                                2.29800e+03, 7.06000e+03, 7.10590e+04, 2.14120e+04, 3.32050e+04,
                                1.43800e+04, 1.15600e+03, 1.20500e+03, 1.64500e+03, 1.02042e+05,
                                1.50500e+04, 8.33600e+03, 2.23900e+04, 5.13687e+05, 3.94466e+05,
                                3.19882e+05, 2.12873e+05, 2.06835e+05, 1.95621e+05, 9.06180e+04,
                                1.31266e+05, 1.72506e+05, 9.39500e+04, 2.37467e+05, 2.90051e+05,
                                2.60656e+05, 4.32304e+05, 4.77223e+05, 4.66634e+05, 4.48657e+05,
                                4.88557e+05, 4.28901e+05, 5.02372e+05, 4.49916e+05, 4.58972e+05,
                                5.21112e+05, 4.24553e+05, 2.76758e+05, 1.11139e+05, 0.00000e+00,
                                2.70000e+01, 9.00000e+00, 1.82000e+02, 1.20000e+01, 1.70000e+01,
                                8.19750e+05, 2.35931e+05, 5.10000e+01, 0.00000e+00, 2.17354e+05,
                                1.73790e+04, 7.36060e+04, 3.36340e+04, 1.16413e+05, 1.16705e+05,
                                2.86696e+05, 4.22740e+04, 2.63751e+05, 9.13910e+04])
    
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([0.00000e+00, 2.37000e+02, 1.72400e+03, 3.57100e+03, 1.98000e+02,
                                8.90000e+01, 2.29800e+03, 7.06900e+03, 7.10590e+04, 2.14120e+04,
                                3.32050e+04, 1.43800e+04, 1.33800e+03, 1.20500e+03, 1.65700e+03,
                                1.02042e+05, 1.50670e+04, 8.33600e+03, 8.42140e+05, 5.13687e+05,
                                3.94466e+05, 3.19933e+05, 4.48804e+05, 2.06835e+05, 4.12975e+05,
                                9.06180e+04, 1.48645e+05, 1.72506e+05, 1.67556e+05, 2.37467e+05,
                                3.23685e+05, 2.60656e+05, 5.48717e+05, 5.93928e+05, 4.66634e+05,
                                7.35353e+05, 4.88557e+05, 4.71175e+05, 7.66123e+05, 4.49916e+05,
                                4.58972e+05, 6.12503e+05, 4.24553e+05, 2.76758e+05, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
                                0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00])  
    
    np.testing.assert_array_equal(test_result, expected_result)
    
    # # D50 active layer
    # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    # expected_result = np.array([
    # 0.0247771, 0.0151654, 0.0151651, 0.0211763, 0.0211762, 0.0178638, 0.0165646, 
    # 0.00860023, 0.0260577, 0.0102222, 0.027685, 0.0183552, 0.0161581, 0.0184497, 
    # 0.00458518, 0.0185011, 0.010417, 0.00217651, 0.000165847, 0.000149322, 
    # 0.000127542, 0.000215278, 0.000171118, 0.000136044, 0.000103275, 0.000994783, 
    # 0.00129788, 0.00209076, 0.000616477, 0.000603861, 0.000514625, 0.000355313, 
    # 0.000316236, 0.000308643, 0.000307156, 0.000190073, 0.000182749, 0, 
    # 0.000321646, 0.000311395, 0.000306277, 0.00028365, 0.000273104, 0.0002603, 
    # 0.0319107, 0.025124, 0.0420758, 0.0180466, 0.0202931, 0.0192818, 6.48827e-05, 
    # 2.94781e-05, 0.0255411, 0.0250317, 0.000164438, 0.00600405, 0.000621912, 
    # 0.000169004, 0.000155712, 0.000350787, 2.85048e-05, 0.000309135, 0.000309159, 
    # 0.000265391
    # ])  
    
    # # the relative tolerance is fixed to 1e-05, because the expected results 
    # # were displayed by spyder, and have 6 significative numbers
    # np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
    print('Tuto bene with Po case test using Wilcock formula, all new options true\n')
    

# def test_Po_Engelund_all_new_options_true_except_tlag():
#     '''150 days are simulated. 
#     We use Engelund. With the "Bed Material Fraction" partitioning. 
#     '''
    
#     # User defined parameters:
#     deposit_layer = 100000
#     eros_max = 1
#     update_slope = False
#     timescale = 365 
#     ts_length = 60 * 60 * 24
#     sed_range = [-8, 3]  
#     n_classes = 6  
#     save_dep_layer = 'never'  
#     roundpar = 0    
    
    
#     # reach data
#     network = gpd.GeoDataFrame.from_file(filename_river_network)  # read shapefine from shp format
#     reach_data = ReachData(network)
#     reach_data.deposit = np.repeat(deposit_layer, reach_data.n_reaches)
#     sorted_indices = reach_data.sort_values_by(reach_data.from_n)
#     Network = graph_preprocessing(reach_data)
    
#     # Q file
#     Q = extract_Q(filename_q)
#     Q_new = np.zeros((Q.shape)) #reorganise Q file according to reachdata sorting
#     for i, idx in enumerate(sorted_indices): 
#         Q_new[:,i] = Q.iloc[:,idx]
#     Q = Q_new
    
#     # Sediment classes 
#     psi = np.linspace(sed_range[0], sed_range[1], num=n_classes, endpoint=True).astype(float)
#     dmi = 2**(-psi).reshape(-1,1)
#     print(min(reach_data.D16) * 1000, ' must be greater than ', np.percentile(dmi, 10, method='midpoint'))
#     print(max(reach_data.D84) * 1000, ' must be lower than ',  np.percentile(dmi, 90, method='midpoint'))
#     Fi_r, _, _ = GSDcurvefit(reach_data.D16, reach_data.D50, reach_data.D84, psi)
    
#      # External sediment
#     Qbi_input = np.zeros((timescale, reach_data.n_reaches, n_classes))

#     # Input sediment load in deposit layer
#     deposit = reach_data.deposit * reach_data.length
#     Qbi_dep_in = np.zeros((reach_data.n_reaches, 1, n_classes))
#     for n in range(reach_data.n_reaches):
#         Qbi_dep_in[n] = deposit[n] * Fi_r[n,:]
        
#     # indexes
#     indx_tr_cap = 3         # Engelund and Hansen
#     indx_tr_partition = 2   # BMF
#     indx_flo_depth = 1      # Manning
    
#     # Options for the cascade algorithm (by default, they are all True):        
#     # If all these options are False, we are reproducing the algorithme of 
#     # the old version. Which means that cascades are all passing if the time step 
#     # is not finished for them (= based on their velocities) + overpassing cascades are 
#     # not considered in the mobilised volume nor transported

#     # Option 1: If True, we consider ovepassing sediment in the output (Qbimob and Qbitr).
#     # But this does not change the way sediment move.
#     op1 = True

#     # Option 2: If True, we now include present cascades from upstream + reach material
#     # in the transport capacity calculation, to check if they should pass or not. 
#     op2 = True

#     # Option 3: If True, we consider a time lag between the beginning of the time step,
#     # and the arrival of the first cascade to the ToN of the reach, 
#     # during which we are able to mobilise from the reach itself
#     op3 = False

#     # Run definition
#     data_output, extended_output = DCASCADE_main(reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
#                                                  roundpar, update_slope, eros_max, save_dep_layer, ts_length,
#                                                  indx_tr_cap, indx_tr_partition, indx_flo_depth,
#                                                  passing_cascade_in_outputs = op1,
#                                                  passing_cascade_in_trcap = op2,
#                                                  time_lag_for_mobilised = op3)
    
        
    
#     # Test the total mobilised volume per reach
#     test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
#     expected_result = np.array([4.14300e+03, 2.29430e+04, 3.12340e+04, 2.48830e+04, 2.35320e+04, 3.22990e+04,
#                                 6.78200e+04, 1.43471e+05, 1.16242e+05, 1.20478e+05, 1.63291e+05, 9.06180e+04,
#                                 5.04410e+04, 5.02020e+04, 1.38259e+05, 9.29290e+04, 8.20530e+04, 3.21620e+04,
#                                 4.23471e+05, 2.87235e+05, 2.19836e+05, 1.46071e+05, 1.52965e+05, 1.34645e+05,
#                                 7.35970e+04, 1.02069e+05, 1.41122e+05, 1.26504e+05, 1.63082e+05, 1.84718e+05,
#                                 1.68956e+05, 2.52966e+05, 3.73264e+05, 3.66168e+05, 3.23256e+05, 4.31321e+05,
#                                 3.76460e+05, 2.46454e+05, 3.24534e+05, 3.16762e+05, 3.15449e+05, 2.90980e+05,
#                                 1.91149e+05, 6.82810e+04, 7.50000e+01, 1.41300e+03, 3.08100e+03, 2.77600e+03,
#                                 2.56800e+03, 7.49000e+02, 9.92928e+05, 1.70585e+05, 3.34000e+02, 1.19500e+03,
#                                 2.72492e+05, 2.53560e+04, 4.00960e+04, 1.10890e+04, 1.84937e+05, 1.01814e+05,
#                                 2.64445e+05, 1.23760e+04, 2.83389e+05, 7.55810e+04])
    

    
#     np.testing.assert_array_equal(test_result, expected_result)
   
#     # Test the total transported volume per reach
#     test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
#     expected_result = np.array([      0.,    4053.,   21994.,   29674.,   23629.,   23642.,
#                                 30924.,     68095.,  136433.,  110250.,  115517.,  155205.,
#                                 88474.,     48056.,   50468.,  133154.,   89723.,   78395.,
#                                 1015077.,  412251.,  277058.,  210785.,  303672.,  147533.,
#                                 385398.,    67917.,  124560.,  138234.,  161802.,  159654.,
#                                 191711.,    165453.,  425568.,  439913.,  357961.,  572729.,
#                                 421573.,    138761.,  740441.,  318031.,  310350.,  379992.,
#                                 284004.,    185390.,       0.,       0.,       0.,       0.,
#                                 0.,       0.,       0.,       0.,       0.,       0.,
#                                 0.,       0.,       0.,       0.,       0.,       0.,
#                                 0.,       0.,       0.,       0.])  
    
#     # the absolute tolerance is fixed to 1e6, because the expected results 
#     # were displayed by spyder, and have 6 significative numbers
#     np.testing.assert_allclose(test_result, expected_result, atol = 1e06)
    
#     # # D50 active layer: DD: TO DO
#     # test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
#     # expected_result = np.array([2.47770230e-02, 1.51755507e-02, 1.51655911e-02, 2.02552886e-02,
#     #        2.03466590e-02, 1.77876247e-02, 1.65305320e-02, 8.71357324e-03,
#     #        2.22509619e-02, 1.02616880e-02, 2.92620602e-02, 1.42418506e-02,
#     #        1.58089862e-02, 1.83735086e-02, 4.62385848e-03, 1.58265987e-02,
#     #        1.01485931e-02, 2.08804488e-03, 8.22115101e-05, 6.77232744e-04,
#     #        9.22278655e-04, 2.08774248e-03, 1.55653800e-03, 8.36720622e-04,
#     #        5.65198322e-05, 1.84719781e-03, 1.46731157e-03, 2.54021205e-03,
#     #        4.89826506e-04, 5.33831955e-04, 4.38677070e-04, 2.74783902e-04,
#     #        2.48700526e-04, 2.39310130e-04, 2.21651510e-04, 1.24331445e-04,
#     #        1.43998111e-04, 2.23117944e-04, 2.04964090e-04, 2.62114246e-04,
#     #        2.96271429e-04, 2.61212012e-04, 2.48002917e-04, 2.41762730e-04,
#     #        3.19105188e-02, 2.51241504e-02, 4.20755494e-02, 1.80465901e-02,
#     #        2.02931896e-02, 1.92805646e-02, 6.48783109e-05, 2.96288850e-05,
#     #        2.55396815e-02, 2.50317658e-02, 1.64401892e-04, 6.00410774e-03,
#     #        6.25982204e-04, 1.69362805e-04, 1.54117240e-04, 3.50491022e-04,
#     #        2.86550815e-05, 3.09206758e-04, 3.09270737e-04, 2.65200462e-04
#     # ])     
    
#     # # the relative tolerance is fixed to 1e-05, because the expected results 
#     # # were displayed by spyder, and have 6 significative numbers
#     # np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
#     print('Tuto bene with Po case test using Engelund formula, all new options false \n')




if __name__ == "__main__":
    test_Po_Engelund_all_new_options_false()
    test_Po_Wilcock_all_new_options_false()   
    test_Po_Engelund_all_new_options_true()
    test_Po_Wilcock_all_new_options_true()
    # test_Po_Engelund_all_new_options_true_except_tlag() #to update with good values