# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:50:28 2024

@author: Diane Doolaeghe
"""

import numpy as np
from DCASCADE_main_script import DCASCADE_main, ReachData
from preprocessing import graph_preprocessing, extract_Q, read_network
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
    al_depth = None
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
                                                 roundpar, update_slope, eros_max, al_depth, save_dep_layer, ts_length,
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
                               1.718600e+05, 1.994500e+05, 3.238720e+05, 3.083550e+05,
                               3.775590e+05, 3.788170e+05, 4.032330e+05, 5.383900e+05,
                               7.827490e+05, 6.997740e+05, 7.258750e+05, 9.430290e+05,
                               8.386780e+05, 4.317970e+05, 7.128370e+05, 7.924940e+05,
                               7.847650e+05, 7.148200e+05, 4.936320e+05, 1.942940e+05,
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
                                975111.,  122746.,  247754.,  336254.,  377665.,  286874.,
                                422685.,  370906.,  833622.,  675993.,  813176., 1255890.,
                                903127.,  158613., 1545168.,  737936.,  776103.,  910449.,
                                699002.,  483035.,       0.,       0.,       0.,       0.,
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
    al_depth = None
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
                                                 roundpar, update_slope, eros_max, al_depth, save_dep_layer, ts_length,
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


def test_Po_Engelund_all_true_no_tlag():
    '''150 days are simulated. 
    We use Engelund. With the "Bed Material Fraction" partitioning. 
    '''
    
    # User defined parameters:
    deposit_layer = 100000
    eros_max = 1
    al_depth = None
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
    op1 = True

    # Option 2: If True, we now include present cascades from upstream + reach material
    # in the transport capacity calculation, to check if they should pass or not. 
    op2 = True

    # Option 3: If True, we consider a time lag between the beginning of the time step,
    # and the arrival of the first cascade to the ToN of the reach, 
    # during which we are able to mobilise from the reach itself
    op3 = False

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, al_depth, save_dep_layer, ts_length,
                                                 indx_tr_cap, indx_tr_partition, indx_flo_depth,
                                                 passing_cascade_in_outputs = op1,
                                                 passing_cascade_in_trcap = op2,
                                                 time_lag_for_mobilised = op3)
    
        
    
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([8.291000e+03, 4.564100e+04, 6.829700e+04, 5.932000e+04,
                               5.918400e+04, 7.451300e+04, 1.430240e+05, 3.523640e+05,
                               3.056510e+05, 2.985600e+05, 4.323130e+05, 2.914700e+05,
                               1.707730e+05, 1.659030e+05, 3.390260e+05, 2.661370e+05,
                               2.414550e+05, 1.234020e+05, 9.160890e+05, 7.596290e+05,
                               6.766940e+05, 5.751970e+05, 5.882520e+05, 5.228490e+05,
                               1.726680e+05, 3.049250e+05, 4.607340e+05, 4.171820e+05,
                               4.782280e+05, 5.766570e+05, 5.656680e+05, 7.555790e+05,
                               1.038937e+06, 1.059025e+06, 9.618000e+05, 1.076785e+06,
                               9.685730e+05, 1.314209e+06, 9.302160e+05, 9.265500e+05,
                               9.218090e+05, 8.164070e+05, 5.510690e+05, 2.055980e+05,
                               1.760000e+02, 2.874000e+03, 6.189000e+03, 5.594000e+03,
                               5.201000e+03, 1.551000e+03, 1.985705e+06, 3.411910e+05,
                               6.870000e+02, 2.504000e+03, 5.450060e+05, 5.068100e+04,
                               7.990200e+04, 2.219700e+04, 3.698170e+05, 2.036160e+05,
                               5.288710e+05, 2.474300e+04, 5.667540e+05, 1.511300e+05])
    
   
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([      0.,    8291.,   45641.,   68297.,   59496.,   62058.,
                                     74513.,  149213.,  352364.,  305651.,  298560.,  432313.,
                                    297064.,  170773.,  171104.,  339026.,  267688.,  241455.,
                                   2109107.,  916089.,  759629.,  677381.,  916388.,  590756.,
                                   1067855.,  172668.,  355606.,  460734.,  497084.,  478228.,
                                    598854.,  565668., 1125396., 1242553., 1059025., 1490671.,
                                   1076785.,  993316., 1880963.,  930216.,  926550., 1072939.,
                                    816407.,  551069.,       0.,       0.,       0.,       0.,
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
    
    print('Tuto bene with Po case test using Engelund formula, all new options true, no time lag \n')


def test_Po_Wilcock_all_true_no_tlag():
    '''150 days are simulated. 
    We use Wilcock and Crowes. 
    '''
    
    # User defined parameters:
    deposit_layer = 100000
    eros_max = 1
    al_depth = None
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
    op1 = True

    # Option 2: If True, we now include present cascades from upstream + reach material
    # in the transport capacity calculation, to check if they should pass or not. 
    op2 = True

    # Option 3: If True, we consider a time lag between the beginning of the time step,
    # and the arrival of the first cascade to the ToN of the reach, 
    # during which we are able to mobilise from the reach itself
    op3 = False

    # Run definition
    data_output, extended_output = DCASCADE_main(reach_data, Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, al_depth, save_dep_layer, ts_length,
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
                               4.88557e+05, 4.28901e+05, 5.06195e+05, 4.53018e+05, 4.60955e+05,
                               5.22529e+05, 4.25408e+05, 2.77164e+05, 1.11201e+05, 0.00000e+00,
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
                               7.35353e+05, 4.88557e+05, 4.71175e+05, 7.69946e+05, 4.53018e+05,
                               4.60955e+05, 6.13920e+05, 4.25408e+05, 2.77164e+05, 0.00000e+00,
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
    
    print('Tuto bene with Po case test using Wilcock formula, all new options true, no time lag \n')



    
def test_Po_Engelund_all_new_options_true():
    '''150 days are simulated. 
    We use Engelund. With the "Bed Material Fraction" partitioning. 
    '''
    
    # User defined parameters:
    deposit_layer = 100000
    eros_max = 1
    al_depth = None
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
                                                 roundpar, update_slope, eros_max, al_depth, save_dep_layer, ts_length, 
                                                 indx_tr_cap , indx_tr_partition, indx_flo_depth)
    
    end = time.time()
    print(end - start)
        
    # Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([8.291000e+03, 4.564100e+04, 6.829700e+04, 5.932000e+04,
                               5.697500e+04, 7.451300e+04, 1.430240e+05, 3.370550e+05,
                               2.902820e+05, 2.856250e+05, 4.335030e+05, 2.911650e+05,
                               1.694430e+05, 1.645040e+05, 3.383230e+05, 2.655190e+05,
                               2.410670e+05, 1.233160e+05, 9.160850e+05, 7.596280e+05,
                               6.768400e+05, 5.747900e+05, 5.860060e+05, 5.177640e+05,
                               1.727080e+05, 2.590090e+05, 3.386420e+05, 3.046100e+05,
                               3.768050e+05, 4.290820e+05, 4.072890e+05, 5.599480e+05,
                               8.023110e+05, 8.360560e+05, 7.551620e+05, 9.538740e+05,
                               8.607500e+05, 1.090585e+06, 8.305540e+05, 8.271050e+05,
                               8.251090e+05, 7.457740e+05, 5.154560e+05, 1.985600e+05,
                               1.760000e+02, 2.874000e+03, 6.189000e+03, 5.594000e+03,
                               5.201000e+03, 1.551000e+03, 1.985705e+06, 3.411910e+05,
                               6.870000e+02, 2.504000e+03, 5.450060e+05, 5.068100e+04,
                               7.990200e+04, 2.219700e+04, 3.698170e+05, 2.036160e+05,
                               5.288710e+05, 2.474300e+04, 5.667540e+05, 1.511300e+05])
    
    np.testing.assert_array_equal(test_result, expected_result)
   
    # Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([      0.,    8291.,   45641.,   68297.,   59496.,   59849.,
                                 74513.,  149213.,  337055.,  290282.,  285625.,  433503.,
                                296759.,  169443.,  169705.,  338323.,  267070.,  241067.,
                               2109021.,  916085.,  759628.,  677527.,  915981.,  588510.,
                               1062770.,  172708.,  309690.,  338642.,  384512.,  376805.,
                                451279.,  407289.,  929765., 1005927.,  836056., 1284033.,
                                953874.,  885493., 1657339.,  830554.,  827105.,  976239.,
                                745774.,  515456.,       0.,       0.,       0.,       0.,
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
    al_depth = None
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
                                                 roundpar, update_slope, eros_max, al_depth, save_dep_layer, ts_length, 
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
                           4.88557e+05, 4.28901e+05, 5.03753e+05, 4.51179e+05, 4.59476e+05,
                           5.21438e+05, 4.24790e+05, 2.76891e+05, 1.11162e+05, 0.00000e+00,
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
                           7.35353e+05, 4.88557e+05, 4.71175e+05, 7.67504e+05, 4.51179e+05,
                           4.59476e+05, 6.12829e+05, 4.24790e+05, 2.76891e+05, 0.00000e+00,
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
    



if __name__ == "__main__":
    test_Po_Engelund_all_new_options_false()
    test_Po_Wilcock_all_new_options_false()  
    test_Po_Engelund_all_true_no_tlag() 
    test_Po_Wilcock_all_true_no_tlag()
    test_Po_Engelund_all_new_options_true()
    test_Po_Wilcock_all_new_options_true()
