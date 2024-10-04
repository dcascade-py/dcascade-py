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

''' List of tests performed here:
    
        test_Vjosa_Engelund_all_new_options_false
        test_Vjosa_Wilcock_all_new_options_false
        (the expected results are the same as the version 1 of dcascade)
        
        test_Vjosa_Engelund_all_new_options_true
        test_Vjosa_Wilcock_all_new_options_true
        (the expected results are the one obtained at commit "Adding some things" 4ba9397)
'''


#Pathes
path_river_network = Path('Input/input_trial/')
name_river_network = 'River_Network.shp'
filename_river_network = path_river_network / name_river_network

path_q = Path('Input/input_trial/')
name_q = 'Q_Vjosa.csv'
filename_q = path_q / name_q

# User defined parameters:
deposit_layer = 100000
eros_max = 1
update_slope = False
timescale = 20 
ts_length = 60 * 60 * 24
sed_range = [-8, 5]  
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


def test_Vjosa_Wilcock_all_new_options_false():
    '''20 days are simulated. 
    We use Wilcock and Crowes. 
    '''        
    # indexes
    indx_tr_cap = 2      # Wilkock and Crowe 2003
    indx_partition = 4   # Shear stress correction
    indx_flo_depth = 1   # Manning
    indx_slope_red = 1   # None
    indx_velocity = 1    # same velocity for all classes
    
    # options in v2    
    consider_overtaking_sed_in_outputs = False
    compare_with_tr_cap = False
    time_lag_for_Vmob = False
    consider_passing_sed_in_tr_cap = False
      
    # run model
    data_output, _ = DCASCADE_main(indx_tr_cap, indx_partition, indx_flo_depth,
                                                 indx_slope_red, indx_velocity, reach_data,
                                                 Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer,
                                                 ts_length, 
                                                 consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                                 time_lag_for_Vmob, consider_passing_sed_in_tr_cap)
        
    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array()   
    np.testing.assert_array_equal(test_result, expected_result)
   
    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array()                      
    np.testing.assert_array_equal(test_result, expected_result)
    
    #----Test D50 active layer
    test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    expected_result = np.array()
    
    # the relative tolerance is fixed to 1e-05, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
    print('\n Tuto bene with Vjosa case test using Wilcock formula, all option false  \n')
    
    
def test_Vjosa_Engelund_all_new_options_false():
    '''20 days are simulated. 
    We use Engelund. With the "Bed Material Fraction" partitioning. 
    '''        
    # indexes
    indx_tr_cap = 3      # Engelund and Hansen
    indx_partition = 2   # BMF
    indx_flo_depth = 1   # Manning
    indx_slope_red = 1   # None
    indx_velocity = 1    # same velocity for all classes
    
    # options in v2                  
    consider_overtaking_sed_in_outputs = False
    compare_with_tr_cap = False
    time_lag_for_Vmob = False
    consider_passing_sed_in_tr_cap = False
      
    # run model
    data_output, _ = DCASCADE_main(indx_tr_cap, indx_partition, indx_flo_depth,
                                                 indx_slope_red, indx_velocity, reach_data,
                                                 Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer,
                                                 ts_length, 
                                                 consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                                 time_lag_for_Vmob, consider_passing_sed_in_tr_cap)
        
    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array()    
    np.testing.assert_array_equal(test_result, expected_result)
   
    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array()          
    # the absolute tolerance is fixed to 1e6, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, atol = 1e06)
    
    #----Test D50 active layer
    test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    expected_result = np.array()
           
    # the relative tolerance is fixed to 1e-05, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
    print('\n Tuto bene with Vjosa case test using Engelund formula, all option false \n')



def test_Vjosa_Wilcock_all_new_options_true():
    '''20 days are simulated. 
    We use Wilcock and Crowes. 
    '''        
    # indexes
    indx_tr_cap = 2      # Wilkock and Crowe 2003
    indx_partition = 4   # Shear stress correction
    indx_flo_depth = 1   # Manning
    indx_slope_red = 1   # None
    indx_velocity = 1    # same velocity for all classes
    
    # options in v2    
    consider_overtaking_sed_in_outputs = True
    compare_with_tr_cap = True
    time_lag_for_Vmob = True
    consider_passing_sed_in_tr_cap = False
      
    # run model
    data_output, _ = DCASCADE_main(indx_tr_cap, indx_partition, indx_flo_depth,
                                                 indx_slope_red, indx_velocity, reach_data,
                                                 Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer,
                                                 ts_length, 
                                                 consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                                 time_lag_for_Vmob, consider_passing_sed_in_tr_cap)
        
    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([2142228.,  501987.,  272118.,   68220.,  770761.,  
                                113206., 175654.])   
    np.testing.assert_array_equal(test_result, expected_result)
   
    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([0., 2802304.,  592712.,  431377.,       
                                0., 0., 0.])                      
    np.testing.assert_array_equal(test_result, expected_result)
    
    #----Test D50 active layer
    test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    expected_result = np.array([0.00235726, 0.00228118, 0.00224194, 0.00217318, 0.00235725,
                                0.00235714, 0.00235705])
    
    # the relative tolerance is fixed to 1e-05, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
    print('\n Tuto bene with Vjosa case test using Wilcock formula, all option true  \n')
    
    
def test_Vjosa_Engelund_all_new_options_true():
    '''20 days are simulated. 
    We use Engelund. With the "Bed Material Fraction" partitioning. 
    '''        
    # indexes
    indx_tr_cap = 3      # Engelund and Hansen
    indx_partition = 2   # BMF
    indx_flo_depth = 1   # Manning
    indx_slope_red = 1   # None
    indx_velocity = 1    # same velocity for all classes
    
    # options in v2                  
    consider_overtaking_sed_in_outputs = True
    compare_with_tr_cap = True
    time_lag_for_Vmob = True
    consider_passing_sed_in_tr_cap = False
      
    # run model
    data_output, _ = DCASCADE_main(indx_tr_cap, indx_partition, indx_flo_depth,
                                                 indx_slope_red, indx_velocity, reach_data,
                                                 Network, Q, Qbi_input, Qbi_dep_in, timescale, psi,
                                                 roundpar, update_slope, eros_max, save_dep_layer,
                                                 ts_length, 
                                                 consider_overtaking_sed_in_outputs, compare_with_tr_cap,
                                                 time_lag_for_Vmob, consider_passing_sed_in_tr_cap)
        
    #----Test the total mobilised volume per reach
    test_result = np.sum(data_output['Mobilized [m^3]'], axis = 0)
    expected_result = np.array([207672.,  64710.,  40064.,  21460.,  42366.,   
                                3755.,   6376.])    
    np.testing.assert_array_equal(test_result, expected_result)
   
    #----Test the total transported volume per reach
    test_result = np.sum(data_output['Transported [m^3]'], axis = 0)
    expected_result = np.array([0., 242459.,  65642.,  43989.,      
                                0.,      0.,      0.])          
    # the absolute tolerance is fixed to 1e6, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, atol = 1e06)
    
    #----Test D50 active layer
    test_result = np.median(data_output['D50 active layer [m]'], axis = 0)
    expected_result = np.array([0.00235726, 0.0011472, 0.00144733, 0.00046104, 
                                0.00235725, 0.00235714, 0.00235705])
           
    # the relative tolerance is fixed to 1e-05, because the expected results 
    # were displayed by spyder, and have 6 significative numbers
    np.testing.assert_allclose(test_result, expected_result, rtol = 1e-05)
    
    print('\n Tuto bene with Vjosa case test using Engelund formula, all option true \n')



if __name__ == "__main__":
    test_Vjosa_Wilcock_all_new_options_false()
    test_Vjosa_Engelund_all_new_options_false()
    test_Vjosa_Wilcock_all_new_options_true()
    test_Vjosa_Engelund_all_new_options_true()