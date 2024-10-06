# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:50:28 2024

@author: FPitscheider
"""

import numpy as np
import numpy.matlib
from transport_capacity_computation import Wilcock_Crowe_formula




def test_wilcock_Crowe_formula():
    
    #input parameters that are used to check if formula in the script gives the same results as the original W&C formula
    Fi_r_reach = np.array([0.01])
    D50 = 0.01
    Slope = 0.05
    Wac = np.array([10])
    h = 0.5
    psi = np.array([-1])
    
    #manually calculated tr_cap
    expected_trCap_WC =  0.0058621118228
    
    # importing tr_cap from W&C formula in script
    trCap_WC_script, tau, tau_r50 = Wilcock_Crowe_formula(Fi_r_reach, D50, Slope, Wac, h, psi)
    
    #comparing the first 4 decimals
    np.testing.assert_allclose(trCap_WC_script[0][0], expected_trCap_WC, atol=1e-4)
    
    '''
    curent calculations for daily transport capacity [m^3/day] with the above mentioned values
    
    daily_expected_trCap_WC = 0.00586 * 60 *60 * 24 = 506,30
    daily_trCap_WC_script = 0.00582 * 60 * 60 * 24 = 502,85
    
    difference = 0.68%

    '''

if __name__ == "__main__":
    test_wilcock_Crowe_formula()