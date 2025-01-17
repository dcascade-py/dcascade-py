# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:34:03 2024

@author: FPitscheider
"""

import numpy as np
import numpy.matlib

from supporting_functions import D_finder

# To account for potential imprecision or errors in the calculations, we use
# an " absolute tolerance" EPSILON of 0.0001.
EPSILON = 1e-4



def test_d_finder():
    '''
    Testing the D_finder function:
    - with D100, several layers.
    - with D50, several layers.
    - with D50, single layer.
    '''

    # Input parameters that are used to check if the formula in the function
    # gives the same results as the original formula
    fi_r = np.array([[0.15, 0.25, 0.35, 0.25], [0.25, 0.25, 0.25, 0.25]])
    psi = np.array([-4, -3, -2, -1])
    d_value = 100

    # Manually calculated transport capacity
    expected_d90_value = [0.016, 0.016]

    # Computing the transport capacity with the D-CASCADE implementation
    d_calculated = D_finder(fi_r, d_value, psi)
    print("d_calculated", d_calculated)

    # Asserting the computed value is equal to manually calculated one, allowing
    # for with error tolerance EPSILON
    np.testing.assert_allclose(d_calculated, expected_d90_value, atol=EPSILON)

    ###################

    # Input parameters that are used to check if the formula in the function
    # gives the same results as the original formula
    fi_r = np.array([[0.15, 0.25, 0.35, 0.25], [0.25, 0.25, 0.25, 0.25]])
    psi = np.array([-4, -3, -2, -1])
    d_value = 50

    # Manually calculated transport capacity
    expected_d90_value = [0.00328134142, 0.004]

    # Computing the transport capacity with the D-CASCADE implementation
    d_calculated = D_finder(fi_r, d_value, psi)
    print("d_calculated", d_calculated)

    # Asserting the computed value is equal to manually calculated one, allowing
    # for with error tolerance EPSILON
    np.testing.assert_allclose(d_calculated, expected_d90_value, atol=EPSILON)

    ###################

    # Input parameters that are used to check if the formula in the function
    # gives the same results as the original formula
    fi_r = np.array([0.15, 0.25, 0.35, 0.25])
    psi = np.array([-4, -3, -2, -1])
    d_value = 50

    # Manually calculated transport capacity
    expected_d90_value = [0.00328134142]

    # Computing the transport capacity with the D-CASCADE implementation
    d_calculated = D_finder(fi_r, d_value, psi)
    print("d_calculated", d_calculated)

    # Asserting the computed value is equal to manually calculated one, allowing
    # for with error tolerance EPSILON
    np.testing.assert_allclose(d_calculated, expected_d90_value, atol=EPSILON)

if __name__ == "__main__":
    test_d_finder()
    print("All tests successfully run.")