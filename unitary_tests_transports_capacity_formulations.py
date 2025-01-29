"""
Created on Thu Sep  5 13:50:28 2024

@author: FPitscheider
"""

import numpy as np
import numpy.matlib

from transport_capacity_computation import TransportCapacityCalculator

# To account for potential imprecision or errors in the calculations, we use
# an " absolute tolerance" EPSILON of 0.0001.
EPSILON = 1e-4

def test_engelund_hansen_formula():
    '''
    Precision of calculations for transport capacity [m^3/s] with the input values:

    expected_tr_cap = 0.046356178
    computed_tr_cap = 0.04635618

    Relative difference = 2.12719452e-09
    '''

    # Input parameters that are used to check if the formula in the function
    # gives the same results as the original formula
    D50 = 0.01
    slope = 0.05
    wac = np.array([10])
    v = 2
    h = 0.5

    # Manually calculated transport capacity
    expected_tr_cap = 0.046356178

    # Computing the transport capacity with the D-CASCADE implementation
    calculator = TransportCapacityCalculator(np.nan, np.nan, slope, np.nan, wac, v, h, np.nan)
    calculator.D50 = D50
    computed_tr_cap = calculator.Engelund_Hansen_formula()

    # Asserting the computed value is equal to manually calculated one, allowing
    # for with error tolerance EPSILON
    np.testing.assert_allclose(computed_tr_cap['tr_cap'][0], expected_tr_cap, atol=EPSILON)


def test_wilcock_crowe_formula():
    '''
    Precision of calculations for transport capacity [m^3/s] with the input values:

    expected_tr_cap = 0.00586
    computed_tr_cap = 0.00582

    Relative difference = 0.68%
    '''

    # Input parameters that are used to check if the formula in the function
    # gives the same results as the original formula
    Fi_r_reach = np.array([0.01])
    D50 = 0.01
    slope = 0.05
    wac = np.array([10])
    h = 0.5
    psi = np.array([-1])

    # Manually calculated transport capacity
    expected_tr_cap = 0.0058621118228

    # Computing the transport capacity with the D-CASCADE implementation
    calculator = TransportCapacityCalculator(Fi_r_reach, np.nan, slope, np.nan, wac, np.nan, h, psi)
    calculator.D50 = D50
    computed_tr_cap = calculator.Wilcock_Crowe_formula()

    # Asserting the computed value is equal to manually calculated one, allowing
    # for with error tolerance EPSILON
    np.testing.assert_allclose(computed_tr_cap['tr_cap'][[0]], expected_tr_cap, atol=EPSILON)


def test_rickenmann_formula():
    '''
    Precision of calculations for transport capacity [m^3/s] with the input values:

    expected_tr_cap =
    computed_tr_cap = 0.04805394

    expected_qc =
    computed_qc = 0.01346169176826027

    Relative difference =
    '''

    # Input parameters that are used to check if the formula in the function
    # gives the same results as the original formula
    D50 = 0.01
    slope = 0.05
    wac = np.array([10])
    discharge = 3

    # Manually calculated transport capacity
    expected_tr_cap = 0.048053935
    expected_qc = 0.013461692

    # Computing the transport capacity with the D-CASCADE implementation
    calculator = TransportCapacityCalculator(np.nan, np.nan, slope, discharge, wac, np.nan, np.nan, np.nan)
    calculator.D50 = D50
    computed_tr_cap = calculator.Rickenmann_formula()

    # Asserting the computed value is equal to manually calculated one, allowing
    # for with error tolerance EPSILON
    np.testing.assert_allclose(computed_tr_cap['tr_cap'][[0]], expected_tr_cap, atol=EPSILON)
    np.testing.assert_allclose(computed_tr_cap['Qc'], expected_qc, atol=EPSILON)


if __name__ == "__main__":
    test_wilcock_crowe_formula()
    test_engelund_hansen_formula()
    test_rickenmann_formula()
    print("All tests successfully run.")

