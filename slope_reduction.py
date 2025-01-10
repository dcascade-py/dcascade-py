# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:21:06 2024

@author: FPitscheider
"""

'''
Reducing the slope for the transport capacity calculations enables to account for form roughness and drag in alpine streams

 The formula proposed by Rickenmann (2005) is based on the D90 and flow depth
    Rickenmann, D. (2005): Geschiebetransport bei steilen Gefällen. In: Mitteilungen der Versuchs- anstalt für Wasserbau, 
    Hydrologie und Glaziologie, ETH Zurich, Nr. 190, pp. 107–119.
    
The formula by Chiari & Richenmann (2011) is based on the D90 and the discharge
    Chiari, M. and Rickenmann, D. (2011), Back-calculation of bedload transport in steep channels with a numerical model. 
    Earth Surf. Process. Landforms, 36: 805-815. https://doi.org/10.1002/esp.2108
    
The formula by Nitsche et al. (2011) is based on the flow depth and the D84
    Nitsche, M., D. Rickenmann, J. M. Turowski, A. Badoux, and J. W. Kirchner (2011), Evaluation of bedload transport 
    predictions using flow resistance equations to account for macro-roughness in steep mountain streams, Water Resour. 
    Res., 47, W08513, doi:10.1029/2011WR010645. 
'''

'''
----TO DO----

- change D values from Reach Data to values that D-Cascade calculates for each timestep
'''

from constants import GRAV
import numpy as np
exponent_a = 1.5 # exponent a between 1-2, typically 1.5

def slopeRed_Rickenmann(slope, h, roughness): 

    reduced_slope = slope * (0.092 * slope ** (-0.35) * (h / roughness) ** (0.33)) ** exponent_a  
 
    return reduced_slope
   
def slopeRed_Chiari_Rickenmann(slope, Q_t, roughness):
    
    reduced_slope = slope * ((0.133 * (Q_t**0.19))/(GRAV**0.096 * roughness**0.47 * slope**0.19)) ** exponent_a

    return reduced_slope

def slopeRed_Nitsche(slope, h, D84):
    reduced_slope = slope * ((2.5 *((h / D84) ** (5/6))) / (6.5 ** 2 + 2.5 ** 2 * ((h /  D84)**(5/3)))) ** exponent_a
    
    return reduced_slope

def choose_slopeRed(reach_data, SedimSys, Q, t, h, indx_slope_red):
    if indx_slope_red == 1:
        slope_t = SedimSys.slope[t]  
    
    elif indx_slope_red == 2:
        slope_t = slopeRed_Rickenmann(SedimSys.slope[t], h, reach_data.roughness)
    
    elif indx_slope_red == 3:
        slope_t = slopeRed_Chiari_Rickenmann(SedimSys.slope[t], Q[t, :], reach_data.roughness)
    
    elif indx_slope_red == 4:
        slope_t = slopeRed_Nitsche(SedimSys.slope[t], h, reach_data.D84)
    
    SedimSys.slope[t] = slope_t
    
    return SedimSys.slope