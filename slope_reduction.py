# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:21:06 2024

@author: FPitscheider
"""

'''
Reducing the slope for the transport capacity calculations enables to account for form roughness and drag in alpine streams

 The formula proposded by Rickenmann (2005) is based on the D90 and flow depth
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

def slopeRed_Rickenmann(Slope, h, ReachData, t):  
   Slope[t] = Slope[t] * (0.092 * Slope[t] ** (-0.35) * (h / ReachData['D90']) ** (0.33)) ** 1.5 #1.5 = factor a
   
   return Slope
   
def slopeRed_Chiari_Rickenmann(Slope, Q, ReachData, t):        
    Slope[t] = Slope[t] * ((0.133 * (Q.iloc[t,:]**0.19))/(9.81**0.096 * ReachData['D90']**0.47 * Slope[t]**0.19)) ** 1.5 #1.5 = factor a
    
    return Slope

def slopeRed_Nitsche(Slope, h, ReachData, t):
    Slope[t] = Slope[t] * ((2.5 *((h / ReachData['D84']) ** (5/6))) / (6.5 ** 2 + 2.5 ** 2 * ((h /  ReachData['D84'])**(5/3)))) ** 1.5 #1.5 = factor e
    
    return Slope


def choose_slopeRed(ReachData, Slope, Q, t, h, slopeRed):
    if slopeRed == 1:
        Slope = Slope    
    
    elif slopeRed == 2:
        Slope = slopeRed_Rickenmann(Slope, h, ReachData, t)
    
    elif slopeRed == 3:
        Slope = slopeRed_Chiari_Rickenmann(Slope, Q, ReachData, t)
    
    elif slopeRed == 4:
        Slope = slopeRed_Nitsche(Slope, h, ReachData, t)
    
    return Slope