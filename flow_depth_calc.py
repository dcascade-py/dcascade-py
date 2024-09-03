# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:05:39 2024

@author: FPitscheider
"""
import numpy as np
import numpy.matlib
from supporting_functions import D_finder    

def h_manning(ReachData, Slope, Q, t):
    h = np.power(Q.iloc[t,:].astype('float')*ReachData['n'].astype('float')/(ReachData['Wac'].astype('float')*np.sqrt(Slope[t])), 3/5)
    v = 1/ReachData['n'].astype('float')*np.power(h,2/3)*np.sqrt(Slope[t])
        
    return h, v

def h_ferguson(ReachData, Slope, Q, t):
    
    #calculate water depth and velocity with the Ferguson formula (2007)
    q_star = Q.iloc[t,:] / (ReachData['Wac'] * np.sqrt(9.81 * Slope[t] * ReachData['D84']**3))
    
    #𝑝…𝑖𝑓 𝑞∗<100 → 𝑝=0.24, 𝑖𝑓 𝑞^∗>100 → 𝑝=0.31
    p = np.where(q_star < 100, 0.24, 0.31)
    
    h = 0.015 * ReachData['D84'] * (q_star**(2*p)) / (p**2.5)    
    v = (np.sqrt(9.81 * h * Slope[t])* 6.5 * 2.5 * (h/ReachData['D84'])) / np.sqrt((6.2 ** 2) * (2.5 ** 2) * ((h/ReachData['D84']) ** (5/3)))
   
    return h, v

def choose_flow_depth(ReachData, Slope, Q, t, flo_depth):
    if flo_depth == 1:
        [h, v] = h_manning(ReachData, Slope, Q, t)
    
    elif flo_depth == 2:
        [h, v] = h_ferguson(ReachData, Slope, Q, t)
    
    return h, v