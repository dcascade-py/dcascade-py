# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:03:00 2023

Different formula for the calculation of the tranport capacity and for the assignment of the velocity

@author: Elisa Bozzolan 
"""

import numpy as np
import numpy.matlib
from supporting_functions import D_finder
from constants import (
    RHO_S,
    RHO_W,
    GRAV,
    R_VAR,
)

def Parker_Klingeman_formula(fi_r_reach, D50, slope, wac, h, psi, **kwargs):
    """
    Returns the value of the transport capacity [Kg/s] for each sediment class
    in the reach measured using the Parker and Klingeman equations.
    This function is for use in the D-CASCADE toolbox.
    CHECK THE UNIT OF THE RETURNED VARIABLE.
    
    References:
    Parker and Klingeman (1982). On why gravel bed streams are paved. Water Resources Research.
    """
    
    if 'gamma' not in kwargs: # if gamma was not given as input
        gamma = 0.05 # hiding factor
    
    dmi = 2**(-psi) / 1000 # sediment classes diameter [m]
    
    ## Transport capacity from Parker and Klingema equations
    
    tau = RHO_W * GRAV * h * slope # bed shear stress [Kg m-1 s-1]
    
    # tau_r50 formula from Mueller et al. (2005)
    # reference shear stress for the mean size of the bed surface sediment [Kg m-1 s-1]
    tau_r50 = (0.021 + 2.18 * slope) * (RHO_W * R_VAR * GRAV * D50)
    
    tau_ri = tau_r50 * (dmi/D50)** gamma # reference shear stress for each sediment class [Kg m-1 s-1]
    phi_ri = tau / tau_ri
    
    # Dimensionless transport rate for each sediment class [-]
    w_i = 11.2 * (np.maximum(1 - 0.853/phi_ri, 0))**4.5
    
    # Dimensionful transport rate for each sediment class [m3/s]
    tr_cap = wac * w_i * fi_r_reach * (tau / RHO_W)**(3/2) / (R_VAR * GRAV)
    tr_cap[np.isnan(tr_cap)] = 0 #if Qbi_tr are NaN, they are put to 0
    
    return tr_cap, tau, tau_r50
    
def Wilcock_Crowe_formula(fi_r_reach, D50, slope, wac, h, psi): 
    """
    Returns the value of the transport capacity [m3/s] for each sediment class
    in the reach measured using the Wilcock and Crowe equations.
    This function is for use in the D-CASCADE toolbox.
    
    References:
    Wilcock, Crowe(2003). Surface-based transport model for mixed-size sediment. Journal of Hydraulic Engineering.
    """

    dmi = 2**(-psi) / 1000 # sediment classes diameter [m]
    
    if fi_r_reach.ndim == 1:
        fi_r_reach = fi_r_reach[:,None]
        # Fraction of sand in river bed (sand considered as sediment with phi > -1)
        Fr_s = np.sum((psi > - 1)[:,None] * 1 * fi_r_reach)
    else:
        Fr_s = np.sum((psi > - 1)[:,None] * 1 * fi_r_reach, axis = 0)[None,:]
    ## Transport capacity from Wilcock-Crowe equations

    tau = np.array(RHO_W * GRAV * h * slope) # bed shear stress [Kg m-1 s-1]
    if tau.ndim != 0:
        tau = tau[None,:] # add a dimension for computation
    
    # reference shear stress for the mean size of the bed surface sediment [Kg m-1 s-1]
    tau_r50 = (0.021 + 0.015 * np.exp(-20 * Fr_s)) * (RHO_W * R_VAR * GRAV * D50)
    
    b = 0.67 / (1 + np.exp(1.5 - dmi / D50)) # hiding factor
    
    fact = (dmi / D50)**b
    tau_ri = tau_r50 * fact[:,None] # reference shear stress for each sediment class [Kg m-1 s-1]
    
    phi_ri = tau / tau_ri
    # Dimensionless transport rate for each sediment class [-]
    # The formula changes for each class according to the phi_ri of the class
    # is higher or lower then 1.35.
    W_i = (phi_ri >= 1.35) * (14 * (np.maximum(1 - 0.894 / np.sqrt(phi_ri), 0))**4.5) + (phi_ri < 1.35) * (0.002 * phi_ri**7.5)
    
    # Dimensionful transport rate for each sediment class [m3/s]
    if wac.ndim == 0:
        tr_cap = wac * W_i * fi_r_reach * (tau / RHO_W)**(3/2) / (R_VAR * GRAV)
        if tr_cap.ndim > 1:
           tr_cap = np.squeeze(tr_cap) # EB: a bit of a mess here with dimensions, corrected a posteriori. I want a 1-d vector as output 
    else:
        wac = np.array(wac)[None, :]
        tr_cap = wac * W_i * fi_r_reach * (tau / RHO_W)**(3/2) / (R_VAR * GRAV)
    
    tr_cap[np.isnan(tr_cap)] = 0 #if Qbi_tr are NaN, they are put to 0

    return tr_cap, tau, tau_r50

def Engelund_Hansen_formula(D50, slope, wac, v, h):
    """
    Returns the value of the transport capacity [m3/s] for each sediment class
    in the reach measured using the Engelund and Hansen equations.
    This function is for use in the D-CASCADE toolbox.
    
    References:
    Engelund, F., and E. Hansen (1967), A Monograph on Sediment Transport in Alluvial Streams, 
    Tekniskforlag, Copenhagen.
    """
    
    # friction factor
    C = (2 * GRAV * slope * h) / v**2   

    # dimensionless shear stress
    tau_eh = (slope * h) / (R_VAR * D50)
    # dimensionless transport capacity
    q_eh = 0.05 / C * tau_eh**(5/2)
    # dimensionful transport capacity per unit width  m3/(s*m )
    q_eh_dim = q_eh * np.sqrt(R_VAR * GRAV * D50**3) # m3/s 
    qs_eh = q_eh_dim * wac
    
    tr_cap = qs_eh # m3/s
    
    return tr_cap

def Yang_formula(fi_r_reach, D50, slope, Q, v, h, psi): 
    """
    Returns the value of the transport capacity [m3/s] for each sediment class
    in the reach measured using the Yang equations.
    This function is for use in the D-CASCADE toolbox.
    
    References: 
    Stevens Jr., H. H. & Yang, C. T. Summary and use of selected fluvial sediment-discharge formulas. (1989).
    see also: Modern Water Resources Engineering: https://books.google.com/books?id=9rW9BAAAQBAJ&pg
    =PA347&dq=yang+sediment+transport+1973&hl=de&sa=X&ved=0ahUKEwiYtKr72_bXAhVH2mMKHZwsCdQQ6AEILTAB
    #v=onepage&q=yang%20sediment%20transport%201973&f=false
    """

    dmi = 2**(-psi) / 1000 # sediment classes diameter [m]
    nu = 1.003*1E-6        # kinematic viscosity @ : http://onlinelibrary.wiley.com/doi/10.1002/9781118131473.app3/pdf
    
    GeoStd = GSD_std(fi_r_reach, dmi);
    
    #  1) settling velocity for grains - Darby, S; Shafaie, A. Fall Velocity of Sediment Particles. (1933)
    #         
    #  Dgr = D50*(GRAV*R_VAR/nu**2)**(1/3);
    #     
    #  if Dgr<=10:  
    #      w = 0.51*nu/D50*(D50**3*GRAV*R_VAR/nu**2)**0.963 # EQ. 4: http://www.wseas.us/e-library/conferences/2009/cambridge/WHH/WHH06.pdf
    #  else:
    #      w = 0.51*nu/D50*(D50**3*GRAV*R_VAR/nu**2)**0.553 # EQ. 4: http://www.wseas.us/e-library/conferences/2009/cambridge/WHH/WHH06.pdf 
    
    #2)  settling velocity for grains - Rubey (1933)
    F = (2 / 3 + 36 * nu**2 / (GRAV * D50**3 * R_VAR))**0.5 - (36 * nu**2/(GRAV * D50**3 * R_VAR))**0.5
    w = F * (D50 * GRAV * R_VAR)**0.5 #settling velocity
    
    # use corrected sediment diameter
    tau = 1000 * GRAV * h * slope
    vstar = np.sqrt(tau / 1000)
    w50 = (16.17 * D50**2)/(1.8 * 10**(-5) + (12.1275 * D50**3)**0.5)
    
    De = (1.8 * D50) / (1 + 0.8 * (vstar / w50)**0.1 * (GeoStd - 1)**2.2)
    
    U_star = np.sqrt(De * GRAV * slope) # shear velocity 
    
    # 1) Yang Sand Formula
    log_C = 5.165 - 0.153 * np.log10(w * De / nu) - 0.297 * np.log10(U_star / w) + (1.78 - 0.36 * np.log10(w * De / nu) - 0.48 * np.log10(U_star / w)) * np.log10(v * slope / w) 
    
    # 2) Yang Gravel Formula
    #log_C = 6.681 - 0.633*np.log10(w*D50/nu) - 4.816*np.log10(U_star/w) + (2.784-0.305*np.log10(w*D50/nu)-0.282*np.log10(U_star/w))*np.log10(v*slope/w) 
   
    QS_ppm = 10**(log_C) # in ppm 
    
    QS_grams = QS_ppm # in g/m3
    QS_grams_per_sec = QS_grams * Q # in g/s
    QS_kg = QS_grams_per_sec / 1000 # in kg/s
    
    QS_Yang = QS_kg / RHO_S # m3/s
    
    tr_cap = QS_Yang
    
    return tr_cap

def Ackers_White_formula(D50, slope, Q, v, h):
    """
    Returns the value of the transport capacity [m3/s] for each sediment class
    in the reach measured using the Ackers and White equations.
    This function is for use in the D-CASCADE toolbox.
    
    References:
    Stevens Jr., H. H. & Yang, C.T. Summary and use of selected fluvial sediment-discharge formulas. (1989).
    Ackers P., White W.R. Sediment transport: New approach and analysis (1973)
    """
    
    #FR = v/np.sqrt(g*h)     #Froude number
    
    # Ackers - White suggest to use the D35 instead of the D50
    D_AW = D50
    
    nu = 1.003*1E-6 # kinematic viscosity @ 20�C: http://onlinelibrary.wiley.com/doi/10.1002/9781118131473.app3/pdf
    #nu = 0.000011337  # kinematic viscosity (ft2/s)
    
    alpha = 10 # coefficient in the rough turbulent equation with a value of 10;
    
    #conv = 0.3048 #conversion 1 feet to meter
    
    ## transition exponent depending on sediment size [n]
    
    D_gr = D_AW * (GRAV * R_VAR / nu**2)**(1/3) #dimensionless grain size - EB change coding line if D_gr is different from a number 
    
    #shear velocity
    u_ast = np.sqrt(GRAV * h * slope)
    
    ## Transport capacity 
    
    #coefficient for dimensionless transport calculation
    C = 0.025
    m = 1.50    # m = 1.78
    A = 0.17
    n = 0
    
    C = np.matlib.repmat(C, 1, 1) #np.matlib.repmat(m, D_gr.shape[0],D_gr.shape[1])
    m = np.matlib.repmat(m, 1, 1) #np.matlib.repmat(m, D_gr.shape[0],D_gr.shape[1])
    A = np.matlib.repmat(A, 1, 1) #np.matlib.repmat(m, D_gr.shape[0],D_gr.shape[1])
    n = np.matlib.repmat(n, 1, 1) #np.matlib.repmat(m, D_gr.shape[0],D_gr.shape[1])
    
    if np.less(D_gr, 60).any(): 
        C[D_gr < 60] = 10 ** (2.79 * np.log10(D_gr[D_gr < 60]) - 0.98 * np.log10(D_gr[D_gr < 60])**2 - 3.46)
        m[D_gr < 60] = 6.83 / D_gr[D_gr < 60] + 1.67     # m = 9.66 / D_gr(D_gr < 60) + 1.34;
        A[D_gr < 60] = 0.23 / np.sqrt(D_gr[D_gr < 60]) + 0.14
        n[D_gr < 60] = 1 - 0.56 * np.log10(D_gr[D_gr < 60])
    
    ## mobility factor
    F_gr = u_ast**n / np.sqrt(GRAV * D_AW * R_VAR) * (v / (np.sqrt(32) * np.log10(alpha * h / D_AW)))**(1 - n)
     
    # dimensionless transport
    G_gr = C * (np.maximum(F_gr / A - 1, 0) )**m
    
    # weight concentration of bed material (Kg_sed / Kg_water)
    QS_ppm = G_gr * (R_VAR + 1) * D_AW * (v / u_ast)**n / h
    
    # transport capacity (Kg_sed / s)
    QS_kg = RHO_W * Q * QS_ppm
    
    # transport capacity [m3/s]
    QS_AW = QS_kg / RHO_S
    
    tr_cap = QS_AW
    
    return tr_cap
    
def GSD_std(Fi_r, dmi):
    """
    Calculates the geometric standard deviation of input X, using the formula
    std = sqrt(D84/D16).
    
    The function finds D84 and D16 by performing a liner interpolation
    between the known points of the grain size distribution (GSD).
    """
    
    #calculates GSD_std
    D_values = [16 , 84]
    D_changes = np.zeros((1, len(D_values)))
    Perc_finer = np.zeros((len(dmi),1))
    Perc_finer[0,:] = 100
    
    for i in range(1, len(Perc_finer)): 
        Perc_finer[i,0] = Perc_finer[i-1,0]-(Fi_r[i-1]*100)

    for i in range(len(D_values)):
        a = np.minimum(np.argwhere(Perc_finer >  D_values[i])[-1][0],len(dmi)-2) 
        D_changes[0,i] = (D_values[i] - Perc_finer[a+1])/(Perc_finer[a] - Perc_finer[a+1])*(dmi[a]-dmi[a+1])+dmi[a+1]
        D_changes[0,i] = D_changes[0,i]*(D_changes[0, i]>0) + dmi[-1]*(D_changes[0,i]<0)
    
    std = np.sqrt(D_changes[0,1]/D_changes[0,0])
    
    return std
    
def Wong_Parker_formula(D50, slope, wac, h):
    """
    Returns the value of the transport capacity [m3/s] for each sediment class
    in the reach measured using the Wong-Parker equations. 
    This function is for use in the D-CASCADE toolbox.
    
    References:
    Wong, M., and G. Parker (2006), Reanalysis and correction of bed-load relation
    of Meyer-Peter and M�uller using their own database, J. Hydraul. Eng., 132(11),
    1159�1168, doi:10.1061/(ASCE)0733-9429(2006)132:11(1159).
    """

    # Wong_Parker parameters 
    alpha = 3.97   # alpha = 4.93
    beta = 1.5     # beta = 1.6
    tauC = 0.0495  # tauC = 0.0470
    
    # dimensionless shear stress
    tauWP = (slope * h) / (R_VAR * D50)
    # dimensionless transport capacity
    qWP = alpha* (np.maximum(tauWP - tauC, 0))**beta
    # dimensionful transport capacity [m3/(s*m)] 
    qWP_dim = qWP * np.sqrt(R_VAR * GRAV * D50**3) # [m3/(s*m)] (formula from the original cascade paper)
    
    QS_WP = qWP_dim * wac # [m3/s]
    
    tr_cap = QS_WP # [m3/s]

    return tr_cap

def Rickenmann_formula(D50, slope, Q, wac):
    """
    Rickenmann's formula for bedload transport capacity based on unit discharge
    for slopes of 0.04% - 20%.
    Critical discharge Qc is based on the equation probosed by Barthust et al. (1987)
    and later refined by Rickenmann (1991).
    
    References: 
    Rickenmann (2001). Comparison of bed load transport in torrents and gravel bed streams. 
    Water Resources Research 37(12): 3295–3305.DOI: 10.1029/2001WR000319.
    Barthust et al. (1987). Bed load discharge equations for steep mountain rivers. 
    In Sediment Transport in Gravel-Bed Rivers, Wiley: New York; 453–477
    Rickenmann (1991). Hyperconcentrated flow and sediment transport at steep flow. 
    Journal ofHydraulic Engineering 117(11): 1419–1439. DOI: 10.1061/(ASCE)0733-9429(1991)117:11(1419)
    """
    
    # Q is on whole width, Qunit = Q/w
    Qunit = Q / wac
    
    exponent_e = 1.5
    # critical unit discharge
    Qc = 0.065 * (R_VAR ** 1.67) * (GRAV ** 0.5) * (D50 ** exponent_e) * (slope ** (-1.12))

    #Check if Q is smaller than Qc
    Qarr = np.full_like(Qc, Qunit)
    
    Qb = np.zeros_like(Qc)
    
    condition = (Qarr - Qc) < 0
    Qb = np.where(condition, 0, 1.5 * (Qarr - Qc) * (slope ** 1.5))

    Qb_Wac = Qb * wac
    tr_cap = Qb_Wac

    return tr_cap, Qc

def Molinas_rates(fi_r, h, v, slope, dmi_finer, D50_finer):
    """
    Returns the Molinas coefficient of fractional transport rates pci, to be
    multiplied by the total sediment load to split it into different classes.
    
    References:   
    Molinas, A., & Wu, B. (2000). Comparison of fractional bed material load
    computation methods in sand-bed channels. Earth Surface Processes and
    Landforms: The Journal of the British Geomorphological Research Group.
    """
    
    # Molinas and wu coefficients 
    # Molinas requires D50 and dmi in mm
                  
    # Hydraulic parameters in each flow percentile for the current reach
    Dn = (1 + (GSD_std(fi_r, dmi_finer) - 1)**1.5) * D50_finer # scaling size of bed material
    
    tau = 1000 * GRAV * h * slope
    vstar = np.sqrt(tau / 1000);
    froude = v / np.sqrt(GRAV * h)     # Froude number
    
    # alpha, beta, and zeta parameter for each flow percentile (columns), and each grain size (rows)
    # EQ 24 , 25 , 26 , Molinas and Wu (2000)
    alpha = - 2.9 * np.exp(-1000 * (v / vstar)**2 * (h / D50_finer)**(-2))
    beta = 0.2 * GSD_std(fi_r, dmi_finer)
    zeta = 2.8 * froude**(-1.2) *  GSD_std(fi_r, dmi_finer)**(-3) 
    zeta[np.isinf(zeta)] == 0 #zeta gets inf when there is only a single grain size. 
    
    # alpha, beta, and zeta parameter for each flow percentile (columns), and each grain size (rows)
    # EQ 17 , 18 , 19 , Molinas and Wu (2003)  
    # alpha = - 2.85* exp(-1000*(v/vstar)^2*(h/D50)^(-2));
    # beta = 0.2* GSD_std(fi_r,dmi);
    # zeta = 2.16*froude^(-1);
    # zeta(isinf(zeta)) = 0; 
    
    # fractioning factor for each flow percentile (columns), and each grain size (rows) 
    frac1 = fi_r * ((dmi_finer / Dn)**alpha + zeta * (dmi_finer / Dn)**beta) # Nominator in EQ 23, Molinas and Wu (2000) 
    pci = frac1 / np.sum(frac1)
        
    return pci  


def choose_formula(Fi_r_reach, D50, slope, Q, wac, v, h, psi, indx_tr_cap): 
    """
    Chooses the transport capacity function based on the given index (indx_tr_cap).
    """
    # EB D50 entries change according to formula index - it would be good to create a class 
    # to call with the string name of the formula 

    tau = np.nan
    taur50 = np.nan
    Qc = np.nan
 
    #choose transport capacity formula
    if indx_tr_cap == 1:
        [tr_cap, tau, taur50] = Parker_Klingeman_formula(Fi_r_reach, D50, slope, wac, h, psi)
    
    elif indx_tr_cap == 2:
        [tr_cap, tau, taur50] =  Wilcock_Crowe_formula(Fi_r_reach, D50, slope, wac, h, psi)
    
    elif indx_tr_cap == 3: 
        tr_cap = Engelund_Hansen_formula(D50, slope, wac, v, h)
    
    elif indx_tr_cap == 4: 
        tr_cap = Yang_formula(Fi_r_reach, D50, slope, Q, v, h, psi) 
    
    elif indx_tr_cap == 5: 
        tr_cap = Wong_Parker_formula(D50, slope, wac, h)
    
    elif indx_tr_cap == 6: 
        tr_cap = Ackers_White_formula(D50, slope, Q, v, h)        
    
    elif indx_tr_cap == 7:
        tr_cap, Qc = Rickenmann_formula(D50, slope, Q, wac)
    
    return tr_cap, Qc

def tr_cap_function(Fi_r_reach, D50, slope, Q, wac, v, h, psi, indx_tr_cap, indx_partition):
    """
    Refers to the transport capacity equation and partitioning 
    formula chosen by the  user and return the value of the transport capacity 
    and the relative Grain Size Distrubution (pci) for each sediment class in the reach.
    """  
    dmi = 2**(-psi) / 1000 #sediment classes diameter (m)
        
    ##choose partitioning formula for computation of sediment transport rates for individual size fractions
    
    #formulas from: 
    #Molinas, A., & Wu, B. (2000): Comparison of fractional bed material load computation methods in sand?bed channels. 
    #Earth Surface Processes and Landforms: The Journal of the British Geomorphological Research Group

    Qtr_cap = np.zeros(len(psi))[None]
    
    if indx_partition == 1: # Direct computation by the size fraction approach  
        
        Qtr_cap,Qc = choose_formula(Fi_r_reach, dmi, slope, Q, wac, v, h, psi, indx_tr_cap)
        pci = Fi_r_reach
        
    elif indx_partition == 2: # The BMF approach (Bed Material Fraction)
        tr_cap, Qc =  choose_formula(Fi_r_reach, dmi, slope, Q, wac, v, h, psi, indx_tr_cap)
        Qtr_cap = Fi_r_reach*tr_cap
        pci = Fi_r_reach 
        
    elif indx_partition == 3: # The TCF approach (Transport Capacity Fraction) with the Molinas formula (Molinas and Wu, 2000)
        pci = Molinas_rates(Fi_r_reach, h, v, slope, dmi*1000, D50*1000)
        tr_cap, Qc = choose_formula(Fi_r_reach, D50, slope, Q, wac, v, h, psi, indx_tr_cap)
        Qtr_cap = pci * tr_cap
    
    elif indx_partition == 4: #Shear stress correction approach (for fractional transport formulas)
        tr_cap, Qc = choose_formula(Fi_r_reach, D50, slope, Q, wac, v, h, psi, indx_tr_cap)
        Qtr_cap = tr_cap #these formulas returns already partitioned results;
        pci = Qtr_cap / np.sum(Qtr_cap)
      
    return Qtr_cap, Qc

def sed_velocity(hVel, wac, tr_cap_per_s, phi, indx_velocity, minvel):
    """
    This function compute the sediment velocity (in m/s), for each sediment 
    classes of a given reach n. This calculation is directly done from the 
    estimated flux (tr_cap) in m3/s, and by dividing it by a section (Active width x transport height). 
    This function directly impacts the path lengths. 
    
    INPUTS:
    hVel:           height of the total section that we choose for infering velocity from a flux.(reaches x classes) 
    wac:            Active width of the reach
    tr_cap_per_s:   transport capacity of each sediment class in the reach (m3/s)
    phi:            sediment porosity
    indx_velocity:  index for choosing the method.
            1- The same velocity is assigned to all classes, based on the total flux. 
            Even sediment classes that have 0 fluxes are given a velocity, 
            in order to make travel sediment of this class that are passing through
            Conceptually, providing the same velocity to all classes, 
            is like weighting the section of this class i by its fraction in the flux. 
            (Si=Stot*Qi/Qtot)
            2- The class velocity is proportional to the flux. 
            Sediment class with 0 flux will have 0 velocity, which can be a 
            problem for sediments of this class that are passing trough..
            The section is divided by the number of classes, equally ditributing
            vertical space among classes.
    minvel:         minimum value for velocity
    
    RETURN:
    v_sed_n:        velocities per class of a given reach.
    """
    Svel = hVel * wac * (1 - phi)  # the global section where all sediments pass through
    if indx_velocity == 1:                     
        v_sed_n = np.sum(tr_cap_per_s) / Svel     # same velocity for each class
        v_sed_n = np.maximum(v_sed_n , minvel)    # apply the min vel threshold
        v_sed_n = np.full(len(tr_cap_per_s), v_sed_n) # put the same value for all classes
    elif indx_velocity == 2:
        Si = Svel / len(tr_cap_per_s)             # same section for all sediments
        v_sed_n = np.maximum(tr_cap_per_s/Si , minvel)
    return v_sed_n

def sed_velocity_OLD(Fi_r, slope_t, Q_t, wac_t, v, h, psi, minvel, phi, indx_tr_cap, indx_partition, indx_velocity): 
    """
    Function for calculating velocities.
    OLD method that recalculate the transport capacity in each reach, 
    using the Fi_r and D50 of the mobilised volume in reach n. 
    The velocity in each reach (in m/s) is then estimated by dividing the new transport
    capacity by a section.
    
    INPUTS:
    Fi_r: sediment fractions per class in each reach
    slope_t: All reaches slopes at time step t
    Q_t: All reaches' discharge at time step t
    wac_t: All reaches' discharge at time step t
    v: All reaches' water velocities
    h: All reaches' water heights 
    minvel: minimum velocity threshold
    phi: sediment size class
    
    OUTPUTS:
    v_sed: [cxn] matrix reporting the velocity for each sediment class c for each reach n [m/s]"""
    
    #active layer definition 
    #active layer as 10% of the water column depth  
    l_a = 0.1 * h #characteristic vertical length scale for transport.
    
    #alternative: active layer as 2*D90 (Parker, 2008)
    #l_a = 2*D_finder_3(Fi_r_reach, 90 )
    
    #D50 definition
    dmi = 2**(-psi)/1000  #grain size classes[m]
    
    # find D values 
    D50 = D_finder(Fi_r, 50, psi )
    
    ## sediment velocity with fractional trasport capacity

    #  by measuring the trasport capacity for the single sed.classes
    #  indipendenty, we obtain different values of sed. velocity
    
    if indx_velocity == 3:
    
        #choose transport capacity formula
        if indx_tr_cap == 1:

            # run tr_cap function independently for each class 
            tr_cap = np.zeros((len(dmi), len(slope_t)))
            # ... run the tr.cap function indipendently for each class, setting
            # the frequency of each class = 1
            for d in range(len(dmi)): 
                Fi_r = np.zeros((len(dmi),1))
                Fi_r[d] = 1
                Fi_r = np.matlib.repmat(Fi_r,1,len(slope_t))
                tr_cap_class = Parker_Klingeman_formula(Fi_r,dmi[d], slope_t, wac_t , h)
                tr_cap[d,:] = tr_cap_class[d,:]
        
        elif indx_tr_cap == 2: 
            # run tr_cap function independently for each class 
            tr_cap = np.zeros((len(dmi), len(slope_t)))
            # ... run the tr.cap function indipendently for each class, setting
            # the frequency of each class = 1
            Fi_r = np.diag(np.full(len(dmi),1))
            Fi_r = np.repeat(Fi_r[:,:,np.newaxis], len(slope_t), axis = 2)
            for d in range(len(dmi)): 
                [tr_cap_class, tau, taur50] = Wilcock_Crowe_formula(Fi_r[d,:,:], dmi[d], slope_t, wac_t , h, psi)
                tr_cap[d,:] = tr_cap_class[d,:]
                
            """Fi_r = np.ones((len(dmi), len(slope_t)))
            [tr_cap_class, tau, taur50] = Wilcock_Crowe_formula(Fi_r, dmi, slope_t, wac_t , h, psi)
            tr_cap[d,:] = tr_cap_class[d,:]"""
            
            """for d in range(len(dmi)): 
                Fi_r = np.zeros((len(dmi),1))
                Fi_r[d] = 1
                Fi_r = np.matlib.repmat(Fi_r,1,len(slope_t))
                [tr_cap_class, tau, taur50] = Wilcock_Crowe_formula(Fi_r, dmi[d], slope_t, wac_t , h, psi) 
                tr_cap[d,:] = tr_cap_class[d,:]"""
            
        elif indx_tr_cap == 3: 
            slope_t_v, dmi_v = np.meshgrid(slope_t, dmi, indexing='xy')
            h_v, dmi_v = np.meshgrid(h, dmi, indexing='xy')
            v_v, dmi_v = np.meshgrid(v, dmi, indexing='xy')
            wac_t_v, dmi_v = np.meshgrid(wac_t, dmi, indexing='xy')
            tr_cap = Engelund_Hansen_formula(dmi_v, slope_t_v, wac_t_v, v_v, h_v)
            
        elif indx_tr_cap == 4: 
            tr_cap = Yang_formula(Fi_r, dmi, slope_t, Q_t, v, h, psi)
            
        elif indx_tr_cap == 5: 
            tr_cap = Wong_Parker_formula(dmi, slope_t, wac_t, h)
    
        #calculate velocity
        multiply = (wac_t * l_a * (1-phi)).to_numpy()
        v_sed = np.maximum( tr_cap/( multiply[None,:] ) , minvel)
        v_sed[:, l_a==0] = minvel
        
    ## sediment velocity with total transport capacity
    
    # sediment velocity found in this way is constant for all sed.classes
    if indx_velocity == 4:
        [ Qtr_cap, pci ] = tr_cap_function( Fi_r , D50 ,  slope_t, Q_t, wac_t, v , h, psi, indx_tr_cap , indx_partition)
        v_sed = np.maximum( Qtr_cap/( wac_t * l_a*(1-phi) * pci ) , minvel)
        
    return v_sed
    