#!/usr/bin/env python3
"""
Created on Thu Mar 12 18:30:20 2020

@author: marco tangi
"""

import numpy as np


def tr_cap_junction( Fi_r_reach , psi , tr_cap_id , D50 ,  Slope, Q, Wac, v , h ):
    '''
    TR_CAP_JUNCTION refers to the transport capacity equation chose by the
    user and return the value of the transport capacity for each sediment
    class in the reach
    '''
    Fi_r_reach = Fi_r_reach.transpose()

    if tr_cap_id ==1:
        Qtr_cap = Wilcock_Crowe_tr_cap( Fi_r_reach, psi ,D50, Slope, Wac , h)
    elif tr_cap_id ==2:
        Qtr_cap = Engelund_Hansen_tr_cap( Fi_r_reach ,psi,  D50 , Slope , Wac, v , h );
    elif tr_cap_id ==3:
        Qtr_cap = Yang_tr_cap( Fi_r_reach, psi, D50 , Slope , Q, v, h );
    elif tr_cap_id ==4:
        Qtr_cap = Wong_Parker_tr_cap( Fi_r_reach,  psi , D50 ,Slope, Wac ,v ,h );
    elif tr_cap_id ==5:
        Qtr_cap = Parker_Klingeman_tr_cap( Fi_r_reach, psi , D50, Slope, Wac , h);

    return Qtr_cap

'''##########################################################################################'''

def GSD_std(Fi_r_reach , dmi_finer):
    '''
     GSD_std(GSD , dmi) calculates the geometric standard deviation of
     input X, using the formula std = sqrt(D84/D16).

    The function finds D84 and D16 by performing a liner interpolation
    between the known points of the GSD.
    '''
    # calculates GSD_std

    D_values = np.array([16, 84])
    D_changes = np.zeros([np.size(D_values),1])
    Perc_finer= np.zeros([np.size(dmi_finer),1]);
    Perc_finer[0,:]=100;

    for i in range(1,np.size(Perc_finer)) :
        Perc_finer[i]=Perc_finer[i-1]-(Fi_r_reach[i-1]*100)


    for i in range(0, np.size(D_values)) :
        indx = np.min( [np.argwhere( Perc_finer >  D_values[i])[-1,0] , np.size(dmi_finer) -1 ])
        D_changes[i] = (D_values[i] - Perc_finer[indx+1] ) / (Perc_finer[indx] - Perc_finer[indx+1] ) * (dmi_finer[indx] - dmi_finer[indx+1]) + dmi_finer[indx+1]
        D_changes[i] = D_changes[i]*(D_changes[i]>0) + dmi_finer[-1]*(D_changes[i]<0)


    std = np.sqrt(D_changes[1]/D_changes[0])
    return std

'''##########################################################################################'''

def Molinas_rates( Fi_r_reach, h, v, Slope, dmi_finer, D50_finer):
    '''
     MOLINAS-rates returns the Molinas coefficient of fractional transport rates Pci, to be multiplied
     by the total sediment load to split it into different classes.

     references

     Molinas, A., & Wu, B. (2000). Comparison of fractional bed?material load computation methods in sand?bed channels. Earth Surface Processes and Landforms: The Journal of the British Geomorphological Research Group
    '''
    # Molinas and wu coefficients

    # Molinas requires D50 and dmi in mm
    g = 9.81

    # Hydraulic parameters in each flow percentile for the current reach
    Dn = (1+(GSD_std(Fi_r_reach,dmi_finer)-1)**1.5)* D50_finer  #scaling size of bed material

    tau = 1000*9.81*h*Slope
    vstar = np.sqrt(tau/1000)
    FR = v/np.sqrt(g*h)     #Froude number

    # alpha, beta, and Zeta parameter for each flow percentile (columns), and each grain size (rows)
    # EQ 24 , 25 , 26 , Molinas and Wu (2000)
    alpha = - 2.9 * np.exp(-1000*(v/vstar)**2 * (h/D50_finer)**(-2));
    beta = 0.2* GSD_std(Fi_r_reach,dmi_finer);
    Zeta = 2.8*FR**(-1.2) *  GSD_std(Fi_r_reach,dmi_finer)**(-3);
    Zeta[np.isinf(Zeta)] = 0; # Zeta gets inf when there is only a single grain size.

    '''
    % alpha, beta, and Zeta parameter for each flow percentile (columns), and each grain size (rows)
    % EQ 17 , 18 , 19 , Molinas and Wu (2003)
    % alpha = - 2.85* exp(-1000*(v/vstar)^2*(h/D50)^(-2));
    % beta = 0.2* GSD_std(Fi_r,dmi);
    % Zeta = 2.16*FR^(-1);
    % Zeta(isinf(Zeta)) = 0;
    '''

    # fractioning factor for each flow percentile (columns), and each grain size (rows)

    frac1 = Fi_r_reach.squeeze() * ( (dmi_finer / Dn)**alpha + Zeta * (dmi_finer/Dn)**beta )  # Nominator in EQ 23, Molinas and Wu (2000)
    Pci = frac1 / (np.sum(frac1))

    return Pci

'''##########################################################################################'''

def Wilcock_Crowe_tr_cap( Fi_r_reach , psi , D50 , Slope , Wac , h):
    '''

    WILCOCK_CROWE_TR_CAP returns the value of the transport capacity for each sediment
    %lass in the reach measured using the wilcock and crowe equations

    %% references
    Wilcock, Crowe(2003). Surface-based transport model for mixed-size sediment. Journal of Hydraulic Engineering

    '''
    'variables initialization'
    Fi_r_reach = Fi_r_reach.reshape(np.size(D50),np.size(psi)).transpose()

    dmi = (2**(-psi)/1000).reshape(-1,1);

    rho_w = 1000 #water density
    rho_s = 2650 #sediment density
    g = 9.81 #gravity acceleration
    R = rho_s / rho_w - 1 #submerged specific gravity of sediment

    Fr_s = np.sum((psi > - 1) * Fi_r_reach.transpose()) # Fraction of sand in river bed (sand considered as sediment with phi > -1)

    'Transport capacity from Wilcock-Crowe equations'

    b = 0.67 / (1 + np.exp(1.5 - dmi/D50))

    tau = (rho_w * g * h * Slope );
    tau_r50 = (0.021 + 0.0015 * np.exp( -20 * Fr_s ) ) * (rho_w * R * g * D50);

    tau_ri = tau_r50 * (dmi/D50)**b;
    phi_ri = tau/tau_ri

    W_i = (phi_ri >= 1.35 ) * (14 * (np.maximum(1 - 0.894 / np.sqrt(phi_ri),0))**4.5) + (phi_ri < 1.35 ) * (0.002 *(phi_ri)**7.5)
    Qtr_cap = Wac * W_i * Fi_r_reach * rho_s * (tau/rho_w)**(3/2) / (R*g);
    # Qtr_cap[np.isnan(Qtr_cap)] = 0; #if Qbi_tr are NaN, they are put to 0

    return Qtr_cap

'''##########################################################################################'''

def Parker_Klingeman_tr_cap( Fi_r_reach , psi , D50 , Slope , Wac , h):
    '''

    PARKER_KLINGEMAN_TR_CAP returns the value of the transport capacity (in Kg/s)
    for each sediment class in the reach measured using the Parker and
    Klingeman equations

    references
    Parker and Klingeman (1982). On why gravel bed streams are paved. Water Resources Research

    '''
    'variables initialization'
    Fi_r_reach = Fi_r_reach.reshape(np.size(D50),np.size(psi)).transpose()

    dmi = (2**(-psi)/1000).reshape(-1,1);

    gamma = 0.05
    rho_w = 1000 #water density
    rho_s = 2650 #sediment density
    g = 9.81 #gravity acceleration
    R = rho_s / rho_w - 1 #submerged specific gravity of sediment

    'Transport capacity from Wilcock-Crowe equations'

    tau = (rho_w * g * h * Slope )
    tau_r50 = (0.021 + 2.18 * Slope) * (rho_w * R * g * D50);

    tau_ri = tau_r50 * (dmi/D50)** gamma;
    phi_ri = tau/tau_ri;

    W_i = 11.2 * (np.maximum(1-0.853/phi_ri,0))**4.5

    # transport capacity for each class (Kg / s)
    Qtr_cap = (Wac * W_i * Fi_r_reach  ) * rho_s * (tau/rho_w)**(3/2) / (R * g);

    # Qtr_cap[np.isnan(Qtr_cap)] = 0; #if Qbi_tr are NaN, they are put to 0

    return Qtr_cap


'''##########################################################################################'''
def  Engelund_Hansen_tr_cap(Fi_r_reach , psi , D50, Slope , Wac, v, h) :

    '''
    ENGELUND_HANSEN_TR_CAP returns the value of the transport capacity (in Kg/s)
    for each sediment class in the reach measured using the Engelund and Hansen equations

    references
    Engelund, F., and E. Hansen (1967), A Monograph on Sediment Transport in Alluvial Streams, Tekniskforlag, Copenhagen.
    '''

    # Transport capacity from Engelund-Hansen equations
    dmi = 2**(-psi)/1000 #sediment classes diameter (m)

    rho_s = 2650 # sediment densit [kg/m^3]
    rho_w = 1000 # water density [kg/m^3]
    g = 9.81

    #friction factor
    C = (2*g*Slope*h)/(v)**2
    #dimensionless shear stress
    tauEH = (Slope*h)/((rho_s/rho_w-1)*D50)
    #dimensionless transport capacity
    qEH = 0.05/C * (tauEH)**(5/2)
    #dimensionful transport capacity m3/s
    qEH_dim = qEH*np.sqrt((rho_s/rho_w-1)*g*(D50)**3) #m3/s (%formula from the original cascade paper)
    QS_EH = qEH_dim*Wac*rho_s #kg/s

    #then the different sediment transport capacities have to be
    #splitted according to Molinas and saved into the Qbi_tr in
    #order to get the right structure for outputs.

    Pci = Molinas_rates( Fi_r_reach, h, v, Slope, dmi*1000, D50*1000);

    Qtr_cap = Pci*QS_EH

    return Qtr_cap

'''##########################################################################################'''
def Wong_Parker_tr_cap(Fi_r_reach , psi, D50 ,  Slope, Wac,  v, h):
    '''
    WONG_PARKER_TR_CAP returns the value of the transport capacity (in Kg/s)
    for each sediment class in the reach measured using the Wong-Parker
    equations

    % references
    Wong, M., and G. Parker (2006), Reanalysis and correction of bed-load relation of Meyer-Peter and M�uller using their own database, J. Hydraul. Eng., 132(11), 1159�1168, doi:10.1061/(ASCE)0733-9429(2006)132:11(1159).
    '''
    # Transport capacity from Wong-Parker equations

    dmi = 2**(-psi)/1000 #sediment classes diameter (m)
    rho_s = 2600 # sediment densit [kg/m^3]
    rho_w = 1000 # water density [kg/m^3]
    g = 9.81

    #Wong_Parker parameters

    alpha = 3.97
    beta = 1.5
    tauC = 0.0495
    '''
     alpha = 4.93;
     beta = 1.6;
     tauC = 0.0470;
    '''

    #dimensionless shear stress
    tauWP = (Slope*h)/((rho_s/rho_w-1)*D50)
    #dimensionless transport capacity
    qWP = alpha* (np.maximum(tauWP - tauC,0) )**(beta)
    #dimensionful transport capacity m3/s
    qWP_dim = qWP * np.sqrt((rho_s/rho_w-1)* g * (D50)**3) # m3/s (%formula from the original cascade paper)
    QS_WP = qWP_dim * Wac * rho_s #kg/s

    #The different sediment transport capacities have to be
    #splitted according to Molinas and saved into the Qbi_tr in
    #order to get the right structure for outputs.

    Pci = Molinas_rates( Fi_r_reach, h, v, Slope, dmi*1000, D50*1000);

    Qtr_cap = Pci * QS_WP;

    return Qtr_cap
'''##########################################################################################'''

def Yang_tr_cap(Fi_r_reach, psi , D50,  Slope , Q, v, h) :
    '''
    %YANG_TR_CAP returns the value of the transport capacity (in Kg/s) for each
    %sediment class in the reach measured using the Yang equations

    %% references
    % Stevens Jr., H. H. & Yang, C. T. Summary and use of selected fluvial sediment-discharge formulas. (1989).
    % see also: Modern Water Resources Engineering: https://books.google.com/books?id=9rW9BAAAQBAJ&pg=PA347&dq=yang+sediment+transport+1973&hl=de&sa=X&ved=0ahUKEwiYtKr72_bXAhVH2mMKHZwsCdQQ6AEILTAB#v=onepage&q=yang%20sediment%20transport%201973&f=false
    '''
    # Transport capacity from Yang equations
    dmi = 2**(-psi)/1000 #sediment classes diameter (m)

    nu = 1.003*1E-6 # kinematic viscosity @ 20�C: http://onlinelibrary.wiley.com/doi/10.1002/9781118131473.app3/pdf
    rho_s = 2650 #sediment densit [kg/m^3]
    rho_w = 1000 # water density [kg/m^3]
    R = rho_s/rho_w - 1 # Relative sediment density []
    g = 9.81

    GeoStd = GSD_std(Fi_r_reach,dmi)
    '''
    %   1) settling velocity for grains - Darby, S; Shafaie, A. Fall Velocity of Sediment Particles. (1933)
    %
    %       Dgr = D50*(g*R/nu^2).^(1/3);
    %
    %       if Dgr<=10
    %            w = 0.51*nu/D50*(D50^3*g*R/nu^2)^0.963; % EQ. 4: http://www.wseas.us/e-library/conferences/2009/cambridge/WHH/WHH06.pdf
    %       else
    %            w = 0.51*nu/D50*(D50^3*g*R/nu^2)^0.553; % EQ. 4: http://www.wseas.us/e-library/conferences/2009/cambridge/WHH/WHH06.pdf
    %       end
    '''
    #   2)  settling velocity for grains - Rubey (1933)
    F = (2/3 + 36*nu**2/(g*D50**3*R))**0.5 - (36*nu**2/(g*D50**3*R))**0.5
    w = F*(D50*g*(R))**0.5  # settling velocity

    #use corrected sediment diameter
    tau = 1000*g*h*Slope
    vstar = np.sqrt(tau/1000)
    w50 = (16.17*(D50)**2)/(1.8*10**(-5)+(12.1275*(D50)**3)**0.5)

    De = (1.8*D50)/(1+0.8*(vstar/w50)**0.1*(GeoStd-1)**2.2)

    U_star = np.sqrt(De*g*Slope)  #shear velocity

    #1)Yang Sand Formula
    log_C = 5.165 - 0.153 * np.log10(w*De/nu) - 0.297 * np.log10(U_star/w) + (1.78 - 0.36 * np.log10(w*De/nu) - 0.48 * np.log10(U_star/w)) * np.log10(v*Slope/w)

    #2)Yang Gravel Formula
    #log_C = 6.681 - 0.633 * np.log10(w*D50/nu) - 4.816 * np.log10(U_star/w) + (2.784 - 0.305 * np.log10(w*D50/nu) - 0.282 * np.log10(U_star/w)) * np.log10(v*Slope/w)

    QS_ppm = 10**(log_C) # in ppm

    QS_grams = QS_ppm  # in g/m3
    QS_grams_per_sec = QS_grams*Q # in g/s
    QS_Yang = QS_grams_per_sec/1000 #kg/s

    #Pci contains the fraction rates for each sediment class
    Pci = Molinas_rates( Fi_r_reach, h, v, Slope, dmi*1000, D50*1000);

    Qtr_cap = Pci * QS_Yang;

    return Qtr_cap


