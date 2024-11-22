# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:03:00 2023

Different formula for the calculation of the tranport capacity and for the assignment of the velocity

@author: Elisa Bozzolan 
"""

import numpy as np
import numpy.matlib
from supporting_functions import D_finder, matrix_compact
from constants import (
    RHO_S,
    RHO_W,
    GRAV,
    R_VAR,
)



class TransportCapacityCalculator:
    def __init__(self, fi_r_reach, D50, slope, Q, wac, v, h, psi, rugosity, gamma=0.05):
    #def __init__(self, fi_r_reach, D50, slope, Q, wac, v, h, psi, gamma=0.05):
        # Dictionary mapping indices to different formula
        self.index_to_formula = {
            1: self.Parker_Klingeman_formula,
            2: self.Wilcock_Crowe_formula,
            3: self.Engelund_Hansen_formula,
            4: self.Yang_formula,
            5: self.Wong_Parker_formula,
            6: self.Ackers_White_formula,
            7: self.Rickenmann_formula
        }
        self.fi_r_reach = fi_r_reach
        self.D50 = D50
        self.slope = slope
        self.Q = Q
        self.wac = wac
        self.v = v
        self.h = h
        self.psi = psi
        self.dmi = 2**(-self.psi) / 1000 # sediment classes diameter [m]
        self.gamma = gamma # hiding factor
        self.rugosity = rugosity
        
    def choose_formula(self, indx_tr_cap):
        """
        Dynamically chooses and calls the transport capacity function based on indx_tr_cap.
        """

        # Look up the function using the index in the dictionary
        formula = self.index_to_formula.get(indx_tr_cap)

        if formula is None:
            raise ValueError(f"No formula associated with index {indx_tr_cap}")

        result = formula()

        # Ensure any missing keys default to np.nan for consistent output
        tr_cap = result.get("tr_cap", np.nan)
        Qc = result.get("Qc", np.nan)
        
        return tr_cap, Qc

    def Parker_Klingeman_formula(self):
        """
        Returns the value of the transport capacity [Kg/s] for each sediment class
        in the reach measured using the Parker and Klingeman equations.
        This function is for use in the D-CASCADE toolbox.
        CHECK THE UNIT OF THE RETURNED VARIABLE.
        
        References:
        Parker and Klingeman (1982). On why gravel bed streams are paved. Water Resources Research.
        """
        
        tau = RHO_W * GRAV * self.h * self.slope # bed shear stress [Kg m-1 s-1]
        
        # tau_r50 formula from Mueller et al. (2005)
        # reference shear stress for the mean size of the bed surface sediment [Kg m-1 s-1]
        tau_r50 = (0.021 + 2.18 * self.slope) * (RHO_W * R_VAR * GRAV * self.D50)
        
        tau_ri = tau_r50 * (self.dmi/self.D50)** self.gamma # reference shear stress for each sediment class [Kg m-1 s-1]
        phi_ri = tau / tau_ri
        
        # Dimensionless transport rate for each sediment class [-]
        w_i = 11.2 * (np.maximum(1 - 0.853/phi_ri, 0))**4.5
        
        # Dimensionful transport rate for each sediment class [m3/s]
        tr_cap = self.wac * w_i * self.fi_r_reach * (tau / RHO_W)**(3/2) / (R_VAR * GRAV)
        tr_cap[np.isnan(tr_cap)] = 0 #if Qbi_tr are NaN, they are put to 0
        
        return {"tr_cap": tr_cap}
        
    def Wilcock_Crowe_formula(self): 
        """
        Returns the value of the transport capacity [m3/s] for each sediment class
        in the reach measured using the Wilcock and Crowe equations.
        This function is for use in the D-CASCADE toolbox.
        
        References:
        Wilcock, Crowe(2003). Surface-based transport model for mixed-size sediment. Journal of Hydraulic Engineering.
        """
        
        if self.fi_r_reach.ndim == 1:
            self.fi_r_reach = self.fi_r_reach[:,None]
            # Fraction of sand in river bed (sand considered as sediment with phi > -1)
            Fr_s = np.sum((self.psi > - 1)[:,None] * 1 * self.fi_r_reach)
        else:
            Fr_s = np.sum((self.psi > - 1)[:,None] * 1 * self.fi_r_reach, axis = 0)[None,:]
        ## Transport capacity from Wilcock-Crowe equations
    
        tau = np.array(RHO_W * GRAV * self.h * self.slope) # bed shear stress [Kg m-1 s-1]
        if tau.ndim != 0:
            tau = tau[None,:] # add a dimension for computation
        
        # reference shear stress for the mean size of the bed surface sediment [Kg m-1 s-1]
        tau_r50 = (0.021 + 0.015 * np.exp(-20 * Fr_s)) * (RHO_W * R_VAR * GRAV * self.D50)
        
        b = 0.67 / (1 + np.exp(1.5 - self.dmi / self.D50)) # hiding factor
        
        fact = (self.dmi / self.D50)**b
        tau_ri = tau_r50 * fact[:,None] # reference shear stress for each sediment class [Kg m-1 s-1]
        
        phi_ri = tau / tau_ri
        # Dimensionless transport rate for each sediment class [-]
        # The formula changes for each class according to the phi_ri of the class
        # is higher or lower then 1.35.
        W_i = (phi_ri >= 1.35) * (14 * (np.maximum(1 - 0.894 / np.sqrt(phi_ri), 0))**4.5) + (phi_ri < 1.35) * (0.002 * phi_ri**7.5)
        
        # Dimensionful transport rate for each sediment class [m3/s]
        if self.wac.ndim == 0:
            tr_cap = self.wac * W_i * self.fi_r_reach * (tau / RHO_W)**(3/2) / (R_VAR * GRAV)
            if tr_cap.ndim > 1:
               tr_cap = np.squeeze(tr_cap) # EB: a bit of a mess here with dimensions, corrected a posteriori. I want a 1-d vector as output 
        else:
            self.wac = np.array(self.wac)[None, :]
            tr_cap = self.wac * W_i * self.fi_r_reach * (tau / RHO_W)**(3/2) / (R_VAR * GRAV)
        
        tr_cap[np.isnan(tr_cap)] = 0 #if Qbi_tr are NaN, they are put to 0
    
        return {"tr_cap": tr_cap}
    
    def Engelund_Hansen_formula(self):
        """
        Returns the value of the transport capacity [m3/s] for each sediment class
        in the reach measured using the Engelund and Hansen equations.
        This function is for use in the D-CASCADE toolbox.
        
        WARNING: Engelund and Hansen use a factor of 0.1 and but this function uses
        a factor of 0.05.
        
        slope: All reaches' slopes
        h: All reaches' water heights
        
        References:
        Engelund, F., and E. Hansen (1967), A Monograph on Sediment Transport in 
        Alluvial Streams, Tekniskforlag, Copenhagen.
        """
        
        # Friction factor (Eq. 3.1.3 of the monograph)
        C = (2 * GRAV * self.slope * self.h) / self.v**2
    
        # Dimensionless shear stress (Eq. 3.2.3)
        tau_eh = (self.slope * self.h) / (R_VAR * self.D50)
        # Dimensionless transport capacity (Eq. 4.3.5), although Engelund and Hansen
        # find a factor of 0.1 and not 0.05.
        q_eh = 0.05 / C * tau_eh**(5/2)
        # Dimensionful transport capacity per unit width [m3/(s*m)]
        # (page 56 of the monograph)
        q_eh_dim = q_eh * np.sqrt(R_VAR * GRAV * self.D50**3) # m3/s
        # Dimensionful transport capacity [m3/s]
        tr_cap = q_eh_dim * self.wac
        
        return {"tr_cap": tr_cap}
    
    def Yang_formula(self): 
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
    
        nu = 1.003*1E-6        # kinematic viscosity @ : http://onlinelibrary.wiley.com/doi/10.1002/9781118131473.app3/pdf
        
        GeoStd = GSD_std(self.fi_r_reach, self.dmi);
        
        #  1) settling velocity for grains - Darby, S; Shafaie, A. Fall Velocity of Sediment Particles. (1933)
        #         
        #  Dgr = D50*(GRAV*R_VAR/nu**2)**(1/3);
        #     
        #  if Dgr<=10:  
        #      w = 0.51*nu/D50*(D50**3*GRAV*R_VAR/nu**2)**0.963 # EQ. 4: http://www.wseas.us/e-library/conferences/2009/cambridge/WHH/WHH06.pdf
        #  else:
        #      w = 0.51*nu/D50*(D50**3*GRAV*R_VAR/nu**2)**0.553 # EQ. 4: http://www.wseas.us/e-library/conferences/2009/cambridge/WHH/WHH06.pdf 
        
        #2)  settling velocity for grains - Rubey (1933)
        F = (2 / 3 + 36 * nu**2 / (GRAV * self.D50**3 * R_VAR))**0.5 - (36 * nu**2/(GRAV * self.D50**3 * R_VAR))**0.5
        w = F * (self.D50 * GRAV * R_VAR)**0.5 #settling velocity
        
        # use corrected sediment diameter
        tau = 1000 * GRAV * self.h * self.slope
        vstar = np.sqrt(tau / 1000)
        w50 = (16.17 * self.D50**2)/(1.8 * 10**(-5) + (12.1275 * self.D50**3)**0.5)
        
        De = (1.8 * self.D50) / (1 + 0.8 * (vstar / w50)**0.1 * (GeoStd - 1)**2.2)
        
        U_star = np.sqrt(De * GRAV * self.slope) # shear velocity 
        
        # 1) Yang Sand Formula
        log_C = 5.165 - 0.153 * np.log10(w * De / nu) - 0.297 * np.log10(U_star / w) \
        + (1.78 - 0.36 * np.log10(w * De / nu) - 0.48 * np.log10(U_star / w)) * np.log10(self.v * self.slope / w) 
        
        # 2) Yang Gravel Formula
        #log_C = 6.681 - 0.633*np.log10(w*D50/nu) - 4.816*np.log10(U_star/w) + (2.784-0.305*np.log10(w*D50/nu)-0.282*np.log10(U_star/w))*np.log10(v*slope/w) 
       
        QS_ppm = 10**(log_C) # in ppm 
        
        QS_grams = QS_ppm # in g/m3
        QS_grams_per_sec = QS_grams * self.Q # in g/s
        QS_kg = QS_grams_per_sec / 1000 # in kg/s
        
        QS_Yang = QS_kg / RHO_S # m3/s
        
        tr_cap = QS_Yang
        
        return {"tr_cap": tr_cap}
    
    def Ackers_White_formula(self):
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
        D_AW = self.D50
        
        nu = 1.003*1E-6 # kinematic viscosity @ 20�C: http://onlinelibrary.wiley.com/doi/10.1002/9781118131473.app3/pdf
        #nu = 0.000011337  # kinematic viscosity (ft2/s)
        
        alpha = 10 # coefficient in the rough turbulent equation with a value of 10;
        
        #conv = 0.3048 #conversion 1 feet to meter
        
        ## transition exponent depending on sediment size [n]
        
        D_gr = D_AW * (GRAV * R_VAR / nu**2)**(1/3) #dimensionless grain size - EB change coding line if D_gr is different from a number 
        
        #shear velocity
        u_ast = np.sqrt(GRAV * self.h * self.slope)
        
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
        F_gr = u_ast**n / np.sqrt(GRAV * D_AW * R_VAR) * (self.v / (np.sqrt(32) * np.log10(alpha * self.h / D_AW)))**(1 - n)
         
        # dimensionless transport
        G_gr = C * (np.maximum(F_gr / A - 1, 0) )**m
        
        # weight concentration of bed material (Kg_sed / Kg_water)
        QS_ppm = G_gr * (R_VAR + 1) * D_AW * (self.v / u_ast)**n / self.h
        
        # transport capacity (Kg_sed / s)
        QS_kg = RHO_W * self.Q * QS_ppm
        
        # transport capacity [m3/s]
        QS_AW = QS_kg / RHO_S
        
        tr_cap = QS_AW
        
        return {"tr_cap": tr_cap}
    
    def Wong_Parker_formula(self):
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
        tauWP = (self.slope * self.h) / (R_VAR * self.D50)
        # dimensionless transport capacity
        qWP = alpha* (np.maximum(tauWP - tauC, 0))**beta
        # dimensionful transport capacity [m3/(s*m)] 
        qWP_dim = qWP * np.sqrt(R_VAR * GRAV * self.D50**3) # [m3/(s*m)] (formula from the original cascade paper)
        
        QS_WP = qWP_dim * self.wac # [m3/s]
        
        tr_cap = QS_WP # [m3/s]
    
        return {"tr_cap": tr_cap}
    
    def Rickenmann_formula(self):
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
        Qunit = self.Q / self.wac
        
        exponent_e = 1.5
        # critical unit discharge
        Qc_Rickenmann = 0.065 * (R_VAR ** 1.67) * (GRAV ** 0.5) * (self.D50 ** exponent_e) * (self.slope ** (-1.12))
        Qc_Lenzi = (GRAV ** 0.5) * (self.D50 ** exponent_e) * (0.745 * (self.D50 / self.rugosity) ** (-0.859))
        Qc = Qc_Lenzi
        
        #Check if Q is smaller than Qc
        Qarr = np.full_like(Qc, Qunit)
        
        Qb = np.zeros_like(Qc)
        
        condition = (Qarr - Qc) < 0
        Qb = np.where(condition, 0, 1.5 * (Qarr - Qc) * (self.slope ** 1.5))
    
        Qb_Wac = Qb * self.wac
        tr_cap = Qb_Wac
    
        return {"tr_cap": tr_cap, "Qc": Qc}

    



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


def tr_cap_function(Fi_r_reach, D50, slope, Q, wac, v, h, psi, rugosity, indx_tr_cap, indx_partition):
#def tr_cap_function(Fi_r_reach, D50, slope, Q, wac, v, h, psi, indx_tr_cap, indx_partition):

    """
    Refers to the transport capacity equation and partitioning 
    formula chosen by the  user and return the value of the transport capacity 
    and the relative Grain Size Distrubution (pci) for each sediment class in the reach.
    """  
        
    ##choose partitioning formula for computation of sediment transport rates for individual size fractions
    
    #formulas from: 
    #Molinas, A., & Wu, B. (2000): Comparison of fractional bed material load computation methods in sand?bed channels. 
    #Earth Surface Processes and Landforms: The Journal of the British Geomorphological Research Group
    
    
    # EB D50 entries change according to formula index - it would be good to create a class 
    # to call with the string name of the formula 
    dmi = 2**(-psi) / 1000 #sediment classes diameter (m)

    Qtr_cap = np.zeros(len(psi))[None]
    
    if indx_partition == 1: # Direct computation by the size fraction approach  
        calculator = TransportCapacityCalculator(Fi_r_reach, dmi, slope, Q, wac, v, h, psi)
        Qtr_cap,Qc = calculator.choose_formula(indx_tr_cap)
        pci = Fi_r_reach
        
    elif indx_partition == 2: # The BMF approach (Bed Material Fraction)
        calculator = TransportCapacityCalculator(Fi_r_reach, dmi, slope, Q, wac, v, h, psi, rugosity)
        #calculator = TransportCapacityCalculator(Fi_r_reach, dmi, slope, Q, wac, v, h, psi)
        tr_cap, Qc = calculator.choose_formula(indx_tr_cap)
        Qtr_cap = Fi_r_reach*tr_cap
        pci = Fi_r_reach 
        
    elif indx_partition == 3: # The TCF approach (Transport Capacity Fraction) with the Molinas formula (Molinas and Wu, 2000)
        pci = Molinas_rates(Fi_r_reach, h, v, slope, dmi*1000, D50*1000)
        calculator = TransportCapacityCalculator(Fi_r_reach, D50, slope, Q, wac, v, h, psi)
        tr_cap, Qc = calculator.choose_formula(indx_tr_cap)
        Qtr_cap = pci * tr_cap
    
    elif indx_partition == 4: #Shear stress correction approach (for fractional transport formulas)
        calculator = TransportCapacityCalculator(Fi_r_reach, D50, slope, Q, wac, v, h, psi)
        tr_cap, Qc = calculator.choose_formula(indx_tr_cap)
        Qtr_cap = tr_cap #these formulas returns already partitioned results;
        pci = Qtr_cap / np.sum(Qtr_cap)
      
    return Qtr_cap, Qc



# def compute_cascades_velocities(reach_cascades_list, 
#                                indx_velocity, indx_velocity_partitioning, hVel,
#                                indx_tr_cap, indx_partition,
#                                reach_width, reach_slope, Q_reach, v, h,
#                                phi, minvel, psi, 
#                                reach_Vdep, active_layer_volume,
#                                roundpar):
    
#     ''' Compute the velocity of the cascades in reach_cascade_list.
#     The velocity must be assessed by re-calculating the transport capacity 
#     in the present reach, considering the effect of the arriving cascade(s).
#     Two methods are proposed to re-evaluated the transport capacity, chosen 
#     by the indx_velocity. 
#     First method: the simplest, we re-calculate the transport capacity on each cascade itself.
#     Second method: we consider the active layer volume, to complete, if needed, 
#     the list of cascade by some reach material. If the cascade volume is more 
#     than the active layer, we consider all the cascade volume.
#     '''

#     if indx_velocity == 1:
#         velocities_list = []
#         for cascade in reach_cascades_list:
#             cascade.velocities = volume_velocities(cascade.volume, 
#                                                    indx_velocity_partitioning, 
#                                                    hVel, phi, minvel, psi,
#                                                    indx_tr_cap, indx_partition,
#                                                    reach_width, reach_slope,
#                                                    Q_reach, v, h)
#             velocities_list.append(cascade.velocities)
#         # In this case, we store the averaged velocities obtained among all the cascades
#         velocities = np.mean(np.array(velocities_list), axis = 0)
            
#     if indx_velocity == 2:
#         # concatenate cascades in one volume, and compact it by original provenance
#         # DD: should the cascade volume be in [m3/s] ?
#         volume_all_cascades = np.concatenate([cascade.volume for cascade in reach_cascades_list], axis=0) 
#         volume_all_cascades = matrix_compact(volume_all_cascades)
        
#         volume_total = np.sum(volume_all_cascades[:,1:])
#         if volume_total < active_layer_volume:
#             _, Vdep_active, _, _ = layer_search(reach_Vdep, active_layer_volume,
#                                     roundpar, Qbi_incoming = volume_all_cascades)
#             volume_all_cascades = np.concatenate([volume_all_cascades, Vdep_active], axis=0) 

#         velocities = volume_velocities(volume_all_cascades, indx_velocity_partitioning, 
#                                        hVel, phi, minvel, psi,
#                                        indx_tr_cap, indx_partition,
#                                        reach_width, reach_slope,
#                                        Q_reach, v, h)
        
#         for cascade in reach_cascades_list:
#             cascade.velocities = velocities
    
#     return velocities
        
            
# def volume_velocities(volume, indx_velocity_partitioning, hVel, phi, minvel, psi,
#                       indx_tr_cap, indx_partition,
#                       reach_width, reach_slope, Q_reach, v, h):
    
#     ''' Compute the velocity of the volume of sediments. The transport capacity [m3/s]
#     is calculated on this volume, and the velocity is calculated by dividing the 
#     transport capacity by a section (hVel x width x (1 - porosity)). 
#     For partionning the section among the different sediment class in the volume, 
#     two methods are proposed. 
#     The first one put the same velocity to all classes.
#     The second divides the section equally among the classes with non-zero transport 
#     capacity, so the velocity stays proportional to the transport capacity of that class.
    
#     '''
#     # Find volume sediment class fractions and D50
#     volume_total = np.sum(volume[:,1:])
#     volume_total_per_class = np.sum(volume[:,1:], axis = 0)
#     sed_class_fraction = volume_total_per_class / volume_total
#     D50 = float(D_finder(sed_class_fraction, 50, psi))
    
#     # Compute the transport capacity
#     [ tr_cap_per_s, pci ] = tr_cap_function(sed_class_fraction, D50,  
#                                        reach_slope, Q_reach, reach_width,
#                                        v , h, psi, 
#                                        indx_tr_cap, indx_partition)
    
#     Svel = hVel * reach_width * (1 - phi)  # the global section where all sediments pass through
#     if Svel == 0:
#         raise ValueError("The section to compute velocities can not be 0.")

#     if indx_velocity_partitioning == 1:
#         velocity_same = np.sum(tr_cap_per_s) / Svel     # same velocity for each class
#         velocity_same = np.maximum(velocity_same , minvel)    # apply the min vel threshold
#         velocities = np.full(len(tr_cap_per_s), velocity_same) # put the same value for all classes
        
#     elif indx_velocity_partitioning == 2:
#         # Get the number of classes that are non 0 in the transport capacity flux:
#         number_with_flux = np.count_nonzero(tr_cap_per_s)
#         if number_with_flux != 0:
#             Si = Svel / number_with_flux             # same section for all sediments
#             velocities = np.maximum(tr_cap_per_s/Si, minvel)
#         else:
#             velocities = np.zeros(len(tr_cap_per_s)) # if transport capacity is all 0, velocity is all 0
#     return velocities





    
    
    
    



# OLD from version 1 (not used)

# def sed_velocity(hVel, wac, tr_cap_per_s, phi, indx_velocity, minvel):
#     """
#     This function compute the sediment velocity (in m/s), for each sediment 
#     classes of a given reach n. This calculation is directly done from the 
#     estimated flux (tr_cap) in m3/s, and by dividing it by a section (Active width x transport height). 
#     This function directly impacts the path lengths. 
    
#     INPUTS:
#     hVel:           height of the total section that we choose for infering velocity from a flux.(reaches x classes) 
#     wac:            Active width of the reach
#     tr_cap_per_s:   transport capacity of each sediment class in the reach (m3/s)
#     phi:            sediment porosity
#     indx_velocity:  index for choosing the method.
#             1- The same velocity is assigned to all classes, based on the total flux. 
#             Even sediment classes that have 0 fluxes are given a velocity, 
#             in order to make travel sediment of this class that are passing through
#             Conceptually, providing the same velocity to all classes, 
#             is like weighting the section of this class i by its fraction in the flux. 
#             (Si=Stot*Qi/Qtot)
#             2- The class velocity is proportional to the flux. 
#             Sediment class with 0 flux will have 0 velocity, which can be a 
#             problem for sediments of this class that are passing trough..
#             The section is divided by the number of classes, equally ditributing
#             vertical space among classes.
#     minvel:         minimum value for velocity
    
#     RETURN:
#     v_sed_n:        velocities per class of a given reach.
#     """
#     Svel = hVel * wac * (1 - phi)  # the global section where all sediments pass through
#     if indx_velocity == 1:                     
#         v_sed_n = np.sum(tr_cap_per_s) / Svel     # same velocity for each class
#         v_sed_n = np.maximum(v_sed_n , minvel)    # apply the min vel threshold
#         v_sed_n = np.full(len(tr_cap_per_s), v_sed_n) # put the same value for all classes
#     elif indx_velocity == 2:
#         Si = Svel / len(tr_cap_per_s)             # same section for all sediments
#         v_sed_n = np.maximum(tr_cap_per_s/Si , minvel)
#     return v_sed_n

# def sed_velocity_OLD(Fi_r, slope_t, Q_t, wac_t, v, h, psi, minvel, phi, indx_tr_cap, indx_partition, indx_velocity): 
#     """
#     Function for calculating velocities.
#     OLD method that recalculate the transport capacity in each reach, 
#     using the Fi_r and D50 of the mobilised volume in reach n. 
#     The velocity in each reach (in m/s) is then estimated by dividing the new transport
#     capacity by a section.
    
#     INPUTS:
#     Fi_r: sediment fractions per class in each reach
#     slope_t: All reaches slopes at time step t
#     Q_t: All reaches' discharge at time step t
#     wac_t: All reaches' discharge at time step t
#     v: All reaches' water velocities
#     h: All reaches' water heights 
#     minvel: minimum velocity threshold
#     phi: sediment size class
    
#     OUTPUTS:
#     v_sed: [cxn] matrix reporting the velocity for each sediment class c for each reach n [m/s]"""
    
#     #active layer definition 
#     #active layer as 10% of the water column depth  
#     l_a = 0.1 * h #characteristic vertical length scale for transport.
    
#     #alternative: active layer as 2*D90 (Parker, 2008)
#     #l_a = 2*D_finder_3(Fi_r_reach, 90 )
    
#     #D50 definition
#     dmi = 2**(-psi)/1000  #grain size classes[m]
    
#     # find D values 
#     D50 = D_finder(Fi_r, 50, psi )
    
#     ## sediment velocity with fractional trasport capacity

#     #  by measuring the trasport capacity for the single sed.classes
#     #  indipendenty, we obtain different values of sed. velocity
    
#     if indx_velocity == 3:
    
#         #choose transport capacity formula
#         if indx_tr_cap == 1:

#             # run tr_cap function independently for each class 
#             tr_cap = np.zeros((len(dmi), len(slope_t)))
#             # ... run the tr.cap function indipendently for each class, setting
#             # the frequency of each class = 1
#             for d in range(len(dmi)): 
#                 Fi_r = np.zeros((len(dmi),1))
#                 Fi_r[d] = 1
#                 Fi_r = np.matlib.repmat(Fi_r,1,len(slope_t))
#                 tr_cap_class = Parker_Klingeman_formula(Fi_r,dmi[d], slope_t, wac_t , h)
#                 tr_cap[d,:] = tr_cap_class[d,:]
        
#         elif indx_tr_cap == 2: 
#             # run tr_cap function independently for each class 
#             tr_cap = np.zeros((len(dmi), len(slope_t)))
#             # ... run the tr.cap function indipendently for each class, setting
#             # the frequency of each class = 1
#             Fi_r = np.diag(np.full(len(dmi),1))
#             Fi_r = np.repeat(Fi_r[:,:,np.newaxis], len(slope_t), axis = 2)
#             for d in range(len(dmi)): 
#                 [tr_cap_class, tau, taur50] = Wilcock_Crowe_formula(Fi_r[d,:,:], dmi[d], slope_t, wac_t , h, psi)
#                 tr_cap[d,:] = tr_cap_class[d,:]
                
#             """Fi_r = np.ones((len(dmi), len(slope_t)))
#             [tr_cap_class, tau, taur50] = Wilcock_Crowe_formula(Fi_r, dmi, slope_t, wac_t , h, psi)
#             tr_cap[d,:] = tr_cap_class[d,:]"""
            
#             """for d in range(len(dmi)): 
#                 Fi_r = np.zeros((len(dmi),1))
#                 Fi_r[d] = 1
#                 Fi_r = np.matlib.repmat(Fi_r,1,len(slope_t))
#                 [tr_cap_class, tau, taur50] = Wilcock_Crowe_formula(Fi_r, dmi[d], slope_t, wac_t , h, psi) 
#                 tr_cap[d,:] = tr_cap_class[d,:]"""
            
#         elif indx_tr_cap == 3: 
#             slope_t_v, dmi_v = np.meshgrid(slope_t, dmi, indexing='xy')
#             h_v, dmi_v = np.meshgrid(h, dmi, indexing='xy')
#             v_v, dmi_v = np.meshgrid(v, dmi, indexing='xy')
#             wac_t_v, dmi_v = np.meshgrid(wac_t, dmi, indexing='xy')
#             tr_cap = Engelund_Hansen_formula(dmi_v, slope_t_v, wac_t_v, v_v, h_v)
            
#         elif indx_tr_cap == 4: 
#             tr_cap = Yang_formula(Fi_r, dmi, slope_t, Q_t, v, h, psi)
            
#         elif indx_tr_cap == 5: 
#             tr_cap = Wong_Parker_formula(dmi, slope_t, wac_t, h)
    
#         #calculate velocity
#         multiply = (wac_t * l_a * (1-phi)).to_numpy()
#         v_sed = np.maximum( tr_cap/( multiply[None,:] ) , minvel)
#         v_sed[:, l_a==0] = minvel
        
#     ## sediment velocity with total transport capacity
    
#     # sediment velocity found in this way is constant for all sed.classes
#     if indx_velocity == 4:
#         [ Qtr_cap, pci ] = tr_cap_function( Fi_r , D50 ,  slope_t, Q_t, wac_t, v , h, psi, indx_tr_cap , indx_partition)
#         v_sed = np.maximum( Qtr_cap/( wac_t * l_a*(1-phi) * pci ) , minvel)
        
#     return v_sed
    