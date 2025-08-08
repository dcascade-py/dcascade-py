"""
Created on Fri Feb  3 12:03:00 2023

Different formula for the calculation of the tranport capacity and for the assignment of the velocity

@author: Elisa Bozzolan
"""

import numpy as np
import numpy.matlib

from constants import GRAV, R_VAR, RHO_S, RHO_W, SPE_GRAV
from d_finder import D_finder


class TransportCapacityCalculator:
    def __init__(self, fi_r_reach, total_D50, slope, Q, wac, v, h, psi, roughness, gamma = 0.05):
        # Dictionaries mapping indices to different formula
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
        self.total_D50 = total_D50              # total D50 in meters
        self.slope = slope
        self.Q = Q
        self.wac = wac
        self.v = v
        self.h = h
        self.psi = psi
        self.class_D50 = 2**(-self.psi) / 1000  # sediment classes diameter in meters
        self.gamma = gamma                      # hiding factor
        self.roughness = roughness              # D90, D84 or roughness from input reachdata class

        self.D50 = np.nan


    def tr_cap_function(self, indx_tr_cap, indx_partition):
        """
        Refers to the transport capacity equation and partitioning
        formula chosen by the  user and return the value of the transport capacity
        and the relative Grain Size Distrubution (pci) for each sediment class in the reach.
        """

        # Verify compatibility between transport formula and partitionning:
        if indx_tr_cap == 2 and indx_partition != 4:
            raise Exception("W&C formula can only be used with the 'shear stress correction' partitioning")
        if indx_tr_cap != 2 and indx_partition == 4:
            raise Exception("the 'shear stress correction' partitioning can only be used for W&C")


        ##choose partitioning formula for computation of sediment transport rates for individual size fractions
        #formulas from:
        #Molinas, A., & Wu, B. (2000): Comparison of fractional bed material load computation methods in sand?bed channels.
        #Earth Surface Processes and Landforms: The Journal of the British Geomorphological Research Group



        # 1: Direct computation by the size fraction approach
        # 2: The BMF approach (Bed Material Fraction)
        # 3: The TCF approach (Transport Capacity Fraction) with the Molinas formula (Molinas and Wu, 2000)
        # 4: Shear stress correction approach (for fractional transport formulas) (these formulas returns already partitioned results)
        index_to_partitioning = {
            1: self.class_D50,
            2: self.class_D50,
            3: self.total_D50,
            4: self.total_D50,
        }
        self.D50 = index_to_partitioning.get(indx_partition)

        # Qtr_cap = np.zeros(len(self.psi))[None]
        Qtr_cap, Qc = self.choose_formula(indx_tr_cap)

        if indx_partition == 2:
            Qtr_cap = self.fi_r_reach * Qtr_cap

        elif indx_partition == 3:
            pci = self.Molinas_rates(self.class_D50*1000, self.total_D50*1000)
            Qtr_cap = pci * Qtr_cap

        return Qtr_cap, Qc


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

        tau_ri = tau_r50 * (self.class_D50/self.D50)** self.gamma # reference shear stress for each sediment class [Kg m-1 s-1]
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

        # Fraction of sand in river bed (sand considered as sediment with phi > -1)
        Fr_s = np.sum((self.psi > - 1) * self.fi_r_reach)
        ## Transport capacity from Wilcock-Crowe equations

        tau = np.array(RHO_W * GRAV * self.h * self.slope) # bed shear stress [Kg m-1 s-1]
        if tau.ndim != 0:
            tau = tau[None,:] # add a dimension for computation

        # reference shear stress for the mean size of the bed surface sediment [Kg m-1 s-1]
        tau_r50 = (0.021 + 0.015 * np.exp(-20 * Fr_s)) * (RHO_W * R_VAR * GRAV * self.D50)

        # tau_r50 after Mueller et al (2005) as presented in eqn 5 in Bizzi et al (2021)
        # tau_r50 = RHO_W * GRAV * R_VAR * self.D50 * (0.021 + 2.18 * self.slope)

        b = 0.67 / (1 + np.exp(1.5 - self.class_D50 / self.D50)) # hiding factor

        fact = (self.class_D50 / self.D50)**b

        tau_ri = tau_r50 * fact # reference shear stress for each sediment class [Kg m-1 s-1]

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

        slope: All reaches' slopes
        h: All reaches' water heights

        References:
        Engelund, F., and E. Hansen (1967), A Monograph on Sediment Transport in
        Alluvial Streams, Tekniskforlag, Copenhagen.
        Stevens Jr., H. H. & Yang, C.T. Summary and use of selected fluvial sediment-discharge formulas. (1989).
        Naito, K., Ma, H., Nittrouer, J. A., Zhang, Y., Wu, B., Wang, Y., … Parker, G. (2019).
        Extended Engelund–Hansen type sediment transport relation for mixtures based on the sand-silt-bed Lower Yellow River, China. Journal of Hydraulic Research, 57(6), 770–785.
        """

        # Friction factor (Eq. 3.1.3 of the monograph)
        C = (2 * GRAV * self.slope * self.h) / self.v**2

        # Dimensionless shear stress (Eq. 3.2.3) (Schield parameter)
        tau_eh = (self.slope * self.h) / (R_VAR * self.D50)
        # Dimensionless transport capacity (Eq. 4.3.5)
        q_eh = 0.1 / C * tau_eh**(5/2)
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

        GeoStd = self.GSD_std(self.class_D50);

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
        Stevens Jr., H. H. & Yang, C.T. Summary and use of selected fluvial
        sediment-discharge formulas. (1989).
        Ackers P., White W.R. Sediment transport: New approach and analysis (1973)
        Ackers P., Sediment transport in open channels: Ackers and White update (1993)
        """
        # Kinematic viscosity : CREATE A TABLE IN THE CONSTANTS?
        # @ 20�C: http://onlinelibrary.wiley.com/doi/10.1002/9781118131473.app3/pdf
        nu = 1.003*1E-6

        # If we compute the total load based on the bed D50
        # (i.e. self.D50 is a float and not a vector in the case of Molinas rate partitionning (3))
        #, then I force it to be a vector:
        if isinstance(self.D50, float):
            D50_s = np.array([self.D50])
        else:
            D50_s = self.D50

        # Dimensionless grain size (Eq. 59 of Stevens & Yang, 1989):
        # TODO: Ackers - White suggest to use the D35 instead of the D50 (p. 21 of Stevens & Yang, 1989)
        D_gr = D50_s * (GRAV * R_VAR / nu**2)**(1/3)


        # Coefficients for dimensionless transport calculation:
        # n - the transition exponent depending on sediment size.
        # A - the value of the Froude number at nominal initial motion.
        # m - the exponent in the sediment transport function.
        # C - the coefficient in the sediment transport function.

        # Values for the coarse size range with D_gr > 60 (2.5 mm sand size).
        n_coarse = 0       # (Eq. 68 of Stevens & Yang, 1989)
        A_coarse = 0.17    # (Eq. 69 of Stevens & Yang, 1989)
        m_coarse = 1.50    # (Eq. 70 of Stevens & Yang, 1989)  (or 1.78 in Eq. 7 in Ackers (1993))
        C_coarse = 0.025   # (Eq. 71 of Stevens & Yang, 1989)

        n = np.full_like(D_gr, n_coarse)
        A = np.full_like(D_gr, A_coarse)
        m = np.full_like(D_gr, m_coarse)
        C = np.full_like(D_gr, C_coarse)

        # Values for the intermediate size range with D_gr > 1 (0.04 mm silt
        # size) to D_gr = 60 (2.5 mm sand size).
        mask_inter = D_gr <= 60
        if mask_inter.any():
            D_gr_inter = D_gr[mask_inter]
            n_inter = 1 - 0.56 * np.log10(D_gr_inter)     # (Eq. 64, S&Y,1989, and Eq. 9 in Ackers (1993))
            A_inter = 0.14 + 0.23 / np.sqrt(D_gr_inter)   # (Eq. 65, S&Y,1989, and Eq. 10 in Ackers (1993))
            m_inter = 1.67 + 6.83 / D_gr_inter            # (Eq. 11 in Ackers (1993)) (or 1.34 + 9.66 / D_gr_inter in Eq. 66, S&Y,1989)
            C_inter = 10**(- 3.46 + 2.79 * np.log10(D_gr_inter)
                           - 0.98 * np.log10(D_gr_inter)**2 )  # (Eq. 12 in Ackers (1993))  (or C_inter = 10**(2.86 * np.log10(D_gr_inter)
                                                                                              # - np.log10(D_gr_inter)**2 - 3.53 in Eq. 67, S&Y,1989)
            n[mask_inter] = n_inter
            A[mask_inter] = A_inter
            m[mask_inter] = m_inter
            C[mask_inter] = C_inter

        # Shear velocity
        u_ast = np.sqrt(GRAV * self.h * self.slope)
        # Coefficient in the rough turbulent equation
        # (value of 10 in Stevens & Yang, 1989)
        alpha = 10
        # Dimensionless mobility number (Eq. 60 of Stevens & Yang, 1989)
        F_gr = (u_ast**n / np.sqrt(GRAV * self.D50 * R_VAR)) * (self.v / (np.sqrt(32) * np.log10(alpha * self.h / self.D50)))**(1 - n)

        # Dimensionless sediment transport of Ackers and White
        # (Eq. 63 of Stevens & Yang, 1989) TOCHECK: WHY THE MAXIMUM, HERE?
        G_gr = C * (np.maximum(F_gr / A - 1, 0) )**m

        # Weight concentration of bed material discharge [Kg_sed / Kg_water]
        # (Eq. 62 of Stevens & Yang, 1989)
        C_s = G_gr * SPE_GRAV * self.D50 * (self.v / u_ast)**n / self.h

        # Total transport capacity [Kg_sed / s]
        Q_kg = RHO_W * self.Q
        QS_kg = Q_kg * C_s

        # Transport capacity [m3/s]
        QS = QS_kg / RHO_S
        tr_cap = QS

        # If we compute the total load based on the bed D50 (float),
        # I put tr_cap back to be a float:
        if isinstance(self.D50, float):
            tr_cap = tr_cap[0]

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
        Rickenmann, D. (1990). Bedload transport capacity of slurry flows at steep slopes
        (Doctoral dissertation, ETH Zurich).
        Barthust et al. (1987). Bed load discharge equations for steep mountain rivers.
        In Sediment Transport in Gravel-Bed Rivers, Wiley: New York; 453–477
        Rickenmann (1991). Hyperconcentrated flow and sediment transport at steep flow.
        Journal ofHydraulic Engineering 117(11): 1419–1439. DOI: 10.1061/(ASCE)0733-9429(1991)117:11(1419)
        Rickenmann (2001). Comparison of bed load transport in torrents and gravel bed streams.
        Water Resources Research 37(12): 3295–3305.DOI: 10.1029/2001WR000319.
        """

        # Unit discharge Qunit. Q is on whole width, Qunit = Q/w.
        Qunit = self.Q / self.wac

        exponent_e = 1.5
        # critical unit discharge
        Qc_Rickenmann = 0.065 * (R_VAR ** 1.67) * (GRAV ** 0.5) * (self.D50 ** exponent_e) * (self.slope ** (-1.12))
        # Qc_Lenzi = (GRAV ** 0.5) * (self.D50 ** exponent_e) * (0.745 * (self.D50 / self.roughness) ** exponent_k)
        Qc = Qc_Rickenmann

        #Check if Q is smaller than Qc
        Qarr = np.full_like(Qc, Qunit)

        # Bedload transport rate per unit of channel width (Eq. 3 in Rickenmann, 2001)
        condition = (Qarr - Qc) < 0
        Qb = np.where(condition, 0, 1.5 * (Qarr - Qc) * self.slope**1.5)

        tr_cap = Qb * self.wac

        return {"tr_cap": tr_cap, "Qc": Qc}



    def Molinas_rates(self, dmi, total_D50):
        """
        Method for paritionning the transport capacity among sediment size classes.
        Falls into the transport capacity fraction approaches (TCF approaches), i.e. it
        directly distribute the sediment transport rates into size groups through a transport
        capacity distribution function.

        Here, the TC distribution function is from Wu and Molinas (1996),
        described also in Molinas and Wu (2000).

        Input grain sizes must be in mm.


        Input:
            - dmi: sediment class size [mm] - array
            - total_D50: total D50 [mm] - float

        Return:
            - pci: Molinas coefficient of fractional transport rates to be multiplied
            by the total sediment load to split it into different classes.

        References:
        Molinas, A., & Wu, B. (2000). Comparison of fractional bed material load
        computation methods in sand-bed channels. Earth Surface Processes and
        Landforms: The Journal of the British Geomorphological Research Group.

        Wu B, Molinas A. (1996). Modeling of alluvial river sediment transport.
        Proceedings of the International Conference on Reservoir Sedimentation,
        Vol. I, Albertson ML, Molinas A, Hotchkiss R (eds).

        """

        # Hydraulic parameters
        tau = RHO_W * GRAV * self.h * self.slope         # Shear stress

        vstar = np.sqrt(GRAV * self.h * self.slope)     # Shear velocity

        froude = self.v / np.sqrt(GRAV * self.h)        # Froude number

        # Accounting for scaling size of bed material (from Wu et al. (2003)) (DD: better understand why)
        Dn = (1 + (self.GSD_std(dmi) - 1)**1.5) * total_D50

        # Geometric standard deviation of bed material
        gsd_std = self.GSD_std(dmi)

        # Alpha, beta, and zeta parameters (eq 24, 25, 26, in Molinas and Wu (2000))
        alpha = - 2.9 * np.exp(-1000 * (self.v / vstar)**2 * (self.h / total_D50)**(-2))

        beta = 0.2 * gsd_std

        zeta = 2.8 * froude**(-1.2) *  gsd_std**(-3)
        zeta[np.isinf(zeta)] == 0 #zeta gets inf when there is only a single grain size.

        # Fractioning factor for each grain size (rows) (eq 23 in Molinas and Wu (2000))
        frac = self.fi_r_reach * ((dmi / Dn)**alpha + zeta * (dmi / Dn)**beta)
        pci = frac / np.sum(frac)

        return pci


    def GSD_std(self, dmi):
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
            Perc_finer[i,0] = Perc_finer[i-1,0]-(self.fi_r_reach[i-1]*100)

        for i in range(len(D_values)):
            a = np.minimum(np.argwhere(Perc_finer >  D_values[i])[-1][0],len(dmi)-2)
            D_changes[0,i] = (D_values[i] - Perc_finer[a+1])/(Perc_finer[a] - Perc_finer[a+1])*(dmi[a]-dmi[a+1])+dmi[a+1]
            D_changes[0,i] = D_changes[0,i]*(D_changes[0, i]>0) + dmi[-1]*(D_changes[0,i]<0)

        std = np.sqrt(D_changes[0,1]/D_changes[0,0])

        return std





