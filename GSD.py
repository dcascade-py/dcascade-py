#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:07:19 2019

@author: marco tangi
"""
import numpy as np
from scipy.optimize import curve_fit


def GSDcurvefit(D16 , D50 , D84 , psi):
    '''
    %GSDcurvefit fits the Rosin curve to the D16, D50 and D84 values in ReachData to obtain the
    %frequency of the sediment classes.

    % INPUT :
    %
    % D16      = 1xN vector defining the D16 of the N input reaches
    % D50      = 1xN vector defining the D50 of the N input reaches
    % D84      = 1xN vector defining the D84 of the N input reaches
    % psi      = 1xC vector defining the mean grain size of the sediment classes in Krumbein phi (φ) scale
    %----
    % OUTPUT:
    %
    % Fi_r     = NxC matrix reporting the grain size frequency of the N reaches for the C sediment classes.
    % resnorm  = 1xN vector listing the squared norm of the residuals of the fitting
    % par_opt = 1x2 vector containing the optimizad values of the two parameter of the Rosin distribution, k and s
    '''

    ' sediment classes diameter (mm) '

    dmi = 2**(-psi).reshape(-1,1);

    ' problem definition '

    lb = [ np.percentile(dmi,10, method='midpoint') , 0.5 ] # k and s lower bound
    ub = [ np.percentile(dmi,90, method='midpoint') , 2 ] # k and s upper bound

    sed_data = np.array([D16, D50, D84]) * 1000;  #sed size in mm
    sed_perc = np.array([0.16, 0.50, 0.84])

#    options = optimoptions('lsqcurvefit','Display','none');

    ' function definition (Rosin distribution) '
    ' def fun_GSD(par, sed_reach ):'
    ' return 1 - np.exp ( - (sed_reach  / par[0] ) ** par[1] )'

    def fun_GSD(sed_reach , k, s  ):
        return 1 - np.exp ( - (sed_reach  / k ) ** s )

    ' initialization '

    par_opt = np.zeros([sed_data.shape[1],2]);
    resnorm = np.zeros([sed_data.shape[1],2]); #squared norm of the residuals of the fitting

    ' curve fitting '
    #popt, pcov = curve_fit(fun_GSD, sed_data[:,0] , sed_perc, p0 = [sed_data[1,0] ,1 ] , bounds = [lb ,ub]   )

    for i in range(0,np.size(sed_data,1)):
        # if dmi.size <= 2:
            # #DD: in this case, lb[0] and ub[0] are the same and can not be used as lower and upper bounds.
            # # The bounding does not change much the fiting results and could be removed in the future ?
        par_opt[i,:], resnormtot = curve_fit(fun_GSD, sed_data[:,i] , sed_perc, p0 = [sed_data[1,i] ,1 ])
        # else:
            # par_opt[i,:], resnormtot = curve_fit(fun_GSD, sed_data[:,i] , sed_perc, p0 = [sed_data[1,i] ,1 ] , bounds = [lb ,ub]   )
        resnorm[i,:] = [resnormtot[0,0], resnormtot[1,1]]

    'find Fi_r'
    #da semplificare!
    F1s = - (np.flip(dmi.reshape(-1,1)) @ (1/par_opt[:,0].reshape(1,-1)))
    F2s = np.sign(F1s) * np.abs(F1s) ** par_opt[:,1]

    F = 1 - np.exp(F2s.transpose())
    F[:,np.size(F,1)-1] = 1

    Fi_r  = np.flip(np.concatenate(( F[:,0].reshape(1,-1) , np.diff(F).transpose() ) ).transpose(),1)

    return (Fi_r, resnorm, par_opt)


