#!/usr/bin/env python3

#################################
# THESE LINES ARE ONLY NEEDED   #
# WHEN FisInMa IS NOT INSTALLED #
# OTHERWISE REMOVE THEM         #
#################################
import os, sys
sys.path.append(os.getcwd())
#################################

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy


# Import custom functions for optimization
from FisInMa import *


# System of equation for pool-model and sensitivities
###############################
### USER DEFINES ODE SYSTEM ###
###############################
def pool_model(t, y, Q, P, Const):
    (a, b, c) = P
    (Temp,H) = Q
    (n0, n_max) = Const
    (n,) = y
    return [(a*Temp + c*H) * (n - n0*np.exp(-b*Temp*t))*(1-n/n_max)]

def dfdx(t, y, Q, P, Const):
    (a, b, c) = P
    (Temp,H) = Q
    (n0, n_max) = Const
    (n,) = y
    return (a*Temp + c*H) * (1-n/n_max) + (a*Temp + c*H) * (n - n0*np.exp(-b*Temp*t))*(-1/n_max)

def dfdp(t, y, Q, P, Const):
    (a, b, c) = P
    (Temp,H) = Q
    (n0, n_max) = Const
    (n,) = y
    return [
        (Temp) * (n - n0*np.exp(-b*Temp*t))*(1-n/n_max),
        (a*Temp + c*H) * (Temp*t*n0*np.exp(-b*Temp*t))*(1-n/n_max),
        (H) * (n - n0*np.exp(-b*Temp*t))*(1-n/n_max)
    ]


if __name__ == "__main__":
    ###############################
    ### USER DEFINES PARAMETERS ###
    ###############################

    # Define ode_args for the simulation duration
    n0 = 0.25
    n_max = 2e4
    Const = (n0, n_max)

    # Define initial parameter guesses
    a = 0.065
    b = 0.01
    c = 1.31

    P = (a, b, c)

    # Initial values for complete ODE (with S-Terms)
    t0 = 0.0
    x0 = n0

    # Define bounds for sampling
    temp_low = 4.0
    temp_high = 21.0

    times_low = t0
    times_high = 16.0

    humidity_low = 0.8
    humidity_high = 1.2

    # Construct parameter hyperspace
    n_times = 4
    n_temps = 2
    n_humidity = 2
    
    # Values for temperatures (Q-Values)
    inputs = [
        np.linspace(temp_low, temp_high, n_temps),
        np.linspace(humidity_low, humidity_high, n_humidity)
    ]
    # Values for times (can be same for every temperature or different)
    # the distinction is made by dimension of array

    fsm = FisherModel(
            ode_fun=pool_model,
            ode_dfdx=dfdx,
            ode_dfdp=dfdp,
            ode_t0=times_low,
            ode_x0=x0,
            times=(times_low, times_high, n_times),
            inputs=inputs,
            parameters=P,
            ode_args=Const,
    )

    fsm2 = copy.deepcopy(fsm)
    fsm2.times = (times_low, times_high, n_times)

    fsmp = FisherModelParametrized.init_from(fsm)

    fsr = calculate_fisher_criterion(fsmp, relative_sensitivities=True)
    print(fsr.criterion)
    for sol in fsr.individual_results:
        print(sol.times)

    ###############################
    ### OPTIMIZATION FUNCTION ? ###
    ###############################
    fsr2 = find_optimal(fsm2, relative_sensitivities=True)# maxiter=2, popsize=30, 
    print(fsr2.criterion)
    for sol in fsr2.individual_results:
        print(sol.times)

    ###############################
    ##### PLOTTING FUNCTION ? #####
    ###############################
    plot_all_solutions(fsr2, outdir="out")
