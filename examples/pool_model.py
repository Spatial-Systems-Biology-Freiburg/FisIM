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
    a, b = P
    Temp, = Q
    n0, c, n_max = Const
    n, = y
    return [(a*Temp + c) * (n - n0*np.exp(-b*Temp*t))*(1-n/n_max)]

def dfdx(t, y, Q, P, Const):
    a, b = P
    Temp, = Q
    n0, c, n_max = Const
    n, = y
    return (a*Temp + c) * (1-n/n_max) + (a*Temp + c) * (n - n0*np.exp(-b*Temp*t))*(-1/n_max)

def dfdp(t, y, Q, P, Const):
    a, b = P
    Temp, = Q
    n0, c, n_max = Const
    n, = y
    return [
        (Temp) * (n - n0*np.exp(-b*Temp*t))*(1-n/n_max),
        (a*Temp + c) * (Temp*t*n0*np.exp(-b*Temp*t))*(1-n/n_max)
    ]

def dfdx0(t, y, Q, P, Const):
    a, b = P
    Temp, = Q
    n0, c, n_max = Const
    n, = y
    return [
        [(a*Temp + c) * (- np.exp(-b*Temp*t))*(1-n/n_max)]
    ]

def g(t, y, Q, P, Const):
    n0, c, n_max = Const
    n, = y
    return [np.log((n+n0)/n0)]

def dgdx(t, y, Q, P, Const):
    n0, c, n_max = Const
    n, = y
    return [
        [1 / (n+n0)]
    ]

def dgdp(t, y, Q, P, Const):
    return [
        [0, 0]
    ]

def dgdx0(t, y, Q, P, Const):
    n0, c, n_max = Const
    n, = y
    return [
        [- n /(n0*(n+n0))]
    ]


if __name__ == "__main__":
    ###############################
    ### USER DEFINES PARAMETERS ###
    ###############################

    # Define ode_args for the simulation duration
    c = 1.31
    n0 = 0.25
    n_max = 2e4
    Const = (n0, c, n_max)

    # Define initial parameter guesses
    a = 0.065
    b = 1e-3
    P = (a, b)

    # Initial values for complete ODE (with S-Terms)
    t0 = 0.0
    x0 = n0

    # Define bounds for sampling
    temp_low = 2.0
    temp_high = 20.0

    times_low = t0
    times_high = 16.0
    # Discretize explicitly
    # dtimes = [2.0, 4.0, 5.0]
    dtimes = 0.5

    # Construct parameter hyperspace
    n_times = 6
    n_temps = 1

    # Values for temperatures (Q-Values)
    inputs = [
        np.linspace(temp_low, temp_high, n_temps),
        # {"lb": temp_low, "ub": temp_high, "n": n_temps},
    ]
    # Values for times (can be same for every temperature or different)
    # the distinction is made by dimension of array

    fsm = FisherModel(
            ode_fun=pool_model,
            ode_dfdx=dfdx,
            ode_dfdp=dfdp,
            ode_dfdx0=dfdx0,
            ode_t0=[times_low],
            # ode_x0={"lb":0.5, "ub":10.0, "n":2},
            ode_x0=n0,
            times={"lb":times_low, "ub":times_high, "n":n_times},
            inputs=inputs,
            parameters=P,
            ode_args=Const,
            obs_fun=g,
            obs_dgdx=dgdx,
            obs_dgdp=dgdp,
            obs_dgdx0=dgdx0,
            covariance={"abs": 100},
    )

    ###############################
    ### OPTIMIZATION FUNCTION ? ###
    ###############################
    fsr = find_optimal(
        fsm,
        optimization_strategy="scipy_differential_evolution",
        criterion=fisher_determinant,
        relative_sensitivities=True,
        maxiter=200,
        polish=False,
        workers=-1,
    )

    ###############################
    ##### PLOTTING FUNCTION ? #####
    ###############################
    plot_all_solutions(fsr, outdir="out")
    json_dump(fsr, "pool_model.json")
