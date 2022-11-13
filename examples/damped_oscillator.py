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
def damped_osci(t, y, inputs, parameters, ode_args):
    d, l = parameters
    T, H = inputs
    A, B = y
    x_offset, = ode_args
    return [
        - l*T*A - d*H*(B-x_offset),
        A
    ]

def damped_osci_dfdx(t, y, inputs, parameters, ode_args):
    d, l = parameters
    T, H = inputs
    A, B = y
    x_offset, = ode_args
    return [
        [- l*T, - d*H],
        [1, 0]
    ]

def damped_osci_dfdp(t, y, inputs, parameters, ode_args):
    d, l = parameters
    T, H = inputs
    A, B = y
    x_offset, = ode_args
    return [
        [- H*(B-x_offset), - T*A],
        [0, 0]
    ]


if __name__ == "__main__":
    ###############################
    ### USER DEFINES PARAMETERS ###
    ###############################    

    # Define initial parameter guesses
    d = 5e-0
    l = 1e-0
    parameters = (d, l)

    # Define optional arguments for the ODE
    x_offset = 20
    ode_args = (x_offset,)

    # Initial values for complete ODE (with S-Terms)
    t0 = 0.0
    x0 = [6.0, 20.0]

    # Define bounds for sampling
    temp_low = 0.8
    temp_high = 1.2

    hum_low = 0.8
    hum_high = 1.2

    times_low = t0
    times_high = 10.0

    # Construct parameter hyperspace
    n_times = 5
    n_temps = 1
    n_hums = 1

    # Values for temperatures and humidities
    inputs = [
        np.linspace(temp_low, temp_high, n_temps),
        np.linspace(hum_low, hum_high, n_hums),
    ]

    fsm = FisherModel(
            ode_fun=damped_osci,
            ode_dfdx=damped_osci_dfdx,
            ode_dfdp=damped_osci_dfdp,
            ode_t0=times_low,
            ode_x0=x0,
            times=(times_low, times_high, n_times),
            inputs=inputs,
            parameters=parameters,
            ode_args=ode_args,
            covariance=("abs", 2.0),
    )

    ###############################
    ### OPTIMIZATION FUNCTION ? ###
    ###############################
    fsr = find_optimal(
        fsm,
        optimization_strategy="scipy_differential_evolution",
        criterion=fisher_determinant,
        maxiter=100,
        polish=False,
        workers=1,
    )

    ###############################
    ##### PLOTTING FUNCTION ? #####
    ###############################
    plot_all_solutions(fsr, outdir="out")
    json_dump(fsr, "damped_osci.json")
