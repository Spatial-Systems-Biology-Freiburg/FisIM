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


# Import custom package to solve such fisher models
from FisInMa import *


# System of equation for pool-model and sensitivities
###############################
### USER DEFINES ODE SYSTEM ###
###############################
def exp_growth(t, y, Q, P, Const):
    (a,) = P
    (Temp,) = Q
    (n_max,) = Const
    (n,) = y
    return [Temp * a * (n_max - n)]

def dedx(t, y, Q, P, Const):
    (a,) = P
    (Temp,) = Q
    (n_max,) = Const
    (n,) = y
    return [[- Temp * a]]

def dedp(t, y, Q, P, Const):
    (a,) = P
    (Temp,) = Q
    (n_max,) = Const
    (n,) = y
    return [[- Temp * n]]


if __name__ == "__main__":
    ###############################
    ### USER DEFINES PARAMETERS ###
    ###############################

    # Define ode_args for the simulation duration
    n_max = 2e4
    ode_args = (n_max,)

    # Define initial parameter guesses
    a = 0.065
    parameters = (a,)

    # Define bounds for sampling
    temp_low = 3.0
    temp_high = 8.0

    # Define bounds for times
    times_low = 0.0
    times_high = 16.0

    # Initial values for complete ODE (with S-Terms)
    n0 = 0.25
    x0 = n0

    # Construct parameter hyperspace
    n_times = 4
    n_temps = 3
    
    # Values for temperatures (Q-Values)
    inputs = [
        (temp_low, temp_high, n_temps)
    ]

    # Create a complete Fisher Model
    fsm = FisherModel(
            ode_fun=exp_growth,
            ode_dfdx=dedx,
            ode_dfdp=dedp,
            ode_t0=times_low,
            ode_x0=x0,
            times=(times_low, times_high, n_times),
            inputs=inputs,
            parameters=parameters,
            ode_args=ode_args,
    )

    ####################
    ### OPTIMIZATION ###
    ####################
    fsr = find_optimal(fsm, "scipy_differential_evolution")

    ####################
    ##### PLOTTING #####
    ####################
    plot_all_odes(fsr)
