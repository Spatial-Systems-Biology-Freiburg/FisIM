#!/usr/bin/env python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom functions for optimization
from src.solving import fischer_determinant
from src.data_structures import FischerModel
from src.optimization import find_optimal


# System of equation for pool-model and sensitivities
###############################
### USER DEFINES ODE SYSTEM ###
###############################
def pool_model_sensitivity(y, t, Q, P, Const):
    (a, b, c) = P
    (Temp,H) = Q
    (n0, n_max) = Const
    (n, sa, sb, sc) = y
    return [
        (a*Temp + c*H) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max),
        (  Temp      ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c*H) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sa,
        (a*Temp + c*H) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c*H) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sb,
        (     H      ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c*H) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sc
    ]


def jacobi(y, t, Q, P, Const):
    (n, sa, sb, sc) = y
    (a, b, c) = P
    (Temp,H) = Q
    (n0, n_max) = Const
    dfdn = (a*Temp + c*H) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t))
    return np.array([
        [   dfdn,                                                                                             0,    0,    0   ],
        [(  Temp      ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c*H) * (1 - 2 / n_max) * sa, dfdn, 0,    0   ],
        [(a*Temp + c*H) * (  -  n0/n_max * t * Temp * np.exp(-b*Temp*t)) + (a*Temp + c*H) * (1 - 2 / n_max) * sb, 0,    dfdn, 0   ],
        [(     H      ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c*H) * (1 - 2 / n_max) * sc, 0,    0,    dfdn]
    ])


if __name__ == "__main__":
    ###############################
    ### USER DEFINES PARAMETERS ###
    ###############################

    # Define constants for the simulation duration
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
    y0 = np.array([n0, 0, 0, 0])

    # Define bounds for sampling
    temp_low = 4.0
    temp_high = 21.0

    times_low = t0
    times_high = 16.0

    humidity_low = 0.8
    humidity_high = 1.2

    # Initial conditions with initial time
    y0_t0 = (y0, t0)

    # Construct parameter hyperspace
    n_times = 7
    n_temps = 2
    n_humidity = 1
    
    # Values for temperatures (Q-Values)
    q_values = [np.linspace(temp_low, temp_high, n_temps), np.linspace(humidity_low, humidity_high, n_humidity)]
    # Values for times (can be same for every temperature or different)
    # the distinction is made by dimension of array
    
    # This chooses the same time for every q_value
    # times = np.linspace(times_low, times_high, n_times)
    
    # This chooses different times for every q_value
    times = np.full(tuple(len(q) for q in q_values) + (n_times,), np.linspace(times_low, times_high, n_times+2)[1:-1])

    fsm = FischerModel(
        # Required arguments
        times=times,
        parameters=P,
        q_values=q_values,
        constants=Const,
        y0_t0=(y0, t0),
        ode_func=pool_model_sensitivity,
        observable_func=fischer_determinant,
        # Optional arguments
        jacobian=jacobi,
        relative_sensitivities=False
    )

    ###############################
    ### OPTIMIZATION FUNCTION ? ###
    ###############################
    fsr = find_optimal(times, times_high, fsm, "scipy_differential_evolution", workers=1)
    d = fsr.observable
    solutions = fsr.ode_solutions

    ###############################
    ##### PLOTTING FUNCTION ? #####
    ###############################
    fig, axs = plt.subplots(len(solutions), figsize=(12, 4*len(solutions)))
    for i, (t, q, r) in enumerate(solutions):
        # Plot solution to ode
        t_values = np.linspace(t0, times_high)
        res = sp.integrate.odeint(pool_model_sensitivity, y0, t_values, args=(q, fsm.parameters, fsm.constants), Dfun=jacobi).T[0]
        axs[i].plot(t_values, res, color="#21918c", label="Ode Solution")

        # Determine where multiple time points overlap by rounding
        t_round = t.round(1)
        unique, indices, counts = np.unique(t_round, return_index=True, return_counts=True)

        # Plot same time points with different sizes to distinguish
        axs[i].scatter(unique, r[0][indices], s=counts*80, alpha=0.5, color="#440154", label="Q_values: " + str(q))
        axs[i].legend()
    fig.savefig("out/Result.svg")
