#!/usr/bin/env python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom functions for optimization
from src.solving import convert_S_matrix_to_determinant, unpack_fischer_model, get_S_matrix
from src.fischer_model import FischerModel


# System of equation for pool-model and sensitivities
###############################
### USER DEFINES ODE SYSTEM ###
###############################
def pool_model_sensitivity(y, t, Q, P, Const):
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    (n, sa, sb, sc) = y
    return [
        (a*Temp + c) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max),
        (  Temp    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sa,
        (a*Temp + c) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sb,
        (     1    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sc
    ]


def jacobi(y, t, Q, P, Const):
    (n, sa, sb, sc) = y
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    dfdn = (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t))
    return np.array([
        [   dfdn,                                                                                             0,    0,    0   ],
        [(  Temp    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sa, dfdn, 0,    0   ],
        [(a*Temp + c) * (  -  n0/n_max * t * Temp * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sb, 0,    dfdn, 0   ],
        [(     1    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sc, 0,    0,    dfdn]
    ])


####################################
### HELPER FUNCTION OPTIMIZATION ###
####################################
def optimizer_function(X, q_values, P, Const, y0, t0, t_shape, full=False):
    times = np.sort(X.reshape(t_shape), axis=-1)

    fsm = FischerModel(
        observable=convert_S_matrix_to_determinant,
        times=times,
        parameters=P,
        q_values=q_values,
        constants=Const,
        y0_t0=(y0, t0),
        rhs=pool_model_sensitivity,
        jacobian=jacobi
    )

    r = unpack_fischer_model(fsm)
    S, C, r = get_S_matrix(*r)
    d = convert_S_matrix_to_determinant(fsm.times, fsm.q_values, fsm.parameters, fsm.constants, S, C)

    if full:
        return -d, S, C, fsm, r
    return - d


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
    temp_low = 2.0
    temp_high = 16.0
    n_temp = 20

    times_low = t0
    times_high = 15.0
    n_times = 20

    # Initial conditions with initial time
    y0_t0 = (y0, t0)

    # Construct parameter hyperspace
    n_times = 10
    n_temps = 3
    
    times = np.array([np.linspace(temp_low, temp_high, n_times)] * n_temps)
    q_values = [np.linspace(temp_low, temp_high, n_temps)]


    ###############################
    ### OPTIMIZATION FUNCTION ? ###
    ###############################
    x0 = times.flatten()

    bounds = [(times_low, times_high) for _ in range(len(times.flatten()))]
    args = (q_values, P, Const, y0, t0, times.shape)

    res = sp.optimize.minimize(
        optimizer_function,
        x0,
        args=args,
        bounds=bounds
    )

    t = res.x[:np.product(times.shape)].reshape(times.shape)
    (d_neg, S, C, fsm, solutions) = optimizer_function(res.x, *args, full=True)


    ###############################
    ##### PLOTTING FUNCTION ? #####
    ###############################
    fig, axs = plt.subplots(len(solutions), figsize=(12, 4*len(solutions)))
    for i, s in enumerate(solutions):
        t_values = np.linspace(t0, times_high)
        res = sp.integrate.odeint(pool_model_sensitivity, y0, t_values, args=(s[1], fsm.parameters, fsm.constants), Dfun=jacobi).T[0]
        axs[i].plot(t_values, res, color="blue", label="Exact solution")
        axs[i].plot(s[0], s[2][0], marker="o", color="k", linestyle="", label="Q_values: " + str(s[1]))
        axs[i].legend()
    fig.savefig("Result.svg")
