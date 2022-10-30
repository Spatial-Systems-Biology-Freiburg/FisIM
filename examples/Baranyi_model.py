#!/usr/bin/env python3
import numpy as np
import os, sys
sys.path.append(os.getcwd())

from FisInMa import *


def ode_fun(t, x, u, P, ode_args):
    x1, x2 = x
    (Temp, ) = u
    (x_max, b, Temp_min) = P
    # Define the maximum growth rate
    mu_max = b**2 * (Temp - Temp_min)**2
    return [
        mu_max * (x2/(x2 + 1)) * (1 - x1/x_max) * x1, # f1
        mu_max * x2                                   # f2
    ]

def ode_dfdx(t, x, u, P, ode_args):
    x1, x2 = x
    (Temp, ) = u
    (x_max, b, Temp_min) = P
    mu_max = b**2 * (Temp - Temp_min)**2
    return [
        [
            mu_max * (x2/(x2 + 1)) * (1 - 2*x1/x_max),  # df1/dx1
            mu_max * 1/(x2 + 1)**2 * (1 - x1/x_max)*x1  # df1/dx2
        ], 
        [
            0,                                          # df2/dx1
            mu_max                                      # df2/dx2
        ]
    ]

def ode_dfdp(t, x, u, P, ode_args):
    x1, x2 = x
    (Temp, ) = u
    (x_max, b, Temp_min) = P
    mu_max = b**2 * (Temp - Temp_min)**2
    return [
        [
            mu_max * (x2/(x2 + 1)) * (x1/x_max)**2,                               # df1/dx_max
             2 * b * (Temp - Temp_min)**2 * (x2/(x2 + 1)) * (1 - x1/x_max)*x1,    # df1/db
            -2 * b**2 * (Temp - Temp_min) * (x2/(x2 + 1)) * (1 - x1/x_max)*x1     # df1/dTemp_min
        ],
        [
            0,                                                                    # df2/dx_max
            2 * b * (Temp - Temp_min)**2 * x2,                                         # df2/db
            2 * b**2 * (Temp - Temp_min) * x2                                          # df2/dTemp_min
        ] 
    ]


def obs_fun(t, x, u, P, ode_args):
    x1, x2 = x
    (Temp, ) = u
    (x_max, b, Temp_min) = P
    return [x1] # y = x1

def obs_dgdx(t, x, u, P, ode_args):
    x1, x2 = x
    (Temp, ) = u
    (x_max, b, Temp_min) = P
    return [
        [1, 0] # [dg/dx1, dg/dx2]
    ]

def obs_dgdp(t, x, u, P, ode_args):
    x1, x2 = x
    (Temp, ) = u
    (x_max, b, Temp_min) = P
    return [
        [0, 0, 0] # [dg/dx_max, dg/db, dg/dTemp_min]
    ]


if __name__ == "__main__":
    # Parameters, ode_args and initial conditions were derived from K. Grijspeerdt1 and P. Vanrolleghem (1999) results
    # for Salmonella enteritidis growth curve in egg yolk at 30 C

    # Define parameters
    p = (np.exp(21.1), 0.038, 2) # (x_max, b, T_min)
    
    # Define ode_args
    c = ()

    # Define initial conditions
    x0 = np.array([np.exp(2.36), 1 / (np.exp(2.66)-1)])

    # Define bounds and number of sampling points for times, inputs
    times = (0.0, 1500.0, 4)

    # Example: Define fixed time points
    # times = [1, 2, 3]

    Temp_low = 4.0
    Temp_high = 8.0
    n_Temp = 3
    inputs = [
        # Sample over the range I0_low to I0_high with I0_n values and discretization dI0
        np.linspace(Temp_low, Temp_high, n_Temp)
    ]

    fsm = FisherModel(
        # Required arguments
        ode_x0=x0,
        ode_t0=0.0,
        ode_fun=ode_fun,
        ode_dfdx=ode_dfdx,
        ode_dfdp=ode_dfdp,
        ode_initial=x0,
        times=times,
        inputs=inputs,
        parameters=p,
        ode_args=c,
        # Optional observable arguments
        obs_fun=obs_fun,#=None
        obs_dfdx=obs_dgdx,#=None
        obs_dfdp=obs_dgdp,#=None
    )

    fsr = find_optimal(fsm, relative_sensitivities=True)

    print(fsr.criterion)
    for sol in fsr.individual_results:
        print("Times for inputs", sol.inputs)
        print(" >> ", sol.times)

    plot_all_solutions(fsr, outdir="out")
    json_dump(fsr, "save.json")
