#!/usr/bin/env python3
import numpy as np
import os, sys
sys.path.append(os.getcwd())

from FisInMa import *


def baranyi_roberts_ode(t, x, u, p, ode_args):
    x1, x2 = x
    (Temp, ) = u
    (x_max, b, Temp_min) = p
    # Define the maximum growth rate
    mu_max = b**2 * (Temp - Temp_min)**2
    return [
        mu_max * (x2/(x2 + 1)) * (1 - x1/x_max) * x1, # f1
        mu_max * x2                                   # f2
    ]

def ode_dfdp(t, x, u, p, ode_args):
    x1, x2 = x
    (Temp, ) = u
    (x_max, b, Temp_min) = p
    mu_max = b**2 * (Temp - Temp_min)**2
    return [
        [
            mu_max * (x2/(x2 + 1)) * (x1/x_max)**2,                               # df1/dx_max
             2 * b * (Temp - Temp_min)**2 * (x2/(x2 + 1))
                * (1 - x1/x_max)*x1,    # df1/db
            -2 * b**2 * (Temp - Temp_min) * (x2/(x2 + 1))
                * (1 - x1/x_max)*x1     # df1/dTemp_min
        ],
        [
            0,                                                                    # df2/dx_max
            2 * b * (Temp - Temp_min)**2 * x2,                                         # df2/db
            2 * b**2 * (Temp - Temp_min) * x2                                          # df2/dTemp_min
        ] 
    ]

def ode_dfdx(t, x, u, p, ode_args):
    x1, x2 = x
    (Temp, ) = u
    (x_max, b, Temp_min) = p
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


if __name__ == "__main__":
    # Parameters, ode_args and initial conditions were derived from K. Grijspeerdt1 and P. Vanrolleghem (1999) results
    # for Salmonella enteritidis growth curve in egg yolk at 30 C

    # Define parameters
    p = (np.exp(21.1), 0.038, 2) # (x_max, b, T_min)

    # Define initial conditions
    x0 = np.array([np.exp(2.36), 1 / (np.exp(2.66)-1)])

    # Define interval and number of sampling points for times
    times = (0.0, 1500.0, 4)

    # Define explicit temperature points
    Temp_low = 4.0
    Temp_high = 8.0
    n_Temp = 3

    # Summarize all input definitions in list (only temperatures)
    inputs = [
        np.linspace(Temp_low, Temp_high, n_Temp)
    ]

    # Create the FisherModel which serves as the entry point
    #  for the solving and optimization algorithms
    fsm = FisherModel(
        ode_x0=x0,
        ode_t0=0.0,
        ode_fun=baranyi_roberts_ode,
        ode_dfdx=ode_dfdx,
        ode_dfdp=ode_dfdp,
        ode_initial=x0,
        times=times,
        inputs=inputs,
        parameters=p
    )

    fsr = find_optimal(fsm, relative_sensitivities=True)

    # Print the result for the criterion (default=determinant)
    print(fsr.criterion)
    
    # Print time points which were chosen by optimization routine
    for sol in fsr.individual_results:
        print("Times for inputs", sol.inputs)
        print(" >> ", sol.times)

    # Plot all ODE results with chosen time points
    # for different data points
    plot_all_solutions(fsr, outdir="out")

    # Dump all information into a json file
    json_dump(fsr, "baranyi.json")

    # Dumps all information to a string
    output = json_dumps(fsr)
