#!/usr/bin/env python3
# The next 2 lines of code are only needed if executing this example 
# without installing the package
import os, sys
sys.path.append(os.getcwd())

import numpy as np
from FisInMa import *

# Defines the right hand side of the ODE of an parametrized damped oscillator
def damped_osci(t, y, inputs, parameters, ode_args):
    a, b, l = parameters
    h, = inputs
    A, B = y
    return [
        -(a*h+b)*A - l*B,
        A
    ]

# Derivative of the RHS of the ODE with respect to the components of the ODE
def damped_osci_dfdx(t, y, inputs, parameters, ode_args):
    a, b, l = parameters
    h, = inputs
    A, B = y
    return [
        [-(a*h+b), -l],
        [1, 0]
    ]

# Derivative with respect to the parameters
def damped_osci_dfdp(t, y, inputs, parameters, ode_args):
    a, b, l = parameters
    h, = inputs
    A, B = y
    return [
        [-h*A, -A, -B],
        [0, 0, 0]
    ]


# The main function executes the script and gathers every needed numerical value.
if __name__ == "__main__": 
    # Define initial parameter guesses
    a = 3.0
    b = 1.0
    l = 5.0
    parameters = (a, b, l)

    # Initial values for ODE
    t0 = 0.0
    x0 = [6.0, 20.0]

    # Numerical values for the input variable h
    h_low = 0.08
    h_high = 0.12
    n_h = 1

    # Numerical values for time points
    times_low = t0
    times_high = 10.0
    n_times = 3

    # Gather information on inputs in list
    inputs = [
        np.linspace(h_low, h_high, n_h)
    ]

    # Create a fisher model containing all information of the system.
    fsm = FisherModel(
            ode_fun=damped_osci,
            ode_dfdx=damped_osci_dfdx,
            ode_dfdp=damped_osci_dfdp,
            ode_t0=times_low,
            ode_x0=x0,
            times={"lb":times_low, "ub":times_high, "n":n_times},
            inputs=inputs,
            parameters=parameters,
    )

    # This finds the optimal time points to estimate the parameters
    fsr = find_optimal(fsm)

    # Automatically plot the results and export as json file
    plot_all_solutions(fsr, outdir="out")
    json_dump(fsr, "damped_osci.json")
