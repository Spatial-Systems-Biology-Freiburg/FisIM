#!/usr/bin/env python3
import numpy as np


def f(x, t, input, params, consts):
    (x1, x2, ) = x
    (Temp, ) = input
    (x_max, b, Temp_min) = params
    # Define the maximum growth rate
    mu_max = b**2 * (Temp - Temp_min)**2
    return [
        mu_max * (x2/(x2 + 1)) * (1 - x1/x_max) * x1, # f1
        mu_max * x2                                   # f2
    ]


def dfdx(x, t, input, params, consts):
    (x1, x2, ) = x
    (Temp, ) = input
    (x_max, b, Temp_min) = params
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


def dfdp(x, t, input, params, consts):
    (x1, x2, ) = x
    (Temp, ) = input
    (x_max, b, Temp_min) = params
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


def g(x, t, input, params, consts):
    (x1, x2, ) = x
    (Temp, ) = input
    (x_max, b, Temp_min) = params
    return [x1] # y = x1


def dgdx(x, t, input, params, consts):
    (x1, x2, ) = x
    (Temp, ) = input
    (x_max, b, Temp_min) = params
    return [
        [1, 0] # [dg/dx1, dg/dx2]
    ]



def dgdp(x, t, input, params, consts):
    (x1, x2, ) = x
    (Temp, ) = input
    (x_max, b, Temp_min) = params
    return [
        [0, 0, 0] # [dg/dx_max, dg/db, dg/dTemp_min]
    ]



def find_optimal(model):
    return result


class model:
    def find_optimal(self):
        return find_optimal(self)


if __name__ == "__main__":
    # Parameters, constants and initial conditions were derived from K. Grijspeerdt1 and P. Vanrolleghem (1999) results
    # for Salmonella enteritidis growth curve in egg yolk at 30 C

    # Define parameters
    p = (np.exp(21.1), 0.038, 2) # (x_max, b, T_min)
    #p = (1, 2, 3)
    
    # Define constants
    c = ()

    # Define initial conditions
    x0 = np.array([np.exp(2.36), 1 / (np.exp(2.66)-1)])

    # Define bounds and number of sampling points for times, inputs
    times = (t_low, t_high, n_t, dt)

    # Example: Define fixed time points
    times = np.linspace(t_low, t_high, n_t) # [1, 2, 3]

    inputs = [
        # Sample over the range I0_low to I0_high with I0_n values and discretization dI0
        (I0_low, I0_high, I0_n, dI0),
        # Define fixed values for second input variable
        np.array([1, 3, 4, 10]),
        # Define sampling over range with no discretization
        (I2_low, I2_high, I2_n)
        # Same as before: 4th argument is optional
        (I2_low, I2_high, I2_n, None)
    ]

    model = Model(
        # Required arguments
        ode_fun=f,
        ode_dfdx=dfdx,
        ode_dfdp=dfdp,
        ode_initial=x0,
        times=times,
        inputs=inputs,
        parameters=p,
        constants=c,
        # Optional observable arguments
        obs_fun=g,#=None
        obs_dfdx=dgdx,#=None
        obs_dfdp=dgdp,#=None
        # Optional arguments for sampling
        identical_times=False,# TODO!
    )

    result = model.find_optimal(
        # Goal: Only specify optional arguments
        relative_sensitivities=False,
        criterion=determinant,
    )
    # >>> The optimization found the following time-input combinations as optimal:
    # >>> array([
    # >>>   [ ..... ],
    # >>>   [ ..... ],
    # >>> ])

    print(result.observable)
    print(result.times)
    print(result.inputs)

    result.plot_all_ode_result()