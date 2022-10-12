#!/usr/bin/env python3


def f(x, t, input, params, consts):
    pass


def dfdx(x, t, input, params, consts):
    pass


def dfdp(x, t, input, params, consts):
    pass


def g(x, t, input, params, consts):
    pass


def dgdx(x, t, input, params, consts):
    pass


def dgdp(x, t, input, params, consts):
    pass


def find_optimal(model):
    return result


class model:
    def find_optimal(self):
        return find_optimal(self)


if __name__ == "__main__":
    # Define parameters
    p = ()

    # Define constants
    c = ()

    # Define initial conditions
    x0 = np.array([???])

    # Define bounds and number of sampling points for times, inputs
    times = (t_low, t_high, n_t, dt)

    # Example: Define fixed time points
    times = np.linspace(t_low, t_high, n_t)

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