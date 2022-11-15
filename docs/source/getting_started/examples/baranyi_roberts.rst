Baranyi and Roberts Model
=========================

The Baranyi and Roberts growth model (1994) is introduced by a two-dimensional vector of state variables :math:`\mathbf{x}=(x_1, x_2)`, where :math:`x_1(t)` denotes the cell concentration of a bacterial population at the time :math:`t`, and :math:`x_2(t)` defines a physiological state of the cells, the process of adjustment (lag-phase):

.. math::
    \begin{alignat}{3}
        &\dot x_1(t) &= \frac{x_2(t)}{x_2(t) + 1} \mu^\text{max} \bigg(1 - \frac{x_1(t)}{x_1^\text{max}}\bigg) x(t)\\
        &\dot x_2(t) &= \mu^\text{max}  x_2(t)
    \end{alignat}
   :label: eq:baranyi_roberts_ode    

Here :math:`\mu^\text{max}` is the maximum growth rate, and :math:`x_1^\text{max}` is bacteria concentration at the saturation. 
To account for the influence of the temperature on the activity of the model, we will use the 'square root' or Ratkowsky-type model for the maximum growth rate

.. math::
   \begin{alignat}{3}
        \sqrt{\mu^\text{max}} = b (T - T_\text{min}),
   \end{alignat}
   :label: eq:ratakowski_model

where :math:`b` is the regression coefficient, and :math:`T_\text{min}` is the minimum temperature at which the growth can occur.
Here :math:`x_1^\text{max}, b, T_\text{min}` are parameters that we estimate. And temperature :math:`T` is an input of the system.

.. code-block:: python3
    :caption: The input file: the ODEs definition.

    #!/usr/bin/env python3
    import numpy as np
    from FisInMa import *

    def baranyi_roberts_ode(t, x, u, p, ode_args):
    # Define the ODEs
        x1, x2 = x
        (Temp, ) = u
        (x_max, b, Temp_min) = p
        # Define the maximum growth rate
        mu_max = b**2 * (Temp - Temp_min)**2
        return [
            mu_max * (x2/(x2 + 1)) * (1 - x1/x_max) * x1,           # f1
            mu_max * x2                                             # f2
        ]

    def ode_dfdp(t, x, u, p, ode_args):
    # Define the derivative of the function in the ODEs w.r.t. parameter vector
        x1, x2 = x
        (Temp, ) = u
        (x_max, b, Temp_min) = p
        mu_max = b**2 * (Temp - Temp_min)**2
        return [
            [
                mu_max * (x2/(x2 + 1)) * (x1/x_max)**2,             # df1/dx_max
                2 * b * (Temp - Temp_min)**2 * (x2/(x2 + 1))
                    * (1 - x1/x_max)*x1,                            # df1/db
                -2 * b**2 * (Temp - Temp_min) * (x2/(x2 + 1))
                    * (1 - x1/x_max)*x1                             # df1/dTemp_min
            ],
            [
                0,                                                  # df2/dx_max
                2 * b * (Temp - Temp_min)**2 * x2,                  # df2/db
                2 * b**2 * (Temp - Temp_min) * x2                   # df2/dTemp_min
            ]
        ]

    def ode_dfdx(t, x, u, p, ode_args):
    # Define the derivative of the function in the ODEs w.r.t. state variable vector
        x1, x2 = x
        (Temp, ) = u
        (x_max, b, Temp_min) = p
        mu_max = b**2 * (Temp - Temp_min)**2
        return [
            [
                mu_max * (x2/(x2 + 1)) * (1 - 2*x1/x_max),          # df1/dx1
                mu_max * 1/(x2 + 1)**2 * (1 - x1/x_max)*x1          # df1/dx2
            ],
            [
                0,                                                  # df2/dx1
                mu_max                                              # df2/dx2
            ]
        ]

As an observable, it is pretty common to measure the bacteria count :math:`x_1` or the logarithm of this value. 
For simplicity, we would consider the prior case :math:`y(t_i) = x_1(t_i)`.

.. code-block:: python3
    :caption: The input file: the observables definition.

    def obs_fun(t, x, u, p, ode_args):
    # Define the observable function
        x1, x2 = x
        (Temp, ) = u
        (x_max, b, Temp_min) = p
        return [
            x1
        ]

    def obs_dgdp(t, x, u, p, ode_args):
    # Define the derivative of the observable function w.r.t. parameter vector
        x1, x2 = x
        (Temp, ) = u
        (x_max, b, Temp_min) = p
        return [
            [0, 0, 0]
        ]

    def obs_dgdx(t, x, u, p, ode_args):
    # Define the derivative of the observable function w.r.t. state variable vector
        x1, x2 = x
        (Temp, ) = u
        (x_max, b, Temp_min) = p
        return [
            [1, 0]
        ]

Define the parameters of the system :code:`p` and initial conditions :code:`x0`.

.. code-block:: python3

    if __name__ == "__main__":
        p = (1e8, 0.2, 1.0) # (x_max, b, T_min)
        x0 = np.array([1e3, .01])

Define optimization of 6 time points with lower bound :code:`0.0`, upper bound :code:`10.0`.

.. code-block:: python3

    times = {"lb": 0.0, "ub": 10.0, "n": 6}

Define optimization of one input value (temperature) with lower bound :code:`3.0`, upper bound :code:`12.0`.

.. code-block:: python3

    inputs = [{"lb": 3.0, "ub": 12.0, "n": 1}]


The resulting Fisher Model:

.. code-block:: python3

    fsm = FisherModel(
        ode_x0=x0,
        ode_t0=0.0,
        ode_fun=baranyi_roberts_ode,
        ode_dfdx=ode_dfdx,
        ode_dfdp=ode_dfdp,
        ode_initial=x0,
        times=times,
        inputs=inputs,
        parameters=p,
        obs_fun=obs_fun,
        obs_dgdx=obs_dgdx,
        obs_dgdp=obs_dgdp,
        covariance={"abs": 0.3, "rel": 0.1}
    )


.. code-block:: python3
    :caption: The input file: optimization function

    fsr = find_optimal(
        fsm,
        relative_sensitivities=True,
        recombination=0.7,
        mutation=(0.1, 0.8),
        workers=20,
        popsize=10,
        polish=False,
    )

Save and plot the results of optimization.

.. code-block:: python3
    :caption: The input file: saving and plotting.

    plot_all_observables(fsr)
    json_dump(fsr, "baranyi_roberts_design.json")

The resulting Optimal Experimental Design:

.. figure:: Observable_Results_baranyi_roberts_ode_fisher_determinant_rel_sensit_cont_6times_1temps_000_x_00.svg
    :align: center
    :width: 400

    The output of the Experimental Design optimization procedure. 
    Line plot: the model solution for the observable, scatter plot: the design time points.