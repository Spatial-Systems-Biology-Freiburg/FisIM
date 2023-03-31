Model Formulation
=================

.. note::

  In-depth information about the theoretical underlying and the calculation methods will be described in a book chapter releasing in the near future.

This library supports systems whose state variable vector :math:`x = (x_1, x_2, ..., x_n)` evolution in time :math:`t` is described by the ODEs:

.. math::
    \begin{alignat}{3}
      &\dot{x}(t) &&= f(t, x, u, p)\\
      &x(t_0) &&= x_0
    \end{alignat}
   :label: overview_ode_def

Here :math:`x_0 (t_0)` is an initial condition, :math:`u` is a vector of an external inputs, and :math:`p` are the estimated parameters of the system.
The observable (measured value) of the system :math:`y` at a time :math:`t_i` is described as

.. math::
    \begin{alignat}{3}
      &y (t_i) &&= g(t_i, x (t_i), u, p) + \epsilon (t_i),
    \end{alignat}
   :label: overview_observable_def

where the function :math:`g` is the model output, and :math:`\epsilon` is the measurement noise. 
