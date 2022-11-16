Baranyi and Roberts Model
=========================

The Baranyi and Roberts growth model (1994) is introduced by a two-dimensional vector of state variables :math:`\mathbf{x}=(x_1, x_2)`, where :math:`x_1(t)` denotes the cell concentration of a bacterial population at the time :math:`t`, and :math:`x_2(t)` defines a physiological state of the cells, the process of adjustment (lag-phase):

.. math::
    :name: eq:baranyi_roberts_ode 
    
    \begin{alignat}{3}
        &\dot x_1(t) &&= \frac{x_2(t)}{x_2(t) + 1} \mu^\text{max} \bigg(1 - \frac{x_1(t)}{x_1^\text{max}}\bigg) x(t)\\
        &\dot x_2(t) &&= \mu^\text{max}  x_2(t)
    \end{alignat}
 

Here :math:`\mu^\text{max}` is the maximum growth rate, and :math:`x_1^\text{max}` is bacteria concentration at the saturation. 
To account for the influence of the temperature on the activity of the model, we will use the 'square root' or Ratkowsky-type model for the maximum growth rate

.. math::
   \begin{alignat}{3}
        \sqrt{\mu^\text{max}} = b (T - T_\text{min}),
   \end{alignat}
   :label: eq:ratakowski_model

where :math:`b` is the regression coefficient, and :math:`T_\text{min}` is the minimum temperature at which the growth can occur.
Here :math:`x_1^\text{max}, b, T_\text{min}` are parameters that we estimate. And temperature :math:`T` is an input of the system.

First of all, import al the necessary libraries.

.. code-block:: python3

    import numpy as np
    from FisInMa import *

Define the system of ODEs.

.. literalinclude:: ../../../../examples/baranyi_roberts.py
   :language: python
   :linenos:
   :lineno-start: 8
   :lines: 8-53

As an observable, it is pretty common to measure the bacteria count :math:`x_1` or the logarithm of this value. 
For simplicity, we would consider the prior case :math:`y(t_i) = x_1(t_i)`.

.. literalinclude:: ../../../../examples/baranyi_roberts.py
   :language: python
   :linenos:
   :lineno-start: 55
   :lines: 55-77

Define the parameters of the system :code:`p` and initial conditions :code:`x0`.

.. literalinclude:: ../../../../examples/baranyi_roberts.py
   :language: python
   :linenos:
   :lineno-start: 82
   :lines: 82-83

Define optimization of 6 time points with lower bound :code:`0.0`, upper bound :code:`10.0`.

.. literalinclude:: ../../../../examples/baranyi_roberts.py
   :language: python
   :linenos:
   :lineno-start: 86
   :lines: 86

Define optimization of one input value (temperature) with lower bound :code:`3.0`, upper bound :code:`12.0`.

.. literalinclude:: ../../../../examples/baranyi_roberts.py
   :language: python
   :linenos:
   :lineno-start: 89
   :lines: 89

The resulting Fisher Model:

.. literalinclude:: ../../../../examples/baranyi_roberts.py
   :language: python
   :linenos:
   :lineno-start: 93
   :lines: 93-107

The optimization is then held using relative sensitivities and D-optimality criterion (determinant).

.. literalinclude:: ../../../../examples/baranyi_roberts.py
   :language: python
   :linenos:
   :lineno-start: 109
   :lines: 109-117

Save and plot the results of optimization.

.. literalinclude:: ../../../../examples/baranyi_roberts.py
   :language: python
   :linenos:
   :lineno-start: 121
   :lines: 121-122

The resulting Optimal Experimental Design:

.. figure:: ../../../source/_static/baranyi_roberts/Observable_Results_baranyi_roberts_ode_fisher_determinant_rel_sensit_cont_6times_1temps_000_x_00.svg
    :align: center
    :width: 400

    The output of the Experimental Design optimization procedure. 
    Line plot: the model solution for the observable, scatter plot: the design time points.