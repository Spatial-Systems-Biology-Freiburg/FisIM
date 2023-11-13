Damped Oscillator
=================

A damped oscillator can be described by the second order ordinary differential equation

.. math::
    :name: eq:damped-osci-ode

    \ddot{x} + \mu(h)\dot{x} + \lambda x = 0

Here we additionally assume that the parameter :math:`\mu` depends on another input 
variable :math:`h` via :math:`\mu(h)=ah + b`.
To obtain a first-order equation, we substitute :math:`\dot{x}=A` and :math:`x=B` and obtain

.. math::
    :name: eq:damped-osci-ode-first-order

    \begin{align}
        \dot{A} &= -\mu(h) A - \lambda B\\
        \dot{B} &= A
    \end{align}

This is now a first order ODE with the parameters :math:`a,b,\lambda` and the input variable :math:`h`.
Now we can begin with the numerical description of the system.
It is good practice to first import every needed libraries at the top of our file.

.. literalinclude:: ../../../../examples/damped_oscillator.py
   :language: python
   :linenos:
   :lineno-start: 7
   :lines: 7-8

To define the system numerically, we write the preceding equations as a function.
Only the right-hand side of the :ref:`ODE equation <eq:damped-osci-ode-first-order>`.

.. literalinclude:: ../../../../examples/damped_oscillator.py
   :language: python
   :linenos:
   :lineno-start: 10
   :lines: 10-18

We seek to maximize the amount of information in the system to most accurately estimate the parameters
:math:`(a, b, \lambda)`.
To achieve this, we must also define the derivatives of the :ref:`ODE equation <eq:damped-osci-ode-first-order>`.
with respect to the components of the ODE :math:`(A, B)` and the parameters.

.. literalinclude:: ../../../../examples/damped_oscillator.py
   :language: python
   :linenos:
   :lineno-start: 20
   :lines: 20-36

Now we have defined the overall structure of the ODE but are still lacking actual numerical values
to be able to solve the system.
We gather them in the main function of our script and start with the initial guesses of the parameters.

.. literalinclude:: ../../../../examples/damped_oscillator.py
   :language: python
   :linenos:
   :lineno-start: 41
   :lines: 41-47

Next, we define the initial values of the :ref:`ODE system<eq:damped-osci-ode-first-order>`.
Notice, that since we have a two-component system, we need to define values for :math:`(A, B)`,
meaning in our case the variable :math:`x` as well as its time derivative :math:`\dot{x}`.
In the next steps, we define helper variables to later pick explicit values for the input variable :math:`h`
and a range to optimize time points :math:`t_i` when to evaluate the solution of the ODE.

.. literalinclude:: ../../../../examples/damped_oscillator.py
   :language: python
   :linenos:
   :lineno-start: 53
   :lines: 53-61

The next statement fixes the explicit values of :math:`h`.

.. literalinclude:: ../../../../examples/damped_oscillator.py
   :language: python
   :linenos:
   :lineno-start: 63
   :lines: 63-66

A short inspection reveals that the following lines of code yield us with a numpy array of explicit values.

.. code:: python

    # Numerical values for the input variable h
    >>> h_low = 0.08
    >>> h_high = 0.12
    >>> n_h = 1
    >>> np.linspace(h_low, h_high, n_h)
    array([0.08])

So far we have not yet used the methods developed in this package.
But now we are ready to define the fisher model which can then be solved to obtain optimal conditions
for our experimental design.

.. literalinclude:: ../../../../examples/damped_oscillator.py
   :language: python
   :linenos:
   :lineno-start: 68
   :lines: 68-78

The next step solves this model and actually does the optimization.
The result is called a fisher result and contains information on final values and the
optimization procedure.

.. literalinclude:: ../../../../examples/damped_oscillator.py
   :language: python
   :linenos:
   :lineno-start: 80
   :lines: 80-81

When executing the script in a terminal, the output might look like the following.

.. literalinclude:: ../../../source/_static/damped_osci_plots/output_example.txt

In our final step we can visualize the results by autmatically generating images or saving
results as a json file.

.. literalinclude:: ../../../../examples/damped_oscillator.py
   :language: python
   :linenos:
   :lineno-start: 83
   :lines: 83-85

This image shows one of the results of this optimization run.
You can see the :math:`B` component of the :ref:`ODE <eq:damped-osci-ode-first-order>`.

.. image:: ../../../source/_static/damped_osci_plots/Observable_Results_damped_osci_fisher_determinant__000_x_01.svg
