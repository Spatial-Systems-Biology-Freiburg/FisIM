FisInMa
=======

FisInMa is designed to analyze the informational content of systems of Ordinary
Differential Equations (see :doc:`Concepts and Background <../theoretical_overview/index>`).
Naturally, the first step is to chose one such system and define it via a function.

.. code-block:: python

   def ode_fun(t, y, inputs, parameters, ode_args):
      ...
      return [ ... ]

We also require additional information in the form of derivatives with respect to the
state variables :code:`y` and :code:`parameters`.

.. code-block:: python

   def ode_dfdx(t, y, inputs, parameters, ode_args):
      ...
      return [[ ... ], ...]
   
   def ode_dfdp(t, y, inputs, parameters, ode_args):
      ...
      return [[ ... ], ...]

We then define the variables :code:`times, inputs, parameters` and optional :code:`ode_args`.
The :doc:`model` module gathers all information to fully define a valid
model. It does this by creating a :class:`.FisherModel`

.. code-block:: python

   fsm = FisherModel(
      ode_fun,
      ode_dfdx,
      ode_dfdp,
      ode_x0,
      ode_t0,
      times,
      inputs,
      parameters,
   )

which can afterwards be used by the :doc:`solving` module to calculate the information 
content of the model. The user can decide to no explicitly specify certain variables and leave
them mutable for optimization. It is thus necessary to transform the :class:`.FisherModel` 
it into a fully parametrized model :class:`.FisherModelParametrized` by using an initial guess
for these mutable variables.

.. code-block:: python

   fsmp = FisherModelParametrized.init_from(fsm)
   calculate_fisher_criterion(fsmp)

In addition to solving a model, we can also optimize previously defined mutable variables
such that the information content given by one of the chosen :mod:`.criteria` is maximized.
This procedure is handled by the :doc:`optimization` module and yields :class:`.FisherResults`.

.. code-block:: python

   fsr = find_optimal(fsm)

After the optimization routine has come to an end, the :doc:`plotting` module visualizes 
the obtained results.

.. code-block:: python

   plot_all_solutions(fsr, outdir="out")

The :doc:`database` module dumps information in json format to a 
file or string.

.. code-block:: python

   json_dump(fsr, "model.json")



.. toctree::
   :maxdepth: 2
   :hidden:

   Model <model>
   Solving <solving>
   Optimization <optimization>
   Database <database>
   Plotting <plotting>
