FisInMa
=======

The package is divided into multiple modules.
The :doc:`model <model>` module gathers all information needed to fully define a numerically
solvable model. It does this by creating a :class:`.FisherModel`

.. code-block:: python

   fsm = FisherModel(
      ode_x0,
      ode_t0,
      ode_fun,
      ode_dfdx,
      ode_dfdp,
      ode_initial,
      times,
      inputs,
      parameters,
   )

which can afterwards be used to calculate the information content of the model.
It is necessary to transform the :class:`.FisherModel` it into a fully 
parametrized model :class:`.FisherModelParametrized` by using an initial guess

.. code-block:: python

   fsmp = FisherModelParametrized.init_from(fsm)
   calculate_fisher_criterion(fsmp)

Additionally to solving a model, we can also optimize previously defined mutable variables
such that the information content given by the desired :mod:`.criteria` is maximized.
This procedure is handled by the :mod:`.optimization` module.

.. toctree::
   :maxdepth: 2
   :hidden:

   Model <model>
   Solving <solving>
   Optimization <optimization>
   Database <database>
   Plotting <plotting>
