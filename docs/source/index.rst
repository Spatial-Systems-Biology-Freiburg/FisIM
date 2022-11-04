.. FisInMa documentation master file, created by
   sphinx-quickstart on Thu Oct 20 17:22:36 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FisInMa
=======

`FisInMa <https://spatial-systems-biology-freiburg.github.io/FisInMa/>`_ is a Python library for designing optimal experimental conditions to estimate parameters of a system described by an ordinary differential equation (ODE) as defined in equation :eq:`overview_ode_def`.

.. math::
   \begin{alignat}{3}
      &\dot{x}(t) &&= f(t, x, u, p)\\
      &x(t_0) &&= x_0
   \end{alignat}
   :label: overview_ode_def


.. note::

   This project is under active development.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   Introduction <introduction/index>
   User Interface <user_interface/index>
   Examples <examples/index>
   Core <core/index>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
