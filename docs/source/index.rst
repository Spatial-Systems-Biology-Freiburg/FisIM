.. FisInMa documentation master file, created by
   sphinx-quickstart on Thu Oct 20 17:22:36 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FisInMa
=======

`FisInMa <https://spatial-systems-biology-freiburg.github.io/FisInMa/>`_ is a Python library for designing optimal experimental conditions to estimate parameters :math:`p` of a system described by an ordinary differential equation (ODE) as defined in equation :eq:`overview_ode_def`.
This equation may be influenced by input variables :math:`u`.

.. math::
   \begin{alignat}{3}
      &\dot{x}(t) &&= f(t, x, u, p)\\
      &x(t_0) &&= x_0
   \end{alignat}
   :label: overview_ode_def

When designing real-life experiments, researchers need to choose appropriate time- and input-datapoints to gain the maximum amount of information to accurately determine the parameters describing the system.
This package assumes that the structure of the experiment is determined by an ODE which can in general be written in explicit form as in equation :eq:`overview_ode_def`.
The input variables :math:`u` are known numerical values and alter the behaviour of the system while the experiment aims to estimate the parameters :math:`p`.
To optimally design an experiment (ie. choose time-points :math:`t_i` or configurations for input values :math:`u`), we want to maximize the total information obtained by the measurements.
The package uses the Fisher Information method in which we calculate the sensitivities.

.. math::
   \begin{equation}
      S_{ij} = \frac{\partial x_i}{\partial p_j}(t_k, u_l)
   \end{equation}

and afterwards the Fisher Information matrix :math:`F` (optionally with covariance matrix :math:`C` determined by specified uncertainties).

.. math::
   \begin{equation}
      F = S^T C^{-1} S
   \end{equation}

Different criteria such as the determinant, minimum eigenvalue or sum of eigenvalues of the matrix :math:`F` calculate the amount of information and total uncertainty.
To achieve a global optimum, different numerical optimization routines can be applied in a multitude of configurations.

.. note::

   In-depth information about the calculation methods will be described in a book chapter releasing in the near future.
   The development of this package is not yet finalized.
   The authors are thankful for contributions and suggestions.


.. toctree::
   :maxdepth: 2
   :hidden:

   Getting Started <getting_started/index>
   Concepts and Background <theoretical_overview/index>
   Documentation <documentation/index>
   Contributing <contributing>
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
