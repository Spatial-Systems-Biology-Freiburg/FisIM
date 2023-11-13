Experimental Design
===================

.. note::

   In-depth information about the theoretical underlying and the calculation methods will be described in a book chapter releasing in the near future.

The Experimental Design goal is to maximize the objective function based on one of the properties of the Fisher information matrix (FIM)  :math:`F`, which is inversely proportional to the minimal squared estimation error :cite:p:`friedenExploratoryData2010`.
Different optimality criteria (properties of the FIM) can be chosen.
Some of the most popular criteria are:

1. *D-optimality criterion*
    Maximizes the determinant :math:`\det (F)` of the FIM.

2. *E-optimality criterion*
    Maximizes the minimal eigenvalue :math:`\lambda_{\min}`.

3. *A-optimality criterion*
    Maximizes the sum of all eigenvalues :math:`\sum_i \lambda_i`.

4. *Modified E-optimality*
    Maximizes the ratio between the minimal and maximal eigenvalue :math:`\lambda_{\min} / \lambda_{\max}`.

.. bibliography::
    :style: plain
    :filter: False

    friedenExploratoryData2010

