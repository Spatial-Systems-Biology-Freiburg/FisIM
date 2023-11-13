Optimization
============

.. note::

    In-depth information about the theoretical underlying and the calculation methods will be described in a book chapter releasing in the near future.

To find the Optimal Experimental Design that corresponds to the maximum objective function (Fisher criterion) different algorithms can be used.
Some of the suggested global optimization algorithms used in the current package are:

1. Differential Evolution
    Stochastic global optimization developed by Storn and Price (1996) :cite:p:`stornDifferentialEvolutionSimple1997`.
    Creating new candidate solutions by combining existing ones to achieve the best solution.

2. Basin-hopping
    Combination of the Monte-Carlo and local optimization introduced by David Wales and Jonathan Doye :cite:p:`walesGlobalOptimizationBasinHopping1997`

3. Brute force
    Calculating the objective function value at each point of a multidimensional grid.

.. bibliography::
    :style: plain
    :filter: False

    stornDifferentialEvolutionSimple1997
    walesGlobalOptimizationBasinHopping1997
