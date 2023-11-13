Theoretical Overview
====================

.. toctree::
   :maxdepth: 2
   :hidden:

   Theoretical Background <theoretical_background/index>

The parameter estimation from the gathered experimental data is a significant part of the mathematical modeling of the real-life system.
However, due to constant measurement noise, the parameter values are always found with some uncertainty.
To reduce this error, the Experimental Design should be optimized to increase the informational worth of the data :cite:p:`derlindenImpactExperimentDesign2013, balsa-cantoe.bangaj.r.COMPUTINGOPTIMALDYNAMIC2008`.

The Experimental Design works iteratively with parameter estimation process.
Firstly, using the chosen model structure and the first parameter estimated values, the first optimal experimental design can be proposed accounting for different constraints.
Then depending on availability, either real or numerical experiments should be conducted based on this design to gather measurement or in-silico data. 
This new data can be used for the new parameter estimations with corresponding uncertainties.
After this, using new parameter values, the process can be repeated several times to increase the precision of the parameter estimates till the desired accuracy is achieved.

.. figure:: ExpDesign_workflow.png
    :align: center
    :width: 300

    The workflow of the iterative process of model optimization for parameter estimation.

In this library, we present the implementation of the Experimental Design part of the described process for a specific type of model.
We are interested in systems widely used in Systems Biology and described by Ordinary Differential Equations (ODE).

.. bibliography::
    :style: plain
    :filter: False

    derlindenImpactExperimentDesign2013
    balsa-cantoe.bangaj.r.COMPUTINGOPTIMALDYNAMIC2008
