Theoretical Overview
====================

.. toctree::
   :maxdepth: 2
   :hidden:

   Theoretical Background <theoretical_background/index>

Mathematical modeling is a widely used tool to describe, understand and predict further behavior of living systems.
In particular, in the field of Predictive Biology, one can find a large variety of works that dwell on building models of different levels of complexity controlled by model parameters, e.g., to describe bacteria growth :cite:p:`bernaertsConceptsToolsPredictive2004`.
Based on a chosen model structure, these parameters can be estimated from the gathered experimental data.
Taking into account that the real experimental data always contains measurement noise, the parameter estimates can be only provided with some uncertainty. 
To decrease the error of the parameter values, not only enough experimental data should be gathered but the quality of this data is also pretty sufficient.
That rises quite an important question of finding the Optimal Experimental Design (OED) where optimized experimental conditions and/or times allow one to reduce the number of measurements without loss of information thus sparing an effort of experimenters :cite:p:`derlindenImpactExperimentDesign2013, balsa-cantoe.bangaj.r.COMPUTINGOPTIMALDYNAMIC2008`.

The Experimental Design works iteratively with parameter estimation process.
Firstly, based on the literature review or prior data from pilot experiments, the first parameter estimation of the chosen model structure should be done.
The obtained values can be applied to propose the first optimal experimental design accounting for different constraints, e.g., the lab limitations, human resources, etc. 
Depending on availability, either real or numerical experiments should be conducted based on this design to gather measurement or in-silico data. 
This new data can be used for the new parameter estimations values with corresponding uncertainties.
After this, using new parameter values, the process can be repeated several times to increase the precision of the parameter estimates till the desired accuracy is achieved.

.. figure:: ExpDesign_workflow.png
    :align: center
    :width: 300

    The workflow of the iterative process of model optimization for parameter estimation.

In this library, we present the implementation of the Experimental Design part of the described process for a specific type of model.
We are interested in systems widely used in Systems Biology and described by Ordinary Differential Equations (ODE).

.. bibliography::
    :style: plain
