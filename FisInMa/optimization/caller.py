from FisInMa.model import FisherModel, FisherModelParametrized
from .scipy_global_optim import __scipy_differential_evolution, __scipy_basinhopping, __scipy_brute
from .display import display_optimization_start, display_optimization_end


OPTIMIZATION_STRATEGIES = {
    "scipy_differential_evolution": __scipy_differential_evolution,
    "scipy_basinhopping": __scipy_basinhopping,
    "scipy_brute": __scipy_brute,
}


def find_optimal(fsm: FisherModel, optimization_strategy: str="scipy_differential_evolution", discrete_penalizer="default", verbose=True, **kwargs):
    r"""Find the global optimum of the supplied FisherModel.

    :param fsm: The FisherModel object that defines the studied system with its all constraints.
    :type fsm: FisherModel
    :param optimization_strategy: Choose the optimization strategy to find global maximum of the objective function. The default is "scipy_differential_evolution".

        - "scipy_differential_evolution" (recommended)
            The global optimization method uses the `scipy.optimize.differential_evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`__ function showing rather good results for nonlinear dynamic problems.
            The strategy was developed by Storn and Price (1996) and work as follows.

            Firstly, the initial population of the vectors of all optimized values (times and inputs) for one Experimental Design (solutions) is randomly chosen from the region of available values.
            Then each solution mutates by mixing with other candidates.
            To a chosen one solution from the initial population :math:`D_0`, a weighted difference between two other random solutions from the same set :math:`(D_\text{rand1} - D_\text{rand2})` is added.
            This process is called mutation and a new vector :math:`D_m` is obtained.
            The next step is to construct a new trial solution.
            This is done by randomly choosing the elements of this vector either from the initial :math:`D_0` or the mutated :math:`D_m` solutions.
            For each new element of trial vector, from the segment [0, 1) the number should be randomly picked and compared to the so-called recombination constant.
            If this number is less than a constant, then the new solution element is chosen from mutated vector :math:`D_m`, otherwise from :math:`D_0`.
            So, in general, the degree of mutation can be controlled by changing this recombination constant.
            When the trial candidate is built, it is compared to initial solution :math:`D_0`, and the best of them is chosen for the next generation.
            This operation is repeated for every solution candidate of the initial population, and the new population generation can be formed.
            The process of population mutation is repeated till the desired accuracy is achieved.
            This method is rather simple, straightforward, does not require the gradient calculation and is able to be parallelized.
        - "scipy_basinhopping"
            The global optimization method uses the `scipy.optimize.basinhopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`__ function.
            The algorithm combines the Monte-Carlo optimization with Methropolis acceptance criterion and local optimization that works as follows.
            The strategy is developed by David Wales and Jonathan Doye and combines the Monte-Carlo and local optimization. 
            The classic Monte-Carlo algorithm implies that the values of the optimized vector are perturbed and are either accepted or rejected.
            However, in this modified strategy, after perturbation, the vector is additionally subjected to local optimization.
            And only after this procedure the move is accepted according to the Metropolis criterion.

        - "scipy_brute"
            The global optimization method uses the `scipy.optimize.brute <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html>`__ function.
            It is a grid search algorithm calculating the objective function value at each point of a multidimensional grid in a chosen region.
            The technique is rather slow and inefficient but the global minimum can be guaranteed.
    
    :type optimization_strategy: str
    :param discrete_penalizer: A function that takes two 1d arrays (values, discretization) and returns a float. It calculates the penalty (1=no penalty, 0=maximum penalty) for datapoints which do not sit on the desired discretization points.
        
        - "default"
            Uses the default penalty function that is described by the function :py:meth:`discrete_penalty_calculator_default`.

        - "product_difference"
            Uses the discretization penalty function described by the function :py:meth:`discrete_penalty_calculator_default`.

        - "individual_zigzag"
            Uses the discretization penalty function described by the function :py:meth:`discrete_penalty_individual_template` with the penalty structure *pen_structure=penalty_structure_zigzag*.
        - "individual_cos"
            Uses the discretization penalty function described by the function :py:meth:`discrete_penalty_individual_template` with the penalty structure *pen_structure=penalty_structure_cos*.

        - "individual_gauss"
            Uses the discretization penalty function described by the function :py:meth:`discrete_penalty_individual_template` with the penalty structure *pen_structure=penalty_structure_gauss*.
    :type discrete_penalizer: str

    :raises KeyError: Raised if the chosen optimization strategy is not implemented.
    :return: The result of the optimization as an object *FisherResults*. Important attributes are the conditions of the Optimal Experimental Design *times*, *inputs*, the resultion value of the objective function *criterion*.
    :rtype: FisherResults
    """
    fsmp = FisherModelParametrized.init_from(fsm)

    if verbose==True:
        display_optimization_start(fsmp)

    if optimization_strategy not in OPTIMIZATION_STRATEGIES.keys():
        # TODO test this statement
        raise KeyError("Please specify one of the following optimization_strategies for optimization: " + str(OPTIMIZATION_STRATEGIES.keys()))

    fsr = OPTIMIZATION_STRATEGIES[optimization_strategy](fsmp, discrete_penalizer, **kwargs)

    if verbose==True:
        display_optimization_end(fsr)

    return fsr
