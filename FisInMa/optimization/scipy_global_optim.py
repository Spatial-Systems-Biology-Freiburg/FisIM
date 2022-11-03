import numpy as np
import scipy as sp
import scipy.optimize as optimize
import itertools
from pydantic.dataclasses import dataclass

from FisInMa.model import FisherModel, FisherModelParametrized, VariableDefinition
from FisInMa.solving import calculate_fisher_criterion, fisher_determinant


def _create_comparison_matrix(n, value=1.0):
    """Creates a matrix for linear constraints of scipy such that lower and higher values can be compared

    Args:
        n (int): Dimensionality of the resulting matrix will be (n-1,n)
        value (float, optional): Values of the matrix' entries.

    Returns:
        np.ndarary: Matrix of dimension (n-1,n) with entries at A[i][i] (positive) and A[i][i+1] (negative).
    """
    
    # Fill the matrix like so:
    #         | 1 -1  0  0 ... |
    # A = | 0  1 -1  0 ... |
    #     | 0  0  1 -1 ... |
    #     | ...            |
    #     This enables us to compare variables like to
    #     a_(i) - a_(i+1) <= - min_distance
    # <=> a_(i) + min_distance <= a_(i+1)
    A = np.zeros((max(0,n-1), max(0,n)))
    for i in range(n-1):
        A[i][i] = value
        A[i][i+1] = -value
    return A


class PenaltyConfig:
    arbitrary_types_allowed = True


@dataclass(config=PenaltyConfig)
class PenaltyInformation:
    penalty: float
    penalty_ode_t0: float
    # TODO - add penalty for ode_x0 when sampling is done
    # penalty_ode_x0: List[List[float]]
    penalty_inputs: float
    penalty_times: float
    penalty_summary: dict


def discrete_penalty_calculator_default(vals, vals_discr):
    # TODO - document this function
    # TODO - should be specifiable as parameter in optimization routine
    # Calculate the penalty for provided values
    prod = np.array([1 - (np.abs(np.prod((vals_discr - v))))**(1.0 / len(vals_discr)) / (np.max(vals_discr) - np.min(vals_discr)) for v in vals])
    pen = np.product(prod)
    # Return the penalty and the output per inserted variable
    return pen, prod


def _discrete_penalizer(fsmp, penalizer=discrete_penalty_calculator_default):
    # Penalty contribution from initial times
    pen_ode_t0 = 1
    pen_ode_t0_full = []
    if type(fsmp.ode_t0_def) is VariableDefinition:
        # Now we can expect that this parameter was sampled
        # thus we want to look for possible discretization values
        discr = fsmp.ode_t0_def.discrete
        if type(discr) is np.ndarray:
            values = fsmp.ode_t0
            pen_ode_t0, pen_ode_t0_full = penalizer(values, discr)

    # Penalty contribution from inputs
    pen_inputs = 1
    pen_inputs_full = []
    for var_def, var_val in zip(fsmp.inputs_def, fsmp.inputs):
        if type(var_def) == VariableDefinition:
            discr = var_def.discrete
            if type(discr) is np.ndarray:
                values = var_val
                p, p_full = penalizer(values, discr)
                pen_inputs *= p
                pen_inputs_full.append(p_full)

    # Penalty contribution from times
    pen_times = 1
    pen_times_full = []
    if type(fsmp.times_def) is VariableDefinition:
        discr = fsmp.times_def.discrete
        if type(discr) is np.ndarray:
            if fsmp.identical_times==True:
                values = fsmp.times
                pen_times, pen_times_full = penalizer(values, discr)
            else:
                pen_times_full = []
                for index in itertools.product(*[range(len(q)) for q in fsmp.inputs]):
                    if fsmp.identical_times==True:
                        values = fsmp.times
                    else:
                        values = fsmp.times[index]
                    p, p_full = penalizer(values, discr)
                    pen_times *= p
                    pen_times_full.append(p_full)

    # Calculate the total penalty
    pen = pen_ode_t0 * pen_inputs * pen_times

    # Create a summary
    pen_summary = {
        "ode_t0": pen_ode_t0_full,
        "inputs": pen_inputs_full,
        "times": pen_times_full
    }

    # Store values in class
    ret = PenaltyInformation(
        penalty=pen,
        penalty_ode_t0=pen_ode_t0,
        penalty_inputs=pen_inputs,
        penalty_times=pen_times,
        penalty_summary=pen_summary,
    )

    # Store all results and calculate total penalty
    return pen, ret


def __scipy_optimizer_function(X, fsmp: FisherModelParametrized, full=False, relative_sensitivities=False, penalizer=discrete_penalty_calculator_default):
    total = 0
    # Get values for ode_t0
    if fsmp.ode_t0_def is not None:
        fsmp.ode_t0 = X[:fsmp.ode_t0_def.n]
        total += fsmp.ode_t0_def.n
    
    # Get values for ode_x0
    if fsmp.ode_x0_def is not None:
        fsmp.ode_x0 = X[total:total + fsmp.ode_x0_def.n * fsmp.ode_x0.size]
        total += fsmp.ode_x0_def.n

    # Get values for times
    if fsmp.times_def is not None:
        fsmp.times = np.sort(X[total:total+fsmp.times.size].reshape(fsmp.times.shape), axis=-1)
        total += fsmp.times.size

    # Get values for inputs
    for i, inp_def in enumerate(fsmp.inputs_def):
        if inp_def is not None:
            fsmp.inputs[i]=X[total:total+inp_def.n]
            total += inp_def.n

    fsr = calculate_fisher_criterion(fsmp, relative_sensitivities=relative_sensitivities)

    # Calculate the discretization penalty
    penalty, penalty_summary = _discrete_penalizer(fsmp, penalizer)
    
    # Include information about the penalty
    fsr.penalty_discrete_summary = penalty_summary

    # Return full result if desired
    if full:
        return fsr
    return -fsr.criterion * penalty


def _scipy_calculate_bounds_constraints(fsmp: FisherModelParametrized):
    # Define array for upper and lower bounds
    ub = []
    lb = []
    
    # Define constraints via equation lc <= B.dot(x) uc
    # lower and upper constraints lc, uc and matrix B
    lc = []
    uc = []

    # Determine the number of mutable variables which can be sampled over
    n_times = np.product(fsmp.times.shape) if fsmp.times_def  is not None else 0
    n_inputs = [len(q) if q_def is not None else 0 for q, q_def in zip(fsmp.inputs, fsmp.inputs_def)]
    n_mut = [
        fsmp.ode_t0_def.n if fsmp.ode_t0_def is not None else 0,
        fsmp.ode_x0_def.n if fsmp.ode_x0_def is not None else 0,
        n_times,
        *n_inputs
    ]
    B = np.eye(0)

    # Go through all possibly mutable variables and gather information about constraints and bounds
    # Check if initial times are sampled over
    if type(fsmp.ode_t0_def)==VariableDefinition:
        # Bounds for value
        lb += [fsmp.ode_t0_def.lb] * fsmp.ode_t0_def.n
        ub += [fsmp.ode_t0_def.ub] * fsmp.ode_t0_def.n
        
        # Constraints on variables
        lc += [-np.inf] * (fsmp.ode_t0_def.n-1)
        uc += [fsmp.ode_t0_def.min_distance if fsmp.ode_t0_def.min_distance is not None else np.inf] * (fsmp.ode_t0_def.n-1)
        
        # Define matrix A which will extend B
        A = _create_comparison_matrix(fsmp.ode_t0_def.n)
        B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])

    # Check if initial values are sampled over
    if type(fsmp.ode_x0_def)==VariableDefinition:
        # Bounds for value
        lb.append(fsmp.ode_x0_def.lb)
        ub.append(fsmp.ode_x0_def.ub)
        
        # Constraints on variables
        lc += []
        uc += []

        # Extend matrix B
        A = np.eye(0)
        B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])

    # Check if times are sampled over
    if type(fsmp.times_def)==VariableDefinition:
        # How many time points are we sampling?
        n_times = np.product(fsmp.times.shape)

        # Store lower and upper bound
        lb += [fsmp.times_def.lb] * n_times
        ub += [fsmp.times_def.ub] * n_times

        # Constraints on variables
        lc += [-np.inf] * (n_times-1)
        uc += [-fsmp.times_def.min_distance if fsmp.times_def.min_distance is not None else 0.0] * (n_times-1)

        # Extend matrix B
        A = _create_comparison_matrix(n_times)
        B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])
    
    # Check which inputs are sampled
    for inp_def in fsmp.inputs_def:
        if type(inp_def)==VariableDefinition:
            # Store lower and upper bound
            lb += [inp_def.lb] * inp_def.n
            ub += [inp_def.ub] * inp_def.n

            # Constraints on variables
            lc += [-np.inf] * (inp_def.n-1)
            uc += [-inp_def.min_distance if inp_def.min_distance is not None else 0.0] * (inp_def.n-1)

            # Create correct matrix matrix to store
            A = _create_comparison_matrix(inp_def.n)
            B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])

    bounds = list(zip(lb, ub))
    constraints = optimize.LinearConstraint(B, lc, uc)
    return bounds, constraints


def __initial_guess(fsmp: FisherModelParametrized):
    x0 = np.concatenate([
        np.array(fsmp.ode_x0).flatten() if fsmp.ode_x0_def is not None else [],
        np.array(fsmp.ode_t0).flatten() if fsmp.ode_t0_def is not None else [],
        np.array(fsmp.times).flatten() if fsmp.times_def is not None else [],
        *[
            np.array(inp_mut_val).flatten() if inp_mut_val is not None else []
            for inp_mut_val in fsmp.inputs_mut
        ]
    ])
    return x0


def __scipy_differential_evolution(fsmp: FisherModelParametrized, relative_sensitivities=False, **args):
    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)

    x0 = __initial_guess(fsmp)

    opt_args = {
        "func": __scipy_optimizer_function,
        "bounds": bounds,
        #"constraints":constraints,
        "args":(fsmp, False, relative_sensitivities),
        "polish":True,
        "disp": True,
        "workers":-1,
        "updating":'deferred',
        "x0": x0
    }
    opt_args.update(args)
    res = optimize.differential_evolution(**opt_args)

    return __scipy_optimizer_function(res.x, fsmp, full=True, relative_sensitivities=relative_sensitivities)


def __scipy_brute(fsmp: FisherModelParametrized, relative_sensitivities=False, **args):
    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)

    opt_args = {
        "func": __scipy_optimizer_function,
        "ranges": bounds,
        "args":(fsmp, False, relative_sensitivities),
        "finish":False,
        "disp": True,
        "workers":-1,
        "Ns":5
    }
    opt_args.update(args)
    res = optimize.brute(**opt_args)

    return __scipy_optimizer_function(res, fsmp, full=True, relative_sensitivities=relative_sensitivities)


def __scipy_basinhopping(fsmp: FisherModelParametrized, relative_sensitivities=False, **args):
    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)

    x0 = __initial_guess(fsmp)

    opt_args = {
        "func": __scipy_optimizer_function,
        "x0": x0,
        "disp": True,
        "minimizer_kwargs":{"args":(fsmp, False, relative_sensitivities), "bounds": bounds}
    }
    opt_args.update(args)
    res = optimize.basinhopping(**opt_args)

    return __scipy_optimizer_function(res.x, fsmp, full=True, relative_sensitivities=relative_sensitivities)


def find_optimal(fsm: FisherModel, optimization_strategy: str="scipy_differential_evolution", criterion=fisher_determinant, **args):
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
    
    :param criterion: Choose the optimality criterion to determine the objective function and quantify the Experimental Design. The default is "fisher_determinant".

        - fisher_determinant
            Use the D-optimality criterion that maximizes the determinant of the Fisher Information matrix.
        - fisher_mineigenval
            Use the E-optimality criterion that maximizes the minimal eigenvalue of the Fisher Information matrix.
        - fisher_sumeigenval
            Use the A-optimality criterion that maximizes the sum of all eigenvalues of the Fisher Information matrix.
        - fisher_ratioeigenval
            Use the modified E-optimality criterion that maximizes the ratio of the minimal and maximal eigenvalues of the Fisher Information matrix.

    :type criterion: callable
    :type optimization_strategy: str
    :raises KeyError: Raised if the chosen optimization strategy is not implemented.
    :return: The result of the optimization as an object *FisherResults*. Important attributes are the conditions of the Optimal Experimental Design *times*, *inputs*, the resultion value of the objective function *criterion*.
    :rtype: FisherResults
    """
    fsmp = FisherModelParametrized.init_from(fsm)

    optimization_strategies = {
        "scipy_differential_evolution": __scipy_differential_evolution,
        "scipy_brute": __scipy_brute,
        "scipy_basinhopping": __scipy_basinhopping
    }
    if optimization_strategy not in optimization_strategies.keys():
        raise KeyError("Please specify one of the following optimization_strategies for optimization: " + str(optimization_strategies.keys()))

    return optimization_strategies[optimization_strategy](fsmp, **args)
