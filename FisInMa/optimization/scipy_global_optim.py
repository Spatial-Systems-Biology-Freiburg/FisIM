import numpy as np
import scipy as sp
import scipy.optimize as optimize
import itertools

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


def _discrete_penalizer(x, dx, x_offset=0.0):
    y = x - x_offset
    n, p = np.divmod(y, dx)
    _, q = np.divmod((n+1) * dx - y, dx)
    r = np.array([p, q]).min(axis=0)
    return 1 - 2 * r / dx


def __scipy_optimizer_function(X, fsmp: FisherModelParametrized, full=False):
    total = 0
    # Get values for ode_t0
    if fsmp.ode_t0_def is not None:
        fsmp.ode_t0 = X[:fsmp.ode_t0_def.n]
        total += fsmp.ode_t0_def.n
    
    # Get values for ode_y0
    if fsmp.ode_y0_def is not None:
        fsmp.ode_y0 = X[total:total + fsmp.ode_y0_def.n * fsmp.ode_y0.size]
        total += fsmp.ode_y0_def.n

    # Get values for times
    if fsmp.times_def is not None:
        fsmp.times = np.sort(X[total:total+fsmp.times.size].reshape(fsmp.times.shape), axis=-1)
        total += fsmp.times.size

    # Get values for inputs
    for i, inp_def in enumerate(fsmp.inputs_def):
        if inp_def is not None:
            fsmp.inputs[i]=X[total:total+inp_def.n]
            total += inp_def.n

    fsr = calculate_fisher_criterion(fsmp)

    if full:
        return fsr
    return -fsr.criterion# * _discrete_penalizer(fsr)


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
        fsmp.ode_y0_def.n if fsmp.ode_y0_def is not None else 0,
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
    if type(fsmp.ode_y0_def)==VariableDefinition:
        # Bounds for value
        lb.append(fsmp.ode_y0_def.lb)
        ub.append(fsmp.ode_y0_def.ub)
        
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
        # lc += [-np.inf] * (n_times-1) + [fsmp.times_def.lb] * n_times

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


def __scipy_differential_evolution(fsmp: FisherModelParametrized, **args):
    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)

    x0 = np.concatenate([
        np.array(fsmp.ode_y0).flatten() if fsmp.ode_y0_def is not None else [],
        np.array(fsmp.ode_t0).flatten() if fsmp.ode_t0_def is not None else [],
        np.array(fsmp.times).flatten() if fsmp.times_def is not None else [],
        *[
            np.array(inp_mut_val).flatten() if inp_mut_val is not None else []
            for inp_mut_val in fsmp.inputs_mut
        ]
    ])

    opt_args = {
        "func": __scipy_optimizer_function,
        "bounds": bounds,
        "constraints":constraints,
        "args":(fsmp,),
        "polish":False,
        "workers":-1,
        "updating":'deferred',
        "x0": x0
    }
    opt_args.update(args)
    res = optimize.differential_evolution(**opt_args)

    return __scipy_optimizer_function(res.x, fsmp, full=True)


def __scipy_brute(times0, fsmp: FisherModelParametrized, **args):
    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)

    opt_args = {
        "func": __scipy_optimizer_function,
        "ranges": bounds,
        "args":(fsmp,),
        "finish":False,
        "workers":-1,
        "Ns":5
    }
    opt_args.update(args)
    res = optimize.brute(**opt_args)

    return __scipy_optimizer_function(res, fsmp, full=True)


def __scipy_basinhopping(times0, fsmp: FisherModelParametrized, **args):
    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)

    opt_args = {
        "func": __scipy_optimizer_function,
        "x0": times0.flatten(),
        "minimizer_kwargs":{"args":(fsmp,), "constraints": constraints, "bounds": bounds}
    }
    opt_args.update(args)
    res = optimize.basinhopping(**opt_args)

    return __scipy_optimizer_function(res.x, fsmp, full=True)


def find_optimal(fsm: FisherModel, optimization_strategy: str, **args):
    """Find the global optimum of the supplied FisherModel.

    :param fsm: _description_
    :type fsm: FisherModel
    :param optimization_strategy: _description_
    :type optimization_strategy: str
    :raises KeyError: _description_
    :return: _description_
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
