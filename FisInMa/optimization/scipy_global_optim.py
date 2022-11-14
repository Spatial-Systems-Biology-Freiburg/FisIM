import numpy as np
import scipy as sp
import scipy.optimize as optimize
import itertools

from FisInMa.model import FisherModel, FisherModelParametrized, VariableDefinition, MultiVariableDefinition
from FisInMa.solving import calculate_fisher_criterion, fisher_determinant
from .penalty import _discrete_penalizer


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


def __scipy_optimizer_function(X, fsmp: FisherModelParametrized, full=False, discrete_penalizer="default", kwargs_dict={}):
    total = 0
    # Get values for ode_t0
    if fsmp.ode_t0_def is not None:
        # TODO test these statements
        fsmp.ode_t0 = X[:fsmp.ode_t0_def.n]
        total += fsmp.ode_t0_def.n
    
    # Get values for ode_x0
    if fsmp.ode_x0_def is not None:
        # TODO test these statements
        n_x = len(fsmp.ode_x0[0])
        temp = X[total:total + fsmp.ode_x0_def.n * n_x].reshape(fsmp.ode_x0_def.n, n_x)
        fsmp.ode_x0 = [t for t in temp]
        total += fsmp.ode_x0_def.n * n_x

    # Get values for times
    if fsmp.times_def is not None:
        fsmp.times = np.sort(X[total:total+fsmp.times.size].reshape(fsmp.times.shape), axis=-1)
        total += fsmp.times.size

    # Get values for inputs
    for i, inp_def in enumerate(fsmp.inputs_def):
        if inp_def is not None:
            # TODO test these statements
            fsmp.inputs[i]=X[total:total+inp_def.n]
            total += inp_def.n

    # Calculate the correct criterion
    fsr = calculate_fisher_criterion(fsmp, **kwargs_dict)

    # Calculate the discretization penalty
    penalty, penalty_summary = _discrete_penalizer(fsmp, discrete_penalizer)
    
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
    if type(fsmp.ode_x0_def)==MultiVariableDefinition:
        # Bounds for value
        # TODO test these statements
        for lb_i, ub_i in zip(fsmp.ode_x0_def.lb, fsmp.ode_x0_def.ub):
            lb += [lb_i] * fsmp.ode_x0_def.n
            ub += [ub_i] * fsmp.ode_x0_def.n
        
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


def __update_arguments(optim_func, optim_args, kwargs):
    # Gather all arguments which can be supplied to the optimization function and check for intersections
    o_keys = set(optim_func.__code__.co_varnames)

    # Take all keys which are ment to go into the routine and put it in the corresponding dictionary
    intersect = {key: kwargs.pop(key) for key in o_keys & kwargs.keys()}

    # Update the arguments for the optimization routine. Pass everything else to our custom methods.
    optim_args.update(intersect)

    return optim_args, kwargs


def __scipy_differential_evolution(fsmp: FisherModelParametrized, discrete_penalizer="default", **kwargs):
    # Create bounds, constraints and initial guess
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
    x0 = __initial_guess(fsmp)

    opt_args = {
        "func": __scipy_optimizer_function,
        "bounds": bounds,
        "args":(fsmp, False, discrete_penalizer, kwargs),
        "disp": True,
        "polish": True,
        "updating": 'deferred',
        "workers": -1,
        #"constraints":constraints,
        "x0": x0
    }

    # Check for intersecting arguments and update the default arguments in opt_args with arguments from kwargs.
    opt_args, kwargs = __update_arguments(optimize.differential_evolution, opt_args, kwargs)

    # Actually call the optimization function
    res = optimize.differential_evolution(**opt_args)

    # Return the full result
    return __scipy_optimizer_function(res.x, fsmp, full=True, discrete_penalizer=discrete_penalizer, kwargs_dict=kwargs)


def __scipy_brute(fsmp: FisherModelParametrized, discrete_penalizer="default", **kwargs):
    # Create bounds and constraints
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)

    opt_args = {
        "func": __scipy_optimizer_function,
        "ranges": bounds,
        "args":(fsmp, False, discrete_penalizer, kwargs),
        "Ns":3,
        "full_output":0,
        "finish": None,
        "disp":True,
        "workers":-1
    }

    # Check for intersecting arguments and update the default arguments in opt_args with arguments from kwargs.
    opt_args, kwargs = __update_arguments(optimize.brute, opt_args, kwargs)

    # Actually call the optimization function
    res = optimize.brute(**opt_args)

    return __scipy_optimizer_function(res, fsmp, full=True, discrete_penalizer=discrete_penalizer, kwargs_dict=kwargs)


def __scipy_basinhopping(fsmp: FisherModelParametrized, discrete_penalizer="default", **kwargs):
    # Create bounds, constraints and initial guess
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
    x0 = __initial_guess(fsmp)

    opt_args = {
        "func": __scipy_optimizer_function,
        "x0": x0,
        "minimizer_kwargs":{"args":(fsmp, False, discrete_penalizer, kwargs), "bounds": bounds},
        "disp":True,
    }

    # Check for intersecting arguments and update the default arguments in opt_args with arguments from kwargs.
    opt_args, kwargs = __update_arguments(optimize.basinhopping, opt_args, kwargs)

    # Actually call the optimization function
    res = optimize.basinhopping(**opt_args)

    return __scipy_optimizer_function(res.x, fsmp, full=True, discrete_penalizer=discrete_penalizer, kwargs_dict=kwargs)
