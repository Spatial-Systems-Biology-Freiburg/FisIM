import numpy as np
import scipy as sp
import itertools

from FisInMa.model import FisherModel, FisherModelParametrized, VariableDefinition
from FisInMa.solving import calculate_fisher_criterion, fisher_determinant


def discrete_penalizer(x, dx, x_offset=0.0):
    y = x - x_offset
    n, p = np.divmod(y, dx)
    _, q = np.divmod((n+1) * dx - y, dx)
    r = np.array([p, q]).min(axis=0)
    return 1 - 2 * r / dx


def __scipy_optimizer_function(X, fsmp: FisherModelParametrized, full=False):
    if fsmp.individual_times==True:
        m = np.product(fsmp._times_shape)
    else:
        m = fsmp.times.shape[-1]
    t = X[:m]
    q = X[m:]
    if fsmp.individual_times == True:
        times = np.sort(t.reshape(fsmp._times_shape), axis=-1)
    else:
        times = np.sort(t)
    
    q_values = []
    tot = 0
    for _, _, k in fsmp.q_mod_values_ranges:
        q_values.append(q[tot:tot+k])
        tot += k

    fsmp.set_times(times)
    fsmp.set_q_values_mod(q)

    fsr = calculate_Fisher_criterion(fsmp, False, fsmp.relative_sensitivities)

    if full:
        return fsr
    if discrete!=None:
        return -fsr.criterion * np.product(discrete_penalizer(fsmp.times.flatten(), discrete[0], discrete[1]))
    return - fsr.criterion


def _scipy_calculate_bounds_constraints(fsmp: FisherModelParametrized):
    # Define array for upper and lower bounds
    ub = []
    lb = []
    
    # Define constraints via equation lc <= B.dot(x) uc
    # lower and upper constraints lc, uc and matrix B
    lc = []
    uc = []

    # Determine the number of mutable variables which can be sampled over
    n_times = np.product(fsmp.times.shape) if fsmp.times_def !=None else 0
    n_inputs = [len(q) if q_def!=None else 0 for q, q_def in zip(fsmp.inputs, fsmp.inputs_def)]
    n_mut = [
        fsmp.ode_t0_def.n if fsmp.ode_t0_def!=None else 0,
        fsmp.ode_y0_def.n if fsmp.ode_y0_def!=None else 0,
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
        lc += [-np.inf] * fsmp.ode_t0_def.n
        uc += [fsmp.ode_t0_def.min_distance if fsmp.ode_t0_def.min_distance!=None else np.inf] * fsmp.ode_t0_def.n
        
        # Define matrix A which will extend B
        A = np.eye(fsmp.ode_t0_def.n)
        B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])

    # Check if initial values are sampled over
    if type(fsmp.ode_y0_def)==VariableDefinition:
        # print("Sampling initial values")
        # Bounds for value
        lb.append(fsmp.ode_y0_def.lb)
        ub.append(fsmp.ode_y0_def.ub)
        
        # Constraints on variables
        lc += []
        uc += []

        # Define matrix A which will extend B
        A = np.eye(len(fsmp.ode_y0 * fsmp.ode_y0_def.n))
        B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])

    # Check if times are sampled over
    if type(fsmp.times_def)==VariableDefinition:
        print("Sampling times")
        # How many time points are we sampling?
        n_times = np.product(fsmp.times.shape)

        # Store lower and upper bound
        lb += [fsmp.times_def.lb] * n_times
        ub += [fsmp.times_def.ub] * n_times

        # Constraints on variables
        lc += [-np.inf] * (n_times-1) + [fsmp.times_def.lb] * n_times
        uc += [-fsmp.times_def.min_distance if fsmp.times_def.min_distance!=None else np.inf] * (n_times-1) + [fsmp.times_def.ub] * n_times

        # Create the correct matrix to store.
        A = np.zeros((n_times*2-1, n_times))
        # Fill the matrix like so:
        #
        #     | 1 -1  0  0 ... |
        # A = | 0  1 -1  0 ... |
        #     | 0  0  1 -1 ... |
        #     | ...            |
        #
        for i in range(n_times-1):
            A[i][i] = 1.0
            A[i][i+1] = -1.0
        for i in range(n_times):
            A[i+n_times-1][i] = 1.0
        B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])

    # Check which inputs are sampled
    for inp_def in fsmp.inputs_def:
        if type(inp_def)==VariableDefinition:
            # Store lower and upper bound
            lb += [inp_def.lb] * inp_def.n
            ub += [inp_def.ub] * inp_def.n

            # Constraints on variables
            m = inp_def.min_distance if inp_def.min_distance!=None else -np.inf
            lc += [+m] * max(1, int(round(inp_def.n*(inp_def.n-1)/2,0)))
            uc += [-m] * max(1, int(round(inp_def.n*(inp_def.n-1)/2,0)))

            # Create correct matrix matrix to store
            A = np.zeros((int(round(inp_def.n*(inp_def.n-1)/2,0)), inp_def.n))
            count = 0
            for i in range(inp_def.n-1):
                A[count:count+inp_def.n-i-1,i] = 1.0
                A[count:count+inp_def.n-i-1,i+1:] = -np.eye(inp_def.n-i-1)
                count += inp_def.n-i-1
            # Fill the matrix like so:
            #
            #     | 1 -1  0  0 ... |
            #     | 1  0 -1  0 ... |
            #     | 1  0  0 -1 ... |
            # A = | ...            |
            #     | 0  1 -1  0 ... |
            #     | 0  1  0 -1 ... |
            #     | ...            |
            # 
            # We denote the vector of mutable variables with
            #   x = [v0, ... vN]
            # and plug it into our linear system
            #   lc <= A x <= uc
            # Together with lower and upper constraints set to +d, -d respectively, we obtain the following equations:
            # d <= vi - vj <= -d
            # Or seperately
            # d <= vi - vj
            # d <= vj - vi
            # which imposes the min_distance condition
            B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])

    bounds = list(zip(lb, ub))
    constraints = sp.optimize.LinearConstraint(B, lc, uc)
    return bounds, constraints


def __scipy_differential_evolution(fsmp: FisherModelParametrized, **args):
    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)

    x0 = np.concatenate((times0.flatten(), *q_values0))

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
    res = sp.optimize.differential_evolution(**opt_args)

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
    res = sp.optimize.brute(**opt_args)

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
    res = sp.optimize.basinhopping(**opt_args)

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
