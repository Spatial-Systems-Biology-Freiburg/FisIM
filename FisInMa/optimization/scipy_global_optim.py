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


def __scipy_optimizer_function(X, fsmp: FisherModelParametrized, discrete=None, full=False):
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


def _scipy_calculate_bounds_constraints(fsmp: FisherModelParametrized, min_distance=None):
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

    print("")

    n_mut_check = []
    # Go through all possibly mutable variables and gather information about constraints and bounds
    # Check if initial times are sampled over
    if type(fsmp.ode_t0_def)==VariableDefinition:
        print("Sampling initial times")
        # Bounds for value
        lb += [fsmp.ode_t0_def.lb] * fsmp.ode_t0_def.n
        ub += [fsmp.ode_t0_def.ub] * fsmp.ode_t0_def.n
        
        # Constraints on variables
        lc += [-np.inf] * fsmp.ode_t0_def.n
        uc += [fsmp.ode_t0_def.min_distance if fsmp.ode_t0_def.min_distance!=None else np.inf] * fsmp.ode_t0_def.n
        
        # Define matrix A which will extend B
        A = np.eye(fsmp.ode_t0_def.n)
        B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])
        n_mut_check.append(fsmp.ode_t0_def.n)
    else:
        n_mut_check.append(0)
        print("Not sampling initial times")

    # Check if initial values are sampled over
    if type(fsmp.ode_y0_def)==VariableDefinition:
        print("Sampling initial values")
        # Bounds for value
        lb.append(fsmp.ode_y0_def.lb)
        ub.append(fsmp.ode_y0_def.ub)
        
        # Constraints on variables
        lc += []
        uc += []

        # Define matrix A which will extend B
        A = np.eye(len(fsmp.ode_y0 * fsmp.ode_y0_def.n))
        B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])
        n_mut_check.append(A.shape[1])
    else:
        n_mut_check.append(0)
        print("Not sampling initial values")

    # Check if times are sampled over
    if type(fsmp.times_def)==VariableDefinition:
        print("Sampling times")
        # How many time points are we sampling?
        n_times = np.product(fsmp.times.shape)

        # Store lower and uppwer bound
        lb += [fsmp.times_def.lb] * n_times
        ub += [fsmp.times_def.ub] * n_times

        #

        # Create the correct matrix to store.
        A = np.zeros((n_times*2-1, n_times))
        for i in range(n_times-1):
            A[i][i] = 1.0
            A[i][i+1] = -1.0
        for i in range(n_times):
            A[i+n_times-1][i] = 1.0
        B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])
        n_mut_check.append(A.shape[1])
    else:
        n_mut_check.append(0)
        print("Not sampling times")

    # Check which inputs are sampled
    i = 0
    for inp_def in fsmp.inputs_def:
        if type(inp_def)==VariableDefinition:
            lb += [inp_def.lb] * inp_def.n
            ub += [inp_def.ub] * inp_def.n
            A = np.eye(inp_def.n)
            B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])
            n_mut_check.append(A.shape[1])
            print("Sampling input", i)
        else:
            n_mut_check.append(0)
            print("Not sampling input", i)
        i += 1

    print(lb)
    print(ub)
    print(B)
    print("Checking if all mutable variables have been identified:", n_mut, n_mut_check)
    print("Checking if B matrix has correct shape:", B.shape[1], np.sum(n_mut))
    print("Checking if lb and ub have same shape as B", len(lc), len(uc), B.shape[0])

    bounds = list(zip(lb, ub))
    # constraints = scipy.LinearConstraint()
    constraints = 1
    print("")
    return bounds, constraints
    
    # Check if initial values are sampled over
    # if fsmp.ode_y0_def!=None:









    # Define linear constraints on times
    # Constraints are t0 <= t1 <= t2 ...
    # and tmin <= ti <= tmax
    
    n_times_mut = np.product(fsmp.times_mut.shape if fsmp.times_mut!=None else [0])

    # Check if we are changing the times
    if n_times_mut > 0:
        # Create time upper/lower values for lb_t <= B.x <= ub_t and matrix A
        # First create matrix block A
        A = np.zeros((max(0, n_times_mut*2-1), n_times_mut))
        for i in range(n_times_mut-1):
            A[i][i] = 1.0
            A[i][i+1] = -1.0
        for i in range(n_times_mut):
            A[i+n_times_mut-1][i] = 1.0
        
        # Now if times are not individual, the created matrix block needs to be copied multiple times
        if fsmp.identical_times == True:
            B = A
            ub_t = np.append(np.full(n_times_mut-1, 0.0 if min_distance==None else -min_distance), np.full((n_times_mut,), fsmp.times_def.ub))
            lb_t = np.append(np.full(n_times_mut-1, - np.inf), np.full((n_times_mut,), fsmp.times_def.lb))
        else:
            B = np.zeros(((max(0, n_times_mut*2 -1)) * n_inputs_mut, n_inputs_mut * n_times_mut))
            # for i in range(n_inputs_mut):
            #     tmp = np.concatenate([np.zeros((n_times_mut*2-1, n_times_mut)) for _ in range(i)] + [A] + [np.zeros((n_times_mut*2-1, n_times_mut)) for _ in range(n_inputs_mut-1-i)], axis=1)
            #     B[i*(2*n_times_mut-1):i*(2*n_times_mut-1)+2*n_times_mut-1] = tmp
            B = np.block(
                [np.zeros(2*n_times_mut-1, n_times_mut)] * i + [A] + [np.zeros(2*n_times_mut-1, n_times_mut)] * n_inputs_mut-1-i
                for i in range(n_inputs_mut)
            )
            print(B)
        
            ub_t = [0]#np.concatenate([np.append(np.full(max(0, n_times_mut-1), 0.0 if min_distance==None else -min_distance), np.full((n_times_mut,), fsmp.times_def.ub))] * n_inputs_mut)
            lb_t = [0]#np.concatenate([np.append(np.full(max(0, n_times_mut-1), - np.inf), np.full((n_times_mut,), fsmp.times_def.lb))] * n_inputs_mut)
    
    # Define linear constraints for 
    n_inputs_mut = np.product([len(q) if q!=None else 1 for q in fsmp.inputs_mut])
    ub = np.concatenate([np.append(np.full(max(0, n_times_mut-1), 0.0 if min_distance==None else -min_distance), np.full((n_times_mut,), fsmp.times_def.ub))] * n_q_values)
    lb = np.concatenate([np.append(np.full(max(0, n_times_mut-1), - np.inf), np.full((n_times_mut,), fsmp.times_def.lb))] * n_q_values)

    constraints = sp.optimize.LinearConstraint(B, lb, ub)

    #
    bounds_t = [(fsmp.times_def.lb, fsmp.times_def.ub) for _ in range(n_times_mut)]
    bounds_q = [x[0:2] for x in fsmp.q_mod_values_ranges for _ in range(x[2])]
    bounds = bounds_t + bounds_q

    return bounds, constraints


def __handle_custom_options(**args):
    custom_args = {}

    # Determine if times should have a minimum distance between each other
    if "min_distance" not in args.keys():
        min_distance=False
    else:
        min_distance=args.pop("min_distance")
    
    # Determine if results should be discretized
    if "discrete" not in args.keys():
        discrete=None
    else:
        discrete=args.pop("discrete")
        try:
            iter(discrete)
        except:
            discrete = (discrete, 0.0)
        if min_distance!=False:
            print("Warning: option 'discrete' overwrites option 'min_distance'")
        min_distance=discrete[0]
    
    custom_args["min_distance"] = min_distance
    custom_args["discrete"] = discrete
    return args, custom_args


def __scipy_differential_evolution(fsmp: FisherModelParametrized, **args):
    # Filter custom options and scipy options
    args, custom_args = __handle_custom_options(**args)
    
    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp, custom_args["min_distance"])

    x0 = np.concatenate((times0.flatten(), *q_values0))

    opt_args = {
        "func": __scipy_optimizer_function,
        "bounds": bounds,
        "constraints":constraints,
        "args":(fsmp, custom_args["discrete"]),
        "polish":False,
        "workers":-1,
        "updating":'deferred',
        "x0": x0
    }
    opt_args.update(args)
    res = sp.optimize.differential_evolution(**opt_args)

    return __scipy_optimizer_function(res.x, fsmp, full=True)


def __scipy_brute(times0, fsmp: FisherModelParametrized, **args):
    # Filter custom options and scipy options
    args, custom_args = __handle_custom_options(**args)

    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp, custom_args["min_distance"])

    opt_args = {
        "func": __scipy_optimizer_function,
        "ranges": bounds,
        "args":(fsmp, custom_args["discrete"]),
        "finish":False,
        "workers":-1,
        "Ns":5
    }
    opt_args.update(args)
    res = sp.optimize.brute(**opt_args)

    return __scipy_optimizer_function(res, fsmp, full=True)


def __scipy_basinhopping(times0, fsmp: FisherModelParametrized, **args):
    # Filter custom options and scipy options
    args, custom_args = __handle_custom_options(**args)

    # Create constraints and bounds
    bounds, constraints = _scipy_calculate_bounds_constraints(fsmp, custom_args["min_distance"])

    opt_args = {
        "func": __scipy_optimizer_function,
        "x0": times0.flatten(),
        "minimizer_kwargs":{"args":(fsmp, custom_args["discrete"]), "constraints": constraints, "bounds": bounds}
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
