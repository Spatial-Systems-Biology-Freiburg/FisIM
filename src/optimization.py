import numpy as np
import scipy as sp
import itertools

from src.data_structures import FischerModel
from src.solving import calculate_fischer_observable, fischer_determinant


def __scipy_minimize_optimizer_function(X, fsm: FischerModel, full=False):
    if fsm._times_1d == False:
        times = np.sort(X.reshape(fsm.times.shape), axis=-1)
    else:
        times = np.sort(X)
    
    fsm.set_times(times)
    fsr = calculate_fischer_observable(fsm, False, fsm.relative_sensitivities)

    if full:
        return fsr
    return - fsr.observable


def __scipy_minimize(times0, tmax, fsm: FischerModel, method="COBYLA", **args):
    x0 = times0.flatten()

    # Define linear constraints on times
    # Constraints are t0 <= t1 <= t2 ...
    # and tmin <= ti <= tmax
    n_times = fsm.times.shape[-1]
    A = np.zeros((n_times*2-1, n_times))
    for i in range(n_times-1):
        A[i][i] = 1.0
        A[i][i+1] = -1.0
    for i in range(n_times):
        A[i+n_times-1][i] = 1.0

    if fsm._times_1d == True:
        n_times = len(x0)
        B = A
    else:
        n_times = fsm.times.shape[-1]
        n_q_values = np.product([len(q) for q in fsm.q_values])
        B = np.zeros(((n_times*2 -1) * n_q_values, n_q_values * n_times))
        for i in range(n_q_values):
            tmp = np.concatenate([np.zeros((n_times*2-1, n_times)) for _ in range(i)] + [A] + [np.zeros((n_times*2-1, n_times)) for _ in range(n_q_values-1-i)], axis=1)
            B[i*(2*n_times-1):i*(2*n_times-1)+2*n_times-1] = tmp
    
    ub = np.concatenate([np.append(np.zeros(n_times-1), np.full((n_times,), tmax))] * n_q_values)
    lb = np.concatenate([np.append(- np.inf * np.ones(n_times-1), np.full((n_times,), fsm.y0_t0[1]))] * n_q_values)
    constraints = sp.optimize.LinearConstraint(B, lb, ub)

    res = sp.optimize.minimize(
        __scipy_minimize_optimizer_function,
        x0,
        args=fsm,
        constraints=constraints,
        method=method,
        **args
    )

    t = np.sort(res.x[:np.product(times0.shape)].reshape(times0.shape), axis=-1)
    fsm.times = t
    fsr = __scipy_minimize_optimizer_function(res.x, fsm, full=True)
    return fsr


def find_optimal(times0, tmax, fsm: FischerModel, optimization_strategy: str, **args):
    optimization_strategies = {
        "scipy_minimize": __scipy_minimize
    }
    if optimization_strategy not in optimization_strategies.keys():
        raise KeyError("Please specify one of the following optimization_strategies for optimization: " + str(optimization_strategies.keys()))

    return optimization_strategies[optimization_strategy](times0, tmax, fsm, **args)
