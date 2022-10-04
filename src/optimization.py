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


def __scipy_minimize(times0, bounds, fsm: FischerModel):
    x0 = times0.flatten()

    res = sp.optimize.minimize(
        __scipy_minimize_optimizer_function,
        x0,
        args=fsm,
        bounds=bounds
    )

    t = np.sort(res.x[:np.product(times0.shape)].reshape(times0.shape), axis=-1)
    fsm.times = t
    fsr = __scipy_minimize_optimizer_function(res.x, fsm, full=True)
    return fsr


def find_optimal(times0, bounds, fsm: FischerModel, method: str):
    methods = {
        "scipy_minimize": __scipy_minimize
    }
    if method not in methods.keys():
        raise KeyError("Please specify one of the following methods for optimization: " + str(methods.keys()))
    
    return methods[method](times0, bounds, fsm)