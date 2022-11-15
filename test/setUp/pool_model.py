import numpy as np
import pytest

from FisInMa import *


def pool(t, y, Q, P, Const):
    a, b = P
    Temp, = Q
    n0, c, n_max = Const
    n, = y
    return [(a*Temp + c) * (n - n0*np.exp(-b*Temp*t))*(1-n/n_max)]

def pool_dfdx(t, y, Q, P, Const):
    a, b = P
    Temp, = Q
    n0, c, n_max = Const
    n, = y
    return (a*Temp + c) * (1-n/n_max) + (a*Temp + c) * (n - n0*np.exp(-b*Temp*t))*(-1/n_max)

def pool_dfdp(t, y, Q, P, Const):
    a, b = P
    Temp, = Q
    n0, c, n_max = Const
    n, = y
    return [
        (Temp) * (n - n0*np.exp(-b*Temp*t))*(1-n/n_max),
        (a*Temp + c) * (Temp*t*n0*np.exp(-b*Temp*t))*(1-n/n_max)
    ]

def pool_dfdx0(t, y, Q, P, Const):
    a, b = P
    Temp, = Q
    n0, c, n_max = Const
    n, = y
    return [
        [(a*Temp + c) * (- np.exp(-b*Temp*t))*(1-n/n_max)]
    ]

def pool_log(t, y, Q, P, Const):
    n0, c, n_max = Const
    n, = y
    return [np.log((n+n0)/n0)]

def pool_log_dgdx(t, y, Q, P, Const):
    n0, c, n_max = Const
    n, = y
    return [
        [1 / (n+n0)]
    ]

def pool_log_dgdp(t, y, Q, P, Const):
    return [
        [0, 0]
    ]

def pool_log_dgdx0(t, y, Q, P, Const):
    n0, c, n_max = Const
    n, = y
    return [
        [- n /(n0*(n+n0))]
    ]


class ModelParamInitialValues:
    def __init__(self, n_times=2, n_temps=1, identical_times=True):
        # Define ode_args for the simulation duration
        c = 1.31
        n0 = 0.25
        n_max = 2e4
        self.ode_args = (n0, c, n_max)

        # Define initial parameter guesses
        self.parameters = (0.065, 1e-3)

        # Initial values for complete ODE (with S-Terms)
        self.t0 = 0.0
        self.x0 = n0

        self.times = {"lb":self.t0, "ub":16.0, "n":n_times}

        # Values for temperatures
        self.inputs = [
            np.linspace(2.0, 20.0, n_temps)
        ]

        self.fsm = FisherModel(
            ode_fun=pool,
            ode_dfdx=pool_dfdx,
            ode_dfdp=pool_dfdp,
            ode_dfdx0=pool_dfdx0,
            ode_t0=self.t0,
            ode_x0=self.x0,
            times=self.times,
            inputs=self.inputs,
            parameters=self.parameters,
            ode_args=self.ode_args,
            obs_fun=pool_log,
            obs_dgdx=pool_log_dgdx,
            obs_dgdp=pool_log_dgdp,
            obs_dgdx0=pool_log_dgdx0,
            identical_times=identical_times,
        )
        self.fsmp = FisherModelParametrized.init_from(self.fsm)


@pytest.fixture()
def pool_model():
    return ModelParamInitialValues()

@pytest.fixture()
def pool_model_parametrized(n_times, n_temps, identical_times):
    return ModelParamInitialValues(n_times, n_temps, identical_times)

@pytest.fixture()
def pool_model_small(identical_times):
    return ModelParamInitialValues(n_times=1, n_temps=1, identical_times=identical_times)
