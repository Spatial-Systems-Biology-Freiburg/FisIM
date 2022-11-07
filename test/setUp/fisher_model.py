import numpy as np
import pytest

from FisInMa import FisherModel, FisherModelParametrized


def f(t, x, inputs, params, consts):
    A, B = x
    T, Q = inputs
    p, q, w = params
    c, d, e = consts[:3]
    return [
        - p*A**2 + p*B + c*(T**2/(e**2+T**2)) +w,
        e*q*A + B + Q + d
    ]


def dfdx(t, x, inputs, params, consts):
    A, B = x
    T, Q = inputs
    p, q, w = params
    c, d, e = consts[:3]
    return [
        [-2*p*A, p],
        [e*q, 1]
    ]


def dfdp(t, x, inputs, params, consts):
    A, B = x
    T, Q = inputs
    p, q, w = params
    c, d, e = consts[:3]
    return [
        [-A**2 + B, 0, 1],
        [0, e*A, 0]
    ]


def g(t, x, inputs, params, consts):
    A, B = x
    return A


def dgdx(t, x, inputs, params, consts):
    A, B = x
    return [1, 0]


def dgdp(t, x, inputs, params, consts):
    return [0 ,0, 0]


class Setup_Class:
    def __init__(self, N_x0=2, n_t0=13, n_times=5, n_inputs_0=7, n_inputs_1=11, identical_times=False):
        # Use prime numbers for sampled parameters to 
        # show errors in code where reshaping is done
        self.x0=[np.array([0.05 / i, 0.001 / i]) for i in range(1, N_x0+1)]
        self.t0=np.linspace(0.0, 0.01, n_t0)
        self.times=np.linspace(0.1, 10.0, n_times)
        self.inputs=[
            np.arange(2, n_inputs_0 + 2),
            np.arange(5, n_inputs_1 + 5)
        ]
        
        self.parameters=(2.95, 8.4768, 0.001)
        # n_ode_args = 3
        self.ode_args=(1.0, 2.0, 1.5)
        self.fsm = FisherModel(
            ode_fun=f,
            ode_dfdx=dfdx,
            ode_dfdp=dfdp,
            ode_x0=self.x0,
            ode_t0=self.t0,
            times=self.times,
            inputs=self.inputs,
            parameters=self.parameters,
            ode_args=self.ode_args,
            obs_fun=g,
            obs_dgdx=dgdx,
            obs_dgdp=dgdp,
            identical_times=identical_times,
        )
        self.fsmp = FisherModelParametrized.init_from(self.fsm)


@pytest.fixture()
def default_model():
    return Setup_Class()

@pytest.fixture()
def default_model_parametrized(N_x0, n_t0, n_times, n_inputs_0, n_inputs_1, identical_times):
    return Setup_Class(N_x0, n_t0, n_times, n_inputs_0, n_inputs_1, identical_times)

@pytest.fixture()
def default_model_small(identical_times):
    return Setup_Class(N_x0=1, n_t0=1, n_times=1, n_inputs_0=1, n_inputs_1=1, identical_times=identical_times)
