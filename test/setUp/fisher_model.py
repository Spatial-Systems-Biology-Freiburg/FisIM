import unittest
import numpy as np
import copy

from FisInMa import FischerModel, FischerModelParametrized


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


class Setup_Class(unittest.TestCase):
    @classmethod
    def setUpClass(self, N_y0=2, n_t0=13, n_times=5, n_inputs=(7, 11), identical_times=False):
        # Use prime numbers for sampled parameters to 
        # show errors in code where reshaping is done
        self.y0=[np.array([0.05 / i, 0.001 / i]) for i in range(1, N_y0+1)]
        self.t0=np.linspace(0.0, 0.01, n_t0)
        self.times=np.linspace(0.1, 10.0, n_times)
        self.inputs=[
            np.arange(2, n_inputs[0]+2),
            np.arange(5, n_inputs[1]+5)
        ]
        
        self.parameters=(2.95, 8.4768, 0.001)
        # n_constants = 3
        self.constants=(1.0, 2.0, 1.5)
        self.fsm = FischerModel(
            ode_fun=f,
            ode_dfdx=dfdx,
            ode_dfdp=dfdp,
            ode_y0=self.y0,
            ode_t0=self.t0,
            times=self.times,
            inputs=self.inputs,
            parameters=self.parameters,
            constants=self.constants,
            obs_fun=g,
            obs_dfdx=dgdx,
            obs_dfdp=dgdp,
            identical_times=identical_times,
        )
        self.fsmp = FischerModelParametrized.init_from(self.fsm)