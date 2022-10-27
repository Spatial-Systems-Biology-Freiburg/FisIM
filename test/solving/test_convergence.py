import unittest
import numpy as np
import copy
import scipy as sp
import time

from FisInMa.model import FisherModel, FisherModelParametrized
from FisInMa.solving import *

from test.setUp import Setup_Class

# Define a RHS of ODE where exact result is known
def ode_fun(t, x, inputs, parameters, ode_args):
    (A, B) = x
    (T,) = inputs
    (a, b) = parameters
    return [
        a - b*T*A,
        b - a*T*B
    ]

def ode_dfdx(t, x, inputs, parameters, ode_args):
    (A, B) = x
    (T,) = inputs
    (a, b) = parameters
    return [
        [- b*T, 0],
        [0, - a*T]
    ]

def ode_dfdp(t, x, inputs, parameters, ode_args):
    (A, B) = x
    (T,) = inputs
    (a, b) = parameters
    return [
        [1, - A*T],
        [- B*T, 1]
    ]

def ode_exact(t, x0, inputs, parameters, ode_args):
    (T,) = inputs
    (a, b) = parameters
    return [
        a/(b*T) - x0/(b*T) * np.exp(-b*T*t),
        b/(a*T) - x0/(a*T) * np.exp(-a*T*t)
    ]

def g(t, x, inputs, params, consts):
    (A, B) = x
    return A

def dgdx(t, x, inputs, params, consts):
    (A, B) = x
    return [1, 0]

def dgdp(t, x, inputs, params, consts):
    return [0 ,0, 0]


class Setup_Convergence(unittest.TestCase):
    @classmethod
    def setUp(self, n_times=4, n_inputs=3, identical_times=False):
        self.y0=[1.0, 0.5]
        self.t0=0.0
        self.times=np.linspace(0.0, 10.0, n_times)
        self.inputs=[
            np.linspace(0.8, 1.2, n_inputs)
        ]
        
        self.parameters=(2.388, 7.4234)
        # n_ode_args = 3
        self.ode_args=None
        self.fsm = FisherModel(
            ode_fun=ode_fun,
            ode_dfdx=ode_dfdx,
            ode_dfdp=ode_dfdp,
            ode_y0=self.y0,
            ode_t0=self.t0,
            times=self.times,
            inputs=self.inputs,
            parameters=self.parameters,
            ode_args=self.ode_args,
            obs_fun=g,
            obs_dfdx=dgdx,
            obs_dfdp=dgdp,
            identical_times=identical_times,
        )
        self.fsmp = FisherModelParametrized.init_from(self.fsm)


class TestConvergence(Setup_Convergence):
    def test_ode_rhs_accuracy(self):
        # Obtain the Sensitivity Matrix from our method
        fsmp = copy.deepcopy(self.fsmp)
        S, C, _ = get_S_matrix(fsmp)
        # Manually create the Fisher matrix as it should be with exact result of ODE
        #
        # TODO
