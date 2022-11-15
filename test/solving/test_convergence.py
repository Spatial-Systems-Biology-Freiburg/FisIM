import numpy as np
import scipy as sp
import time
import pytest

from FisInMa.model import FisherModel, FisherModelParametrized
from FisInMa.solving import *


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
        [1, - T*A],
        [- T*B, 1]
    ]

def g(t, x, inputs, parameters, ode_args):
    (A, B) = x
    return A**2

def dgdx(t, x, inputs, parameters, ode_args):
    (A, B) = x
    return [
        [2*A, 0]
    ]

def dgdp(t, x, inputs, parameters, ode_args):
    return [
        [0 ,0]
    ]

def x_exact(t, x0, inputs, parameters, ode_args):
    (T,) = inputs
    (a, b) = parameters
    int_constant = [
        x0[0] - a/(b*T),
        x0[1] - b/(a*T)
    ]
    return [
        a/(b*T) + int_constant[0] * np.exp(-b*T*t),
        b/(a*T) + int_constant[1] * np.exp(-a*T*t)
    ]

def dxdp_exact(t, x0, inputs, parameters, ode_args):
    (T,) = inputs
    (a, b) = parameters
    int_constant = [
        x0[0] - a/(b*T),
        x0[1] - b/(a*T)
    ]
    return [
        [(1 - np.exp(-b*T*t))/(b*T), a/(b**2*T)*(-1+np.exp(-b*T*t)+b*T*t*np.exp(-b*T*t))-x0[0]*t*T*np.exp(-b*T*t)],
        [b/(a**2*T)*(-1+np.exp(-a*T*t)+a*T*t*np.exp(-a*T*t))-x0[1]*t*T*np.exp(-a*T*t), (1 - np.exp(-a*T*t))/(a*T)]
    ]


class Setup_Convergence:
    def __init__(self, n_times=4, n_inputs=3, identical_times=False):
        self.x0=[1.0, 0.5]
        self.t0=0.0
        self.times=np.linspace(0.0, 10.0, n_times)
        self.n_times = n_times
        self.n_inputs = n_inputs
        self.inputs=[
            np.linspace(0.8, 1.2, n_inputs)
        ]
        
        self.parameters=(0.2388, 0.74234)
        # n_ode_args = 3
        self.ode_args=None
        self.fsm = FisherModel(
            ode_fun=ode_fun,
            ode_dfdx=ode_dfdx,
            ode_dfdp=ode_dfdp,
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
def convergence_model(identical_times):
    return Setup_Convergence(identical_times=identical_times)


class TestConvergence:
    @pytest.mark.parametrize("identical_times", [True, False])
    def test_ode_rhs_exact_solution(self, convergence_model):
        # Obtain the Sensitivity Matrix from our method
        fsmp = convergence_model.fsmp
        S, C, solutions = get_S_matrix(fsmp)
        # Manually create the Fisher matrix as it should be with exact result of ODE

        # Calculate observables of exact solution for all entries
        n_x0 = len(fsmp.ode_x0[0])
        n_p = len(fsmp.parameters)
        n_inputs = convergence_model.n_inputs

        # Determine the number of components of the observable
        n_obs = np.array(g(fsmp.ode_t0[0], fsmp.ode_x0[0], [q[0] for q in fsmp.inputs], fsmp.parameters, fsmp.ode_args)).size

        # The shape of the initial S matrix is given by
        S_own = np.zeros((n_p, n_obs, n_inputs, convergence_model.n_times))
        
        # Test that the ODE is solved correctly
        for sol in solutions:# , sol_ode_own, sol_sens_own in zip(solutions, solutions_ode_exact_own, solutions_sens_exact_own):
            sol_ode_calc = sol.ode_solution.y[:n_x0].T
            sol_dxdp_calc = sol.ode_solution.y[n_x0:].T

            sol_ode_own = np.array([x_exact(t, sol.ode_x0, sol.inputs, sol.parameters, sol.ode_args) for t in sol.times])
            sol_dxdp_own = np.array([dxdp_exact(t, sol.ode_x0, sol.inputs, sol.parameters, sol.ode_args) for t in sol.times])
            sol_sens_own = np.array([
                # Calculate the sensitivities via the chain rule from exact results
                # (total derivative)   (partial derivative)
                #       dgdp         =  dgdp + dgdx * dxdp
                # Calculate first term
                dgdp(t, x_exact(t, sol.ode_x0, sol.inputs, sol.parameters, sol.ode_args), sol.inputs, sol.parameters, sol.ode_args) +
                # Calculte second term
                np.array(dgdx(t, x_exact(t, sol.ode_x0, sol.inputs, sol.parameters, sol.ode_args), sol.inputs, sol.parameters, sol.ode_args)) @ np.array(dxdp_exact(t, sol.ode_x0, sol.inputs, sol.parameters, sol.ode_args))
            for t in sol.times])

            s = sol_sens_own.reshape((-1, n_obs, n_p)).swapaxes(0, 2)
            comp = np.isclose(fsmp.inputs[0], sol.inputs[0])
            i = next(i for i, el in enumerate(comp) if el==True)
            S_own[(slice(None), slice(None), i, slice(None))] = s

            np.testing.assert_almost_equal(sol_ode_calc, sol_ode_own, decimal=3)
            np.testing.assert_almost_equal(sol_dxdp_calc, sol_dxdp_own.reshape((len(sol.times), -1)), decimal=3)
        # Test that the resulting sensitivities are the same
        S_own = S_own.reshape((n_p, -1))
        F_own = np.matmul(S_own, S_own.T)
        F = np.matmul(S,S.T)
        np.testing.assert_almost_equal(S_own, S, decimal=3)
        np.testing.assert_almost_equal(F_own, F, decimal=2)
