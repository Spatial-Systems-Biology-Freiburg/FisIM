import unittest
import numpy as np
import copy
import scipy as sp
import time

from FisInMa.model import FisherModelParametrized
from FisInMa.solving import *

from test.setUp import Setup_Class

class Test_SolvingMethods(Setup_Class):
    def test_get_S_matrix(self):
        # We do not want to do stress testing but need to verify for a certain amount of combinatorics
        # This tries to find a middle ground in testing
        for N_x0, n_t0, n_times, n_inputs_0, n_inputs_1 in [
            [2,3,5,2,1],
            [1,2,3,5,2],
            [2,1,2,3,5],
            [5,2,1,2,3],
            [3,5,2,1,2]
        ]:
            self.setUpClass(N_x0, n_t0, n_times, (n_inputs_0, n_inputs_1))
            fsmp = copy.deepcopy(self.fsmp)
            # TODO test relative sensititivies:
            S, C, solutions = get_S_matrix(fsmp)
        # Run the default function again to restore default for coming simulations
        self.setUpClass()
    
    def test_get_S_matrix_relative_sensitivities(self):
        # We do not want to do stress testing but need to verify for a certain amount of combinatorics
        # This tries to find a middle ground in testing
        for N_x0, n_t0, n_times, n_inputs_0, n_inputs_1 in [
            [2,3,5,2,1],
            [1,2,3,5,2],
            [2,1,2,3,5],
            [5,2,1,2,3],
            [3,5,2,1,2]
        ]:
            self.setUpClass(N_x0, n_t0, n_times, (n_inputs_0, n_inputs_1))
            fsmp = copy.deepcopy(self.fsmp)
            # TODO test relative sensititivies:
            S, C, solutions = get_S_matrix(fsmp, relative_sensitivities=True)
        # Run the default function again to restore default for coming simulations
        self.setUpClass()

    def test_get_S_matrix_identical_times(self):
        # We do not want to do stress testing but need to verify for a certain amount of combinatorics
        # This tries to find a middle ground in testing
        for N_x0, n_t0, n_times, n_inputs_0, n_inputs_1 in [
            [2,3,5,2,1],
            [1,2,3,5,2],
            [2,1,2,3,5],
            [5,2,1,2,3],
            [3,5,2,1,2]
        ]:
            self.setUpClass(N_x0, n_t0, n_times, (n_inputs_0, n_inputs_1), identical_times=True)
            fsmp = copy.deepcopy(self.fsmp)
            # TODO test relative sensititivies:
            S, C, solutions = get_S_matrix(fsmp)
        # Run the default function again to restore default for coming simulations
        self.setUpClass()

    def test_ode_rhs(self):
        fsmp = copy.deepcopy(self.fsmp)

        for x0, t0, i_inputs in itertools.product(
            fsmp.ode_x0,
            fsmp.ode_t0,
            itertools.product(*[range(len(q)) for q in fsmp.inputs])
        ):
            Q = [fsmp.inputs[i][j] for i, j in enumerate(i_inputs)]

            # Helper variables
            n_x = len(x0)
            n_p = len(fsmp.parameters)

            # Test for initial values (initial values for sensitivities are 0 by default)
            x0 = np.concatenate((x0, np.zeros(n_x * n_p)))
            res = ode_rhs(t0, x0, fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, Q, fsmp.parameters, fsmp.ode_args, n_x, n_p)
            
            np.testing.assert_almost_equal(res[:n_x], fsmp.ode_fun(t0, x0, Q, fsmp.parameters, fsmp.ode_args))
            np.testing.assert_almost_equal(res[n_x:], np.array(fsmp.ode_dfdp(t0, x0, Q, fsmp.parameters, fsmp.ode_args)).flatten())

            # Test for non-zero sensitivity values
            s0 = (np.zeros(n_x * n_p) + 1.0).reshape((n_x, n_p))
            x0 = np.concatenate((x0, s0.flatten()))
            res = ode_rhs(t0, x0, fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, Q, fsmp.parameters, fsmp.ode_args, n_x, n_p)
            
            # Mimic calculation of sensitivities
            f_ty = fsmp.ode_fun(t0, x0, Q, fsmp.parameters, fsmp.ode_args)
            np.testing.assert_almost_equal(res[:n_x], f_ty)
            dfdp_ty = fsmp.ode_dfdp(t0, x0, Q, fsmp.parameters, fsmp.ode_args)
            dfdx_ty = fsmp.ode_dfdx(t0, x0, Q, fsmp.parameters, fsmp.ode_args)
            sensitivities = dfdp_ty + np.matmul(dfdx_ty, s0)
            np.testing.assert_almost_equal(res[:n_x], np.array(f_ty).flatten())
            np.testing.assert_almost_equal(res[n_x:], sensitivities.flatten())

class TestCriterions(Setup_Class):
    pass


class Test_CriterionCalculationAutomation(Setup_Class):
    pass
