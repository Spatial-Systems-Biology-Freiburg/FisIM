import unittest
import numpy as np
import copy
import scipy as sp

from FisInMa.model import FischerModelParametrized
from FisInMa.solving import *

from model import Setup_Class

class Test_SolvingMethods(Setup_Class):
    def test_get_S_matrix(self):
        fsmp = copy.deepcopy(self.fsmp)
        # TODO test relative sensititivies:
        S, C, solutions = get_S_matrix(fsmp)# relative_sensitivities=True)
        print(S.shape)
        print(C.shape)
        F = S.dot(S.T)
        print(F)
        print(np.linalg.det(F))
    
    def test_ode_rhs(self):
        fsmp = copy.deepcopy(self.fsmp)

        for y0, t0, i_inputs in itertools.product(
            fsmp.ode_y0,
            fsmp.ode_t0,
            itertools.product(*[range(len(q)) for q in fsmp.inputs])
        ):
            Q = [fsmp.inputs[i][j] for i, j in enumerate(i_inputs)]

            # Helper variables
            n_x = len(y0)
            n_p = len(fsmp.parameters)

            # Test for initial values (initial values for sensitivities are 0 by default)
            x0 = np.concatenate((y0, np.zeros(n_x * n_p)))
            res = ode_rhs(t0, x0, fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, Q, fsmp.parameters, fsmp.constants, n_x, n_p)
            
            np.testing.assert_almost_equal(res[:n_x], fsmp.ode_fun(t0, y0, Q, fsmp.parameters, fsmp.constants))
            np.testing.assert_almost_equal(res[n_x:], np.array(fsmp.ode_dfdp(t0, y0, Q, fsmp.parameters, fsmp.constants)).flatten())

            # Test for non-zero sensitivity values
            s0 = (np.zeros(n_x * n_p) + 1.0).reshape((n_x, n_p))
            x0 = np.concatenate((y0, s0.flatten()))
            res = ode_rhs(t0, x0, fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, Q, fsmp.parameters, fsmp.constants, n_x, n_p)
            
            # Mimic calculation of sensitivities
            f_ty = fsmp.ode_fun(t0, y0, Q, fsmp.parameters, fsmp.constants)
            np.testing.assert_almost_equal(res[:n_x], f_ty)
            dfdp_ty = fsmp.ode_dfdp(t0, y0, Q, fsmp.parameters, fsmp.constants)
            dfdx_ty = fsmp.ode_dfdx(t0, y0, Q, fsmp.parameters, fsmp.constants)
            sensitivities = dfdp_ty + np.matmul(dfdx_ty, s0)
            np.testing.assert_almost_equal(res[:n_x], np.array(f_ty).flatten())
            np.testing.assert_almost_equal(res[n_x:], sensitivities.flatten())

class TestCriterions(Setup_Class):
    pass


class Test_CriterionCalculationAutomation(Setup_Class):
    pass
