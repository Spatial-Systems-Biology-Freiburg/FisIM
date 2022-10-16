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

        for index in itertools.product(*[range(len(q)) for q in fsmp.inputs]):
            Q = [fsmp.inputs[i][j] for i, j in enumerate(index)]
            if fsmp.identical_times==True:
                t = fsmp.times
            else:
                t = fsmp.times[index]
            # y = ode_rhs(fsmp.ode_t0, fsmp.ode_y0, fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, Q, fsmp.parameters, fsmp.constants)

            n_x = len(fsmp.ode_y0)
            n_p = len(fsmp.parameters)
            y0 = np.concatenate((fsmp.ode_y0, np.zeros(n_x * n_p)))
            res = solve_ivp(fun=ode_rhs, t_span=(fsmp.ode_t0, np.max(t)), y0=y0, t_eval=t, args=(fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, Q, fsmp.parameters, fsmp.constants, n_x, n_p), method="Radau")#, jac=fsmp.ode_dfdx)
            
            res.y[n_x:]
            # y_fun, y_dfdx, y_dfdp, rest = lists = np.split(y0, [n_x, n_x+n_x**2, n_x+n_x**2+n_x*m_p])
            # print("Lists:")
            # print(y_fun)
            # print(y_dfdx)
            # print(y_dfdp)


class TestCriterions(Setup_Class):
    pass


class Test_CriterionCalculationAutomation(Setup_Class):
    pass
