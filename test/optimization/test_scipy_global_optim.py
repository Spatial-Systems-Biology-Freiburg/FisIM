import numpy as np
import copy

from FisInMa.optimization.scipy_global_optim import _scipy_calculate_bounds_constraints
from FisInMa import FisherModelParametrized

from test.setUp import Setup_Class


class Test_ScipyGlobalOptimAlgorithms(Setup_Class):
    def test_scipy_differential_evolution(self):
        fsm = copy.deepcopy(self.fsm)
        # find_optimal(fsm, "scipy_differential_evolution")


class Test_ScipyCalculateConstraints(Setup_Class):
    def test_scipy_calculate_bounds_constraints_sample_ode_t0(self):
        fsm = copy.deepcopy(self.fsm)
        fsm.ode_t0 = (0.00, 0.001, 3)
        fsmp = FisherModelParametrized.init_from(fsm)
        bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
        # print(bounds)
        np.testing.assert_almost_equal(bounds, [fsm.ode_t0[0:2]]*fsm.ode_t0[2])
        # np.testing.assert_almost_equal(constraints, np.eye(fsm.ode_t0[2]))
        # TODO test constraints
    
    def test_scipy_calculate_bounds_constraints_sample_ode_y0(self):
        fsm = copy.deepcopy(self.fsm)
        fsm.ode_y0 = [[0.0,0.0],[0.1,0.05]]
        fsmp = FisherModelParametrized.init_from(fsm)
        bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
        np.testing.assert_almost_equal(bounds, [])
        # TODO test constraints
    
    def test_scipy_calculate_bounds_constraints_sample_times(self):
        fsm = copy.deepcopy(self.fsm)
        fsm.identical_times=True
        fsm.times = (0.0, 10.0, 5)
        fsmp = FisherModelParametrized.init_from(fsm)
        bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
        n_inputs = np.product([len(q) for q in fsmp.inputs])
        n_times = fsmp.times.shape[-1] if fsm.identical_times==True else n_inputs * fsmp.times.shape[-1]
        np.testing.assert_almost_equal(bounds, [fsm.times[0:2]] * n_times)
        # TODO test constraints
    """
    def test_scipy_calculate_bounds_constraints_sample_inputs(self):
        fsm = copy.deepcopy(self.fsm)
        fsm.inputs = [
            (1.0, 2.0, 3),
            (3.0, 44.0, 6)
        ]
        fsmp = FisherModelParametrized.init_from(fsm)
        bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
        np.testing.assert_almost_equal(bounds, [fsm.inputs[0][0:2]]*fsm.inputs[0][2] + [fsm.inputs[1][0:2]]*fsm.inputs[1][2])
        # TODO test constraints

    # Combinations (2)
    def test_scipy_calculate_bounds_constraints_sample_ode_t0_ode_y0(self):
        pass

    def test_scipy_calculate_bounds_constraints_sample_ode_t0_times(self):
        pass

    def test_scipy_calculate_bounds_constraints_sample_ode_t0_inputs(self):
        pass

    def test_scipy_calculate_bounds_constraints_sample_ode_y0_times(self):
        pass

    def test_scipy_calculate_bounds_constraints_sample_ode_y0_inputs(self):
        pass

    def test_scipy_calculate_bounds_constraints_sample_times_inputs(self):
        pass

    # Combinations (3)
    def test_scipy_calculate_bounds_constraints_sample_ode_t0_ode_y0_times(self):
        pass

    def test_scipy_calculate_bounds_constraints_sample_ode_t0_ode_y0_inputs(self):
        pass

    def test_scipy_calculate_bounds_constraints_sample_ode_t0_times_inputs(self):
        pass

    def test_scipy_calculate_bounds_constraints_sample_ode_y0_times_inputs(self):
        pass

    # Combination (4)
    def test_scipy_calculate_bounds_constraints_sample_ode_t0_ode_y0_times_inputs(self):
        pass
    """