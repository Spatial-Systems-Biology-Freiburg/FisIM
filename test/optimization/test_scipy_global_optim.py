import numpy as np
import copy

from FisInMa.optimization.scipy_global_optim import _scipy_calculate_bounds_constraints, _create_comparison_matrix, find_optimal
from FisInMa import FisherModelParametrized

from test.setUp import Setup_Class


class Test_ScipyGlobalOptimAlgorithms(Setup_Class):
    def test_scipy_differential_evolution(self):
        fsm = copy.deepcopy(self.fsm)
        fsm.ode_t0 = 0.0
        fsm.ode_y0 = [np.array([0.05, 0.001])]
        fsm.inputs=[
            np.arange(2, 2+2),
            np.arange(5, 2+5)
        ]
        fsm.times = (0.0, 10.0, 2)
        find_optimal(fsm, "scipy_differential_evolution", workers=1, maxiter=10, popsize=10)


class Test_ScipyCalculateConstraints(Setup_Class):
    def test_create_comparison_matrix(self):
        # Explicit testing
        test_matrices = [
            np.array([
                [1.0, -1.0]
            ]),
            np.array([
                [1.0, -1.0,  0.0],
                [0.0,  1.0, -1.0]
            ]),
            np.array([
                [1.0, -1.0,  0.0,  0.0],
                [0.0,  1.0, -1.0,  0.0],
                [0.0,  0.0,  1.0, -1.0]
            ]),
            np.array([
                [1.0, -1.0,  0.0,  0.0,  0.0],
                [0.0,  1.0, -1.0,  0.0,  0.0],
                [0.0,  0.0,  1.0, -1.0,  0.0],
                [0.0,  0.0,  0.0,  1.0, -1.0]
            ]),
            np.array([
                [1.0, -1.0,  0.0,  0.0,  0.0,  0.0],
                [0.0,  1.0, -1.0,  0.0,  0.0,  0.0],
                [0.0,  0.0,  1.0, -1.0,  0.0,  0.0],
                [0.0,  0.0,  0.0,  1.0, -1.0,  0.0],
                [0.0,  0.0,  0.0,  0.0,  1.0, -1.0]
            ])
        ]
        for t in test_matrices:
            A = _create_comparison_matrix(t.shape[1])
            np.testing.assert_almost_equal(t, A)
        # Implicit testing
        for k in range(1, 100):
            A = _create_comparison_matrix(k)
            for i in range(k-1):
                # Test if correct entries are non-zero
                np.testing.assert_almost_equal(A[i,i], 1.0)
                np.testing.assert_almost_equal(A[i,i+1], -1.0)
                # Test how many non-zero entries the matrix has. If the count matches, the matrix is correct
                np.testing.assert_equal(np.sum(A!=0.0), 2*(k-1))


    def test_scipy_calculate_bounds_constraints_sample_ode_t0(self):
        fsm = copy.deepcopy(self.fsm)
        fsm.ode_t0 = (0.00, 0.001, 3)
        fsmp = FisherModelParametrized.init_from(fsm)
        bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
        np.testing.assert_almost_equal(bounds, [fsm.ode_t0[0:2]]*fsm.ode_t0[2])
        np.testing.assert_almost_equal(constraints.lb, [-np.inf]*(fsm.ode_t0[2]-1))
        np.testing.assert_almost_equal(constraints.ub, [np.inf]*(fsm.ode_t0[2]-1))
        np.testing.assert_almost_equal(constraints.A, _create_comparison_matrix(fsm.ode_t0[2]))
    
    def test_scipy_calculate_bounds_constraints_sample_ode_y0(self):
        fsm = copy.deepcopy(self.fsm)
        fsm.ode_y0 = [[0.0,0.0],[0.1,0.05]]
        fsmp = FisherModelParametrized.init_from(fsm)
        bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
        np.testing.assert_almost_equal(bounds, [])
        np.testing.assert_almost_equal(constraints.lb, [])
        np.testing.assert_almost_equal(constraints.ub, [])
        np.testing.assert_almost_equal(constraints.A, _create_comparison_matrix(0))
    
    def test_scipy_calculate_bounds_constraints_sample_times(self):
        fsm = copy.deepcopy(self.fsm)
        fsm.identical_times=True
        fsm.times = (0.0, 10.0, 5)
        fsmp = FisherModelParametrized.init_from(fsm)
        bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
        n_inputs = np.product([len(q) for q in fsmp.inputs])
        n_times = fsmp.times.shape[-1] if fsm.identical_times==True else n_inputs * fsmp.times.shape[-1]
        # Test bounds and constraints
        np.testing.assert_almost_equal(bounds, [fsm.times[0:2]] * n_times)
        np.testing.assert_almost_equal(constraints.ub, [0.0]*(fsm.times[2]-1))
        np.testing.assert_almost_equal(constraints.lb, [-np.inf]*(fsm.times[2]-1))
        # Create matrix to compare against
        A = _create_comparison_matrix(fsm.times[2])
        np.testing.assert_almost_equal(constraints.A, A)
    
    def test_scipy_calculate_bounds_constraints_sample_inputs(self):
        fsm = copy.deepcopy(self.fsm)
        fsm.inputs = [
            (1.0, 2.0, 3),
            (3.0, 44.0, 6)
        ]
        fsmp = FisherModelParametrized.init_from(fsm)
        bounds, constraints = _scipy_calculate_bounds_constraints(fsmp)
        # Test bounds and constraints
        np.testing.assert_almost_equal(bounds, [fsm.inputs[0][0:2]]*fsm.inputs[0][2] + [fsm.inputs[1][0:2]]*fsm.inputs[1][2])
        np.testing.assert_almost_equal(constraints.lb, [-np.inf]*(fsm.inputs[0][2]-1+fsm.inputs[1][2]-1))
        np.testing.assert_almost_equal(constraints.ub, [0.0]*(fsm.inputs[0][2]-1+fsm.inputs[1][2]-1))
        # Create matrix to compare against
        B = np.eye(0)
        for i in range(len(fsm.inputs)):
            A = _create_comparison_matrix(fsm.inputs[i][2])
            B = np.block([[B,np.zeros((B.shape[0],A.shape[1]))],[np.zeros((A.shape[0],B.shape[1])),A]])
        np.testing.assert_almost_equal(constraints.A, B)

    # Combinations (2)
    """
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