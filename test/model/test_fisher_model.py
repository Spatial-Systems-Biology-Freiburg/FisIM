#!/usr/bin/env python3

import unittest
import numpy as np
import copy

from test.setUp import Setup_Class

from FisInMa import FisherModelParametrized


class Test_FisherModelParametrized_Init(Setup_Class):
    # Individual sampling tests
    def test_sample_ode_x0_explicit(self):
        fsm = copy.deepcopy(self.fsm)
        x0 = [[0.02, 0.0005], [0.015, 0.001]]
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        for p, q in zip(x0, fsmp.ode_x0):
            np.testing.assert_almost_equal(p, q)

    def test_fixed_ode_x0_explicit_single_vector(self):
        fsm = copy.deepcopy(self.fsm)
        x0 = np.array([0.02, 0.0005])
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal([x0], fsmp.ode_x0)
    
    def test_fixed_ode_x0_explicit_single_vector_list(self):
        fsm = copy.deepcopy(self.fsm)
        x0 = [0.02, 0.0005]
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal([x0], fsmp.ode_x0)

    @unittest.expectedFailure
    def test_fixed_ode_x0_too_large_array(self):
        fsm = copy.deepcopy(self.fsm)
        x0 = np.array([[0.02, 0.05],[0.015, 0.04767]])
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)

    def test_fixed_ode_x0_explicit_multiple_vector(self):
        fsm = copy.deepcopy(self.fsm)
        x0 = [[0.02, 0.0005],[0.015, 0.001]]
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(x0, fsmp.ode_x0)
    
    def test_fixed_ode_x0_explicit_single_float(self):
        fsm = copy.deepcopy(self.fsm)
        x0 = 0.2
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal([[x0]], fsmp.ode_x0)

    def test_sample_ode_t0(self):
        fsm = copy.deepcopy(self.fsm)
        t0 = (1.1, 7.2, 4)
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, np.linspace(*t0))

    def test_fixed_ode_t0_float(self):
        fsm = copy.deepcopy(self.fsm)
        t0 = 0.0
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, [t0])
    
    def test_fixed_ode_t0_list(self):
        fsm = copy.deepcopy(self.fsm)
        t0 = [0.0, 0.1]
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, t0)

    def test_fixed_ode_t0_np_array(self):
        fsm = copy.deepcopy(self.fsm)
        t0 = np.array([0.0, 0.1])
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, t0)

    def test_sample_times_identical(self):
        fsm = copy.deepcopy(self.fsm)
        t = (3.21, 11.44, 2)
        fsm.times = t
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        times = fsmp.times
        np.testing.assert_almost_equal(times, np.linspace(*t))
    
    def test_sample_times_not_identical(self):
        fsm = copy.deepcopy(self.fsm)
        t = (3.35, 78.2, 6)
        fsm.times = t
        fsmp = FisherModelParametrized.init_from(fsm)
        times = fsmp.times
        np.testing.assert_almost_equal(times, np.full(tuple(len(q) for q in fsmp.inputs) + (t[2],), np.linspace(*t)))
    
    def test_sample_inputs(self):
        fsm = copy.deepcopy(self.fsm)
        inp0 = (-2.0, 51.2, 3)
        inp1 = np.array([1,2,3,4.5,5.2,3.4])
        fsm.inputs = [inp0, inp1]
        fsmp = FisherModelParametrized.init_from(fsm)
        inputs = fsmp.inputs
        for i, j in zip(inputs, [np.linspace(*inp0), inp1]):
            np.testing.assert_almost_equal(i, j)
    
    # Test combinations (2)
    def test_sample_ode_x0_ode_t0(self):
        fsm = copy.deepcopy(self.fsm)
        x0 = [[0.0187, 0.000498], [0.0291, 0.002]]
        t0 = (0.0, 1.0, 7)
        fsm.ode_x0 = x0
        fsm.ode_t0 = t0
        fsmp = FisherModelParametrized.init_from(fsm)
        for p, q in zip(fsmp.ode_x0, x0):
            np.testing.assert_almost_equal(p, q)
        np.testing.assert_almost_equal(fsmp.ode_t0, np.linspace(*t0))
    
    def test_sample_ode_x0_times(self):
        fsm = copy.deepcopy(self.fsm)
        x0 = [[0.0187, 0.000498], [0.0291, 0.002]]
        t = (3.21, 11.44, 2)
        fsm.ode_x0 = x0
        fsm.times = t
        fsmp = FisherModelParametrized.init_from(fsm)
        for p, q in zip(fsmp.ode_x0, x0):
            np.testing.assert_almost_equal(p, q)
        np.testing.assert_almost_equal(fsmp.times, np.full(tuple(len(q) for q in fsmp.inputs) + (t[2],), np.linspace(*t)))

    def test_sample_ode_t0_times(self):
        fsm = copy.deepcopy(self.fsm)
        t0 = (-2.13, 5.05, 6)
        t = (3.21, 11.44, 2)
        fsm.ode_t0 = t0
        fsm.times = t
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, np.linspace(*t0))
        np.testing.assert_almost_equal(fsmp.times, np.full(tuple(len(q) for q in fsmp.inputs) + (t[2],), np.linspace(*t)))

    # TODO
    # def test_sample_ode_x0_inputs(self):
    #     pass
    # 
    # def test_sample_ode_t0_inputs(self):
    #     pass
    # 
    # def test_sample_times_inputs(self):
    #     pass
    # 
    # # Test combinations (3)
    # def test_sample_ode_x0_times_inputs(self):
    #     pass
    # 
    # def test_sample_ode_t0_times_inputs(self):
    #     pass
    # 
    # def test_sample_times_inputs(self):
    #     pass
    # 
    # def test_sample_ode_x0_ode_t0_inputs(self):
    #     pass
    # 
    # # Test combinations (4)
    # def test_sample_ode_x0_ode_t0_times_inputs(self):
    #     pass


class Test_FisherModelParametrized_Set_Get(Setup_Class):
    def test_set_get_t0(self):
        fsm = copy.deepcopy(self.fsm)
        t0 = (2.22, 56.3, 8)
        fsm.ode_t0 = t0
        fsmp = FisherModelParametrized.init_from(fsm)
        t0 = 1.0
        fsmp.t0 = t0
        self.assertEqual(fsmp.t0, t0)

    def test_set_get_x0(self):
        fsm = copy.deepcopy(self.fsm)
        x0 = np.array([4.22, 9.44])
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        x0 = np.array([33.2, 12.3])
        fsmp.x0 = x0
        np.testing.assert_almost_equal(x0, fsmp.x0)
    
    def test_set_get_times_identical(self):
        fsm = copy.deepcopy(self.fsm)
        times = (4.22, 9.44, 8)
        fsm.times = times
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        times = np.array([2.0, 3.0, 66.0])
        fsmp.times = times
        np.testing.assert_almost_equal(times, fsmp.times)
    
    def test_set_get_times_not_identical(self):
        fsm = copy.deepcopy(self.fsm)
        t = (4.22, 9.44, 8)
        fsm.times = t
        fsmp = FisherModelParametrized.init_from(fsm)
        times = np.full(fsmp.times.shape, np.linspace(3.119, 6.489, t[2]))
        fsmp.times = times
        np.testing.assert_almost_equal(times, fsmp.times)
    
    def test_set_get_inputs(self):
        fsm = copy.deepcopy(self.fsm)
        inp0 = (-2.0, 51.2, 3)
        inp1 = np.array([1,2,3,4.5,5.2,3.4])
        fsm.inputs = [inp0, inp1]
        fsmp = FisherModelParametrized.init_from(fsm)
        inputs = [
            np.linspace(12.0, 15.0),
            None
        ]
        fsmp.inputs = inputs
        for i, j in zip(inputs, fsmp.inputs):
            if i is not None:
                np.testing.assert_almost_equal(i, j)
    
    def test_get_parameters(self):
        fsmp = copy.deepcopy(self.fsmp)
        np.testing.assert_almost_equal(fsmp.parameters, self.parameters)
    
    def test_get_ode_args(self):
        fsmp = copy.deepcopy(self.fsmp)
        np.testing.assert_almost_equal(fsmp.ode_args, self.ode_args)

    @unittest.expectedFailure
    def test_set_immutable_x0(self):
        fsm = copy.deepcopy(self.fsm)
        fsmp.x0 = np.linspace(0, 10)
    
    @unittest.expectedFailure
    def test_set_immutable_t0(self):
        fsm = copy.deepcopy(self.fsm)
        fsmp.t0 = np.linspace(0, 10)
    
    @unittest.expectedFailure
    def test_set_immutable_times(self):
        fsm = copy.deepcopy(self.fsm)
        fsmp.times = np.linspace(0, 10)

    @unittest.expectedFailure
    def test_set_immutable_inputs(self):
        fsm = copy.deepcopy(self.fsm)
        fsmp.inputs = [
            np.linspace(0, 10),
            np.linspace(3.0, 45.0)
        ]
