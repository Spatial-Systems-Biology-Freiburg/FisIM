#!/usr/bin/env python3

import unittest
import numpy as np
import copy

from FisInMa import FischerModel, FischerModelParametrized


def f(x, t, inputs, params, consts):
    A, B = x
    T, Q = inputs
    p, q = params
    c, d, e = consts
    return np.array([
        p*A**2 + p*B + c*(T**2/(e**2+T**2)),
        e*q*A + B + Q + d
    ])


def dfdx(x, t, inputs, params, consts):
    A, B = x
    T, Q = inputs
    p, q = params
    c, d, e = consts
    return np.array([
        p*A**2 + p*B + c*(T**2/(e**2+T**2)),
        e*q*A + B + Q + d
    ])


def dfdp(x, t, inputs, params, consts):
    A, B = x
    T, Q = inputs
    p, q = params
    c, d, e = consts
    return np.array([
        p*A**2 + p*B + c*(T**2/(e**2+T**2)),
        e*q*A + B + Q + d
    ])


class Setup_Class(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.y0=np.array([0, 0])
        self.t0=0.0
        self.times=np.linspace(0.0, 10.0)
        self.inputs=[
            np.array([2,3]),
            np.array([5,6,7])
        ]
        self.parameters=(2.95, 8.4768)
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
            constants=self.constants
        )
        self.fsmp = FischerModelParametrized.init_from(self.fsm)


class Test_FischerModelParametrized_Init(Setup_Class):
    # Individual sampling tests
    def test_sample_ode_y0(self):
        fsm = copy.deepcopy(self.fsm)
        y0 = (4.22, 9.44, 5)
        fsm.ode_y0 = y0
        fsmp = FischerModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_y0, np.linspace(*y0))

    def test_sample_ode_t0(self):
        fsm = copy.deepcopy(self.fsm)
        t0 = (1.1, 7.2, 4)
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FischerModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, np.linspace(*t0))

    def test_sample_times_identical(self):
        fsm = copy.deepcopy(self.fsm)
        t = (3.21, 11.44, 2)
        fsm.times = t
        fsm.identical_times = True
        fsmp = FischerModelParametrized.init_from(fsm)
        times = fsmp.times
        np.testing.assert_almost_equal(times, np.linspace(*t))
    
    def test_sample_times_not_identical(self):
        fsm = copy.deepcopy(self.fsm)
        t = (3.35, 78.2, 6)
        fsm.times = t
        fsmp = FischerModelParametrized.init_from(fsm)
        times = fsmp.times
        np.testing.assert_almost_equal(times, np.full(tuple(len(q) for q in fsmp.inputs) + (t[2],), np.linspace(*t)))
    
    def test_sample_inputs(self):
        fsm = copy.deepcopy(self.fsm)
        inp0 = (-2.0, 51.2, 3)
        inp1 = np.array([1,2,3,4.5,5.2,3.4])
        fsm.inputs = [inp0, inp1]
        fsmp = FischerModelParametrized.init_from(fsm)
        inputs = fsmp.inputs
        for i, j in zip(inputs, [np.linspace(*inp0), inp1]):
            np.testing.assert_almost_equal(i, j)
    
    # Test combinations (2)
    def test_sample_ode_y0_ode_t0(self):
        fsm = copy.deepcopy(self.fsm)
        y0 = (3.102, 3.569, 5)
        t0 = (0.0, 1.0, 7)
        fsm.ode_y0 = y0
        fsm.ode_t0 = t0
        fsmp = FischerModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_y0, np.linspace(*y0))
        np.testing.assert_almost_equal(fsmp.ode_t0, np.linspace(*t0))
    
    def test_sample_ode_y0_times(self):
        fsm = copy.deepcopy(self.fsm)
        y0 = (3.102, 3.569, 5)
        t = (3.21, 11.44, 2)
        fsm.ode_y0 = y0
        fsm.times = t
        fsmp = FischerModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_y0, np.linspace(*y0))
        np.testing.assert_almost_equal(fsmp.times, np.full(tuple(len(q) for q in fsmp.inputs) + (t[2],), np.linspace(*t)))

    def test_sample_ode_t0_times(self):
        fsm = copy.deepcopy(self.fsm)
        t0 = (-2.13, 5.05, 6)
        t = (3.21, 11.44, 2)
        fsm.ode_t0 = t0
        fsm.times = t
        fsmp = FischerModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, np.linspace(*t0))
        np.testing.assert_almost_equal(fsmp.times, np.full(tuple(len(q) for q in fsmp.inputs) + (t[2],), np.linspace(*t)))

    # TODO
    # def test_sample_ode_y0_inputs(self):
    #     pass
    # 
    # def test_sample_ode_t0_inputs(self):
    #     pass
    # 
    # def test_sample_times_inputs(self):
    #     pass
    # 
    # # Test combinations (3)
    # def test_sample_ode_y0_times_inputs(self):
    #     pass
    # 
    # def test_sample_ode_t0_times_inputs(self):
    #     pass
    # 
    # def test_sample_times_inputs(self):
    #     pass
    # 
    # def test_sample_ode_y0_ode_t0_inputs(self):
    #     pass
    # 
    # # Test combinations (4)
    # def test_sample_ode_y0_ode_t0_times_inputs(self):
    #     pass


class Test_FischerModelParametrized_Set_Get(Setup_Class):
    def test_set_get_t0(self):
        fsm = copy.deepcopy(self.fsm)
        t0 = (2.22, 56.3, 8)
        fsm.ode_t0 = t0
        fsmp = FischerModelParametrized.init_from(fsm)
        t0 = 1.0
        fsmp.t0 = t0
        self.assertEqual(fsmp.t0, t0)

    def test_set_get_y0(self):
        fsm = copy.deepcopy(self.fsm)
        y0 = (4.22, 9.44, 5)
        fsm.ode_y0 = y0
        fsmp = FischerModelParametrized.init_from(fsm)
        y0 = np.array([33.2, 12.3])
        fsmp.y0 = y0
        np.testing.assert_almost_equal(y0, fsmp.y0)
    
    def test_set_get_times_identical(self):
        fsm = copy.deepcopy(self.fsm)
        times = (4.22, 9.44, 8)
        fsm.times = times
        fsm.identical_times = True
        fsmp = FischerModelParametrized.init_from(fsm)
        times = np.array([2.0, 3.0, 66.0])
        fsmp.times = times
        np.testing.assert_almost_equal(times, fsmp.times)
    
    def test_set_get_times_not_identical(self):
        fsm = copy.deepcopy(self.fsm)
        t = (4.22, 9.44, 8)
        fsm.times = t
        fsmp = FischerModelParametrized.init_from(fsm)
        times = np.full(fsmp.times.shape, np.linspace(3.119, 6.489, t[2]))
        fsmp.times = times
        np.testing.assert_almost_equal(times, fsmp.times)
    
    def test_set_get_inputs(self):
        fsm = copy.deepcopy(self.fsm)
        inp0 = (-2.0, 51.2, 3)
        inp1 = np.array([1,2,3,4.5,5.2,3.4])
        fsm.inputs = [inp0, inp1]
        fsmp = FischerModelParametrized.init_from(fsm)
        inputs = [
            np.linspace(12.0, 15.0),
            None
        ]
        fsmp.inputs = inputs
        for i, j in zip(inputs, fsmp.inputs):
            if i is not None:
                np.testing.assert_almost_equal(i, j)
    
    def test_get_parameters(self):
        fsmp = self.fsmp
        np.testing.assert_almost_equal(fsmp.parameters, self.parameters)
    
    def test_get_constants(self):
        fsmp = self.fsmp
        np.testing.assert_almost_equal(fsmp.constants, self.constants)

    @unittest.expectedFailure
    def test_set_immutable_y0(self):
        fsm = copy.deepcopy(self.fsm)
        fsmp.y0 = np.linspace(0, 10)
    
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
