#!/usr/bin/env python3

import pytest
import numpy as np
from pydantic import ValidationError

from test.setUp import default_model, model_init_params, pool_model

from FisInMa import FisherModelParametrized


class Test_fmsp_init_from_fsm:
    # Individual sampling tests
    def test_sample_ode_x0_explicit(self, default_model):
        fsm = default_model.fsm
        x0 = [[0.02, 0.0005], [0.015, 0.001]]
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        for p, q in zip(x0, fsmp.ode_x0):
            np.testing.assert_almost_equal(p, q)

    def test_sample_ode_x0_and_param(self, model_init_params):
        fsm = model_init_params.fsm
        x0 = [[0.02, 0.0005], [0.015, 0.001]]
        fsm.ode_x0 = x0
        with pytest.raises(ValueError):
            fsmp = FisherModelParametrized.init_from(fsm)

    def test_fixed_ode_x0_explicit_single_vector(self, model_init_params):
        fsm = model_init_params.fsm
        x0 = np.array([0.02, 0.0005])
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal([x0], fsmp.ode_x0)
    
    def test_fixed_ode_x0_explicit_single_vector_list(self, default_model):
        fsm = default_model.fsm
        x0 = [0.02, 0.0005]
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal([x0], fsmp.ode_x0)

    def test_fixed_ode_x0_too_large_array(self, default_model):
        fsm = default_model.fsm
        x0 = np.array([[0.02, 0.05],[0.015, 0.04767]])
        with pytest.raises(ValidationError):
            fsm.ode_x0 = x0
            fsmp = FisherModelParametrized.init_from(fsm)

    def test_fixed_ode_x0_explicit_multiple_vector(self, default_model):
        fsm = default_model.fsm
        x0 = [[0.02, 0.0005],[0.015, 0.001]]
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(x0, fsmp.ode_x0)
    
    def test_fixed_ode_x0_explicit_single_float(self, pool_model):
        fsm = pool_model.fsm
        x0 = 0.2
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal([[x0]], fsmp.ode_x0)

    def test_sample_ode_t0(self, default_model):
        fsm = default_model.fsm
        t0 = {"lb":1.1, "ub":7.2, "n":4}
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, np.linspace(t0["lb"], t0["ub"], t0["n"]))

    def test_fixed_ode_t0_float(self, default_model):
        fsm = default_model.fsm
        t0 = 0.0
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, [t0])
    
    def test_fixed_ode_t0_list(self, default_model):
        fsm = default_model.fsm
        t0 = [0.0, 0.1]
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, t0)

    def test_fixed_ode_t0_np_array(self, default_model):
        fsm = default_model.fsm
        t0 = np.array([0.0, 0.1])
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, t0)

    def test_fixed_ode_t0_np_array_2(self, model_init_params):
        fsm = model_init_params.fsm
        t0 = np.array([0.0, 0.1])
        fsm.ode_t0 = t0
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, t0)

    def test_sample_times_identical(self, default_model):
        fsm = default_model.fsm
        t = {"lb":3.21, "ub":11.44, "n":2}
        fsm.times = t
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        times = fsmp.times
        np.testing.assert_almost_equal(times, np.linspace(t["lb"], t["ub"], t["n"]))
    
    def test_sample_times_not_identical(self, default_model):
        fsm = default_model.fsm
        t = {"lb":3.35, "ub":78.2, "n":6}
        fsm.times = t
        fsmp = FisherModelParametrized.init_from(fsm)
        times = fsmp.times
        np.testing.assert_almost_equal(times, np.full(tuple(len(q) for q in fsmp.inputs) + (t["n"],), np.linspace(t["lb"], t["ub"], t["n"])))
    
    def test_sample_inputs(self, default_model):
        fsm = default_model.fsm
        inp0 = {"lb":-2.0, "ub":51.2, "n":3}
        inp1 = np.array([1,2,3,4.5,5.2,3.4])
        fsm.inputs = [inp0, inp1]
        fsmp = FisherModelParametrized.init_from(fsm)
        inputs = fsmp.inputs
        for i, j in zip(inputs, [np.linspace(inp0["lb"], inp0["ub"], inp0["n"]), inp1]):
            np.testing.assert_almost_equal(i, j)
    
    # Test combinations (2)
    def test_sample_ode_x0_ode_t0(self, default_model):
        fsm = default_model.fsm
        x0 = [[0.0187, 0.000498], [0.0291, 0.002]]
        t0 = {"lb":0.0, "ub":1.0, "n":7}
        fsm.ode_x0 = x0
        fsm.ode_t0 = t0
        fsmp = FisherModelParametrized.init_from(fsm)
        for p, q in zip(fsmp.ode_x0, x0):
            np.testing.assert_almost_equal(p, q)
        np.testing.assert_almost_equal(fsmp.ode_t0, np.linspace(t0["lb"], t0["ub"], t0["n"]))
    
    def test_sample_ode_x0_times(self, default_model):
        fsm = default_model.fsm
        x0 = [[0.0187, 0.000498], [0.0291, 0.002]]
        t = {"lb":3.21, "ub":11.44, "n":2}
        fsm.ode_x0 = x0
        fsm.times = t
        fsmp = FisherModelParametrized.init_from(fsm)
        for p, q in zip(fsmp.ode_x0, x0):
            np.testing.assert_almost_equal(p, q)
        np.testing.assert_almost_equal(fsmp.times, np.full(tuple(len(q) for q in fsmp.inputs) + (t["n"],), np.linspace(t["lb"], t["ub"], t["n"])))

    def test_sample_ode_t0_times(self, default_model):
        fsm = default_model.fsm
        t0 = {"lb":-2.13, "ub":5.05, "n":6}
        t = {"lb":3.21, "ub":11.44, "n":2}
        fsm.ode_t0 = t0
        fsm.times = t
        fsmp = FisherModelParametrized.init_from(fsm)
        np.testing.assert_almost_equal(fsmp.ode_t0, np.linspace(t0["lb"], t0["ub"], t0["n"]))
        np.testing.assert_almost_equal(fsmp.times, np.full(tuple(len(q) for q in fsmp.inputs) + (t["n"],), np.linspace(t["lb"], t["ub"], t["n"])))

    # TODO
    # def test_sample_ode_x0_inputs(self, default_model):
    #     pass
    # 
    # def test_sample_ode_t0_inputs(self, default_model):
    #     pass
    # 
    # def test_sample_times_inputs(self, default_model):
    #     pass
    # 
    # # Test combinations (3)
    # def test_sample_ode_x0_times_inputs(self, default_model):
    #     pass
    # 
    # def test_sample_ode_t0_times_inputs(self, default_model):
    #     pass
    # 
    # def test_sample_times_inputs(self, default_model):
    #     pass
    # 
    # def test_sample_ode_x0_ode_t0_inputs(self, default_model):
    #     pass
    # 
    # # Test combinations (4)
    # def test_sample_ode_x0_ode_t0_times_inputs(self, default_model):
    #     pass


class Test_fsmp_set_get:
    def test_set_get_t0(self, default_model):
        fsm = default_model.fsm
        t0 = {"lb":2.22, "ub":56.3, "n":8}
        fsm.ode_t0 = t0
        fsmp = FisherModelParametrized.init_from(fsm)
        t0 = 1.0
        fsmp.t0 = t0
        assert fsmp.t0 == t0

    def test_set_get_x0(self, default_model):
        fsm = default_model.fsm
        x0 = np.array([4.22, 9.44])
        fsm.ode_x0 = x0
        fsmp = FisherModelParametrized.init_from(fsm)
        x0 = np.array([33.2, 12.3])
        fsmp.x0 = x0
        np.testing.assert_almost_equal(x0, fsmp.x0)
    
    def test_set_get_times_identical(self, default_model):
        fsm = default_model.fsm
        times = {"lb":4.22, "ub":9.44, "n":8}
        fsm.times = times
        fsm.identical_times = True
        fsmp = FisherModelParametrized.init_from(fsm)
        times = np.array([2.0, 3.0, 66.0])
        fsmp.times = times
        np.testing.assert_almost_equal(times, fsmp.times)
    
    def test_set_get_times_not_identical(self, default_model):
        fsm = default_model.fsm
        t = {"lb":4.22, "ub":9.44, "n":8}
        fsm.times = t
        fsmp = FisherModelParametrized.init_from(fsm)
        times = np.full(fsmp.times.shape, np.linspace(3.119, 6.489, t["n"]))
        fsmp.times = times
        np.testing.assert_almost_equal(times, fsmp.times)
    
    def test_set_get_inputs(self, default_model):
        fsm = default_model.fsm
        inp0 = {"lb":-2.0, "ub":51.2, "n":3}
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
    
    def test_get_parameters(self, default_model):
        fsm = default_model.fsm
        np.testing.assert_almost_equal(fsm.parameters, default_model.fsmp.parameters)
    
    def test_get_ode_args(self, default_model):
        fsm = default_model.fsm
        np.testing.assert_almost_equal(fsm.ode_args, default_model.fsmp.ode_args)

    def test_set_immutable_x0(self, default_model):
        fsmp = default_model.fsmp
        with pytest.raises(AttributeError):
            fsmp.ode_x0 = np.linspace(0, 10)
    
    def test_set_immutable_t0(self, default_model):
        fsmp = default_model.fsmp
        # with pytest.raises(AttributeError):
        with pytest.raises(AttributeError):
            fsmp.ode_t0 = np.linspace(0, 10)
    
    def test_set_immutable_times(self, default_model):
        fsmp = default_model.fsmp
        with pytest.raises(AttributeError):
            fsmp.times = np.linspace(0, 10)

    def test_set_immutable_inputs(self, default_model):
        fsmp = default_model.fsmp
        with pytest.raises(AttributeError):
            fsmp.inputs = [
                np.linspace(0, 10),
                np.linspace(3.0, 45.0)
            ]
