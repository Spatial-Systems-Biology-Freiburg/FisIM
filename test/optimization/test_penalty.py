import pytest
import numpy as np

from FisInMa.model import FisherModelParametrized
from FisInMa.optimization.penalty import _discrete_penalizer, DISCRETE_PENALTY_FUNCTIONS

from test.setUp import default_model_parametrized


def generate_penalty_combs():
    params = []
    return [[2, 2, 2, 2, 3, False, key] for key in DISCRETE_PENALTY_FUNCTIONS.keys()]

    # for penalty_name in DISCRETE_PENALTY_FUNCTIONS.keys():
    #     params.append([2, 2, 2, 2, 3, False, penalty_name])
    # return params


class Test_DiscrPenalty:
    @pytest.mark.parametrize("N_x0,n_t0,n_times,n_inputs_0,n_inputs_1,identical_times,penalty_name", generate_penalty_combs())
    def test_ode_t0(self, default_model_parametrized, penalty_name):
        fsm = default_model_parametrized.fsm
        disc = 0.0002
        fsm.ode_t0 = {"lb":0.00, "ub":0.001, "n":3, "discrete":disc}
        # Initialize model with initial guess
        fsmp = FisherModelParametrized.init_from(fsm)

        # Test if discretization was correctly used
        np.testing.assert_almost_equal(fsmp.ode_t0_def.discrete, np.arange(fsm.ode_t0.lb, fsm.ode_t0.ub + disc/2, disc))
        
        # Calculate penalty for initial_guess = discretization
        # The penalty should be non-effective (ie. = 1.0)
        res, _ = _discrete_penalizer(fsmp, penalizer_name=penalty_name)
        np.testing.assert_almost_equal(res, 1.0)
        
        # Now set the values to a non-discrete conforming value
        fsmp.ode_t0 = [0.000, 0.0006, 0.0005]

        # Test if the penalty is now below 1.0
        res, _ = _discrete_penalizer(fsmp, penalizer_name=penalty_name)
        assert res < 1.0

        # Now see if after some time the penalty returns to 1 when going near specified discretization value
        n_runs = 100
        res_prev = None
        converge = False
        for i in (n_runs - np.arange(n_runs+1)):
            fsmp.ode_t0 = [0.000, 0.001 + 0.0001*i/n_runs, 0.001 + 0.0002*i/n_runs]
            res, _ = _discrete_penalizer(fsmp, penalizer_name=penalty_name)
            if res_prev !=None and res > res_prev:
                converge = True
            if converge == True:
                assert res_prev < res
            res_prev = res
        
        # Also test if we have reached 1.0 again
        np.testing.assert_almost_equal(res, 1.0)

    @pytest.mark.parametrize("N_x0,n_t0,n_times,n_inputs_0,n_inputs_1,identical_times,penalty_name", generate_penalty_combs())
    def test_times(self, default_model_parametrized, penalty_name):
        fsm = default_model_parametrized.fsm
        disc = 0.5
        fsm.times = {"lb":0.00, "ub":10.0, "n":5, "discrete":disc}
        # Initialize model with initial guess
        fsmp = FisherModelParametrized.init_from(fsm)

        # Test if discretization was correctly used
        np.testing.assert_almost_equal(fsmp.times_def.discrete, np.arange(fsm.times.lb, fsm.times.ub + disc/2, disc))
        
        # Calculate penalty for initial_guess = discretization
        # The penalty should be non-effective (ie. = 1.0)
        res, _ = _discrete_penalizer(fsmp, penalizer_name=penalty_name)
        np.testing.assert_almost_equal(res, 1.0)
        
        # Now set the values to a non-discrete conforming value
        fsmp.times = np.full((2,3,5), np.array([
            [0.0, 1.1, 1.6, 2.0, 7.5],
            [0.0, 2.1, 2.4, 6.5, 9.5],
            [0.0, 2.2, 2.6, 6.0, 10.0]
        ]))

        # Test if the penalty is now below 1.0
        res, _ = _discrete_penalizer(fsmp, penalizer_name=penalty_name)
        assert res < 1.0

        # Now see if after some time the penalty returns to 1 when going near specified discretization value
        n_runs = 100
        res_prev = None
        converge = False
        for i in (n_runs - np.arange(n_runs+1)):
            fsmp.times = np.full((2,3,5), np.array([
                [0.0, 0.5 + 0.1*i/n_runs, 1.5 + 0.2*i/n_runs, 2.0, 7.5],
                [0.0, 2.0 + 0.1*i/n_runs, 2.5 - 0.1*i/n_runs, 6.5, 9.5],
                [0.0, 2.0 + 0.2*i/n_runs, 2.5 + 0.1*i/n_runs, 6.0, 10.0]
            ]))
            res, _ = _discrete_penalizer(fsmp, penalizer_name=penalty_name)
            if res_prev !=None and res > res_prev:
                converge = True
            if converge == True:
                assert res_prev < res
            res_prev = res

        # Also test if we have reached 1.0 again
        np.testing.assert_almost_equal(res, 1.0)
    
    @pytest.mark.parametrize("N_x0,n_t0,n_times,n_inputs_0,n_inputs_1,identical_times,penalty_name", generate_penalty_combs())
    def test_inputs(self, default_model_parametrized, penalty_name):
        fsm = default_model_parametrized.fsm
        disc = 0.25
        fsm.inputs[0] = {"lb":5.0, "ub":8.0, "n":3, "discrete":disc}
        # Initialize model with initial guess
        fsmp = FisherModelParametrized.init_from(fsm)

        # Test if discretization was correctly used
        np.testing.assert_almost_equal(fsmp.inputs_def[0].discrete, np.arange(fsm.inputs[0]["lb"], fsm.inputs[0]["ub"] + disc/2, disc))

        # Calculate penalty for initial_guess = discretization
        # The penalty should be non-effective (ie. = 1.0)
        res, _ = _discrete_penalizer(fsmp, penalizer_name=penalty_name)
        np.testing.assert_almost_equal(res, 1.0)

        # Now set the values to a non-discrete conforming value
        fsmp.inputs[0] = np.array([5.0, 5.2, 6.3])

        # Test if the penalty is now below 1.0
        res, _ = _discrete_penalizer(fsmp, penalizer_name=penalty_name)
        assert res < 1.0

        # Now see if after some time the penalty returns to 1 when going near specified discretization value
        n_runs = 100
        res_prev = None
        converge = False
        for i in (n_runs - np.arange(n_runs+1)):
            fsmp.inputs[0] = np.array([5.0, 5.0 + 0.2*i/n_runs, 6.5 - 0.2*i/n_runs])
            res, _ = _discrete_penalizer(fsmp, penalizer_name=penalty_name)
            if res_prev !=None and res > res_prev:
                converge = True
            if converge == True:
                assert res_prev < res
            res_prev = res
        
        # Also test if we have reached 1.0 again
        np.testing.assert_almost_equal(res, 1.0)

    # TODO - but needs sampling over x0 first!
    # def test_ode_x0_discr_penalty(self, default_model):
