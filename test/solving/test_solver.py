import numpy as np
import scipy as sp
import pytest
import itertools

from FisInMa.model import FisherModelParametrized
from FisInMa.solving import *

from test.setUp import default_model, default_model_parametrized, default_model_small, pool_model_small


def comb_gen_solving():
    # We do not want to do stress testing but need to verify for a certain amount of combinatorics
    # This tries to find a middle ground in testing
    # params = []
    for identical_times, relative_sensitivities in itertools.product(*([[True, False]]*2)):
        print(identical_times, relative_sensitivities)
    
        yield [2,3,5,2,1,identical_times,relative_sensitivities]
        yield [1,2,3,5,2,identical_times,relative_sensitivities]
        yield [2,1,2,3,5,identical_times,relative_sensitivities]
        yield [5,2,1,2,3,identical_times,relative_sensitivities]
        yield [3,5,2,1,2,identical_times,relative_sensitivities]


@pytest.mark.parametrize("N_x0,n_t0,n_times,n_inputs_0,n_inputs_1,identical_times,relative_sensitivities", comb_gen_solving())
def test_get_S_matrix(default_model_parametrized, relative_sensitivities):
    fsmp = default_model_parametrized.fsmp
    S, C, solutions = get_S_matrix(fsmp)#, relative_sensitivities=relative_sensitivities)


def test_ode_rhs(default_model):
    fsmp = default_model.fsmp

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
        x0_full = np.concatenate((x0, np.zeros(n_x * n_p)))
        res = ode_rhs(t0, x0_full, fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, fsmp.ode_dfdx0, Q, fsmp.parameters, fsmp.ode_args, n_x, n_p)
        
        np.testing.assert_almost_equal(res[:n_x], fsmp.ode_fun(t0, x0, Q, fsmp.parameters, fsmp.ode_args))
        np.testing.assert_almost_equal(res[n_x:], np.array(fsmp.ode_dfdp(t0, x0, Q, fsmp.parameters, fsmp.ode_args)).flatten())

        # Test for non-zero sensitivity values
        s0 = (np.zeros(n_x * n_p) + 1.0).reshape((n_x, n_p))
        x0_full = np.concatenate((x0, s0.flatten()))
        res = ode_rhs(t0, x0_full, fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, fsmp.ode_dfdx0, Q, fsmp.parameters, fsmp.ode_args, n_x, n_p)
        
        # Mimic calculation of sensitivities
        f_ty = fsmp.ode_fun(t0, x0, Q, fsmp.parameters, fsmp.ode_args)
        np.testing.assert_almost_equal(res[:n_x], f_ty)
        dfdp_ty = fsmp.ode_dfdp(t0, x0, Q, fsmp.parameters, fsmp.ode_args)
        dfdx_ty = fsmp.ode_dfdx(t0, x0, Q, fsmp.parameters, fsmp.ode_args)
        sensitivities = dfdp_ty + np.matmul(dfdx_ty, s0)
        np.testing.assert_almost_equal(res[:n_x], np.array(f_ty).flatten())
        np.testing.assert_almost_equal(res[n_x:], sensitivities.flatten())


def comb_gen_automation():
    # We do not want to do stress testing but need to verify for a certain amount of combinatorics
    # This tries to find a middle ground in testing
    criteria = [
        fisher_determinant,
        fisher_mineigenval,
        fisher_ratioeigenval,
        fisher_sumeigenval
    ]
    return list(itertools.product([True, False], criteria, *([[True,False]])))


# These tests just check that the functions can be called properly
@pytest.mark.parametrize("identical_times,criterion,relative_sensitivities", comb_gen_automation())
def test_calculate_criterion(default_model_small, criterion, relative_sensitivities):
    fsmp = default_model_small.fsmp
    fsr = calculate_fisher_criterion(fsmp, criterion=criterion, relative_sensitivities=relative_sensitivities)


@pytest.mark.parametrize("identical_times,criterion,relative_sensitivities", comb_gen_automation())
def test_use_initial_value_as_parameter(pool_model_small, criterion, relative_sensitivities):
    fsmp = pool_model_small.fsmp
    fsr = calculate_fisher_criterion(fsmp, criterion=criterion, relative_sensitivities=relative_sensitivities)
