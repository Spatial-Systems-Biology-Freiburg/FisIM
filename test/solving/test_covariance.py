import numpy as np
import pytest

from FisInMa.model import FisherModelParametrized
from FisInMa.solving import *

from test.setUp import default_model_small, pool_model_small


def comb_gen_covar():
    return list(itertools.product(*([[True]]*3), [0.2, 1.3, 3.0]))


class TestCovariance:
    @pytest.mark.parametrize("identical_times,relative_sensitivities,cov_rel_abs,cov_val", comb_gen_covar())
    def test_covariance_def(self, default_model_small, relative_sensitivities, cov_rel_abs, cov_val):
        fsm = default_model_small.fsm
        fsm.covariance = {"rel": cov_val} if cov_rel_abs else {"abs": cov_val}
        fsmp = FisherModelParametrized.init_from(fsm)

    @pytest.mark.parametrize("identical_times,relative_sensitivities,cov_rel_abs,cov_val", comb_gen_covar())
    def test_covariance_calc(self, default_model_small, relative_sensitivities, cov_rel_abs, cov_val):
        fsm = default_model_small.fsm
        cov_def = {"rel": cov_val} if cov_rel_abs else {"abs": cov_val}
        fsm.covariance = cov_def
        fsmp = FisherModelParametrized.init_from(fsm)

        S, C, _ = get_S_matrix(fsmp, relative_sensitivities)
        if cov_rel_abs==relative_sensitivities:
            np.testing.assert_allclose(np.diag(C), np.full(C.shape[0], 1/cov_def["rel"]**2 if cov_rel_abs else 1/cov_def["abs"]**2))
        else:
            np.testing.assert_allclose(np.diag(C)!=1, np.ones(C.shape[0]))
    
    @pytest.mark.parametrize("identical_times,relative_sensitivities,cov_rel_abs,cov_val", comb_gen_covar())
    def test_covariance_def_2(self, pool_model_small, relative_sensitivities, cov_rel_abs, cov_val):
        fsm = pool_model_small.fsm
        fsm.covariance = {"rel": cov_val} if cov_rel_abs else {"abs": cov_val}
        fsmp = FisherModelParametrized.init_from(fsm)

    @pytest.mark.parametrize("identical_times,relative_sensitivities,cov_rel_abs,cov_val", comb_gen_covar())
    def test_covariance_calc_2(self, pool_model_small, relative_sensitivities, cov_rel_abs, cov_val):
        fsm = pool_model_small.fsm
        cov_def = {"rel": cov_val} if cov_rel_abs else {"abs": cov_val}
        fsm.covariance = cov_def
        print(type(cov_def))
        print(fsm.covariance)
        fsmp = FisherModelParametrized.init_from(fsm)

        S, C, _ = get_S_matrix(fsmp, relative_sensitivities)
        if cov_rel_abs==relative_sensitivities:
            np.testing.assert_allclose(np.diag(C), np.full(C.shape[0], 1/cov_def["rel"]**2 if cov_rel_abs else 1/cov_def["abs"]**2))
        else:
            np.testing.assert_allclose(np.diag(C)!=1, np.ones(C.shape[0]))
