import numpy as np
import pytest

from FisInMa.model import FisherModelParametrized
from FisInMa.solving import *

from test.setUp import extended_baranyi_model, extended_baranyi_model_small, extended_baranyi_model_parametrized


class TestObservableGeneration:
    @pytest.mark.parametrize("identical_times", [True, False])
    def test_ode_sol_identical(self, extended_baranyi_model_small):
        fsm = extended_baranyi_model_small.fsm
        results = []
        for i in range(4):
            fsm.obs_fun = [i]
            fsmp = FisherModelParametrized.init_from(fsm)

            results.append(get_S_matrix(fsmp))

        for i, j in itertools.product(range(4), range(4)):
            for res_1, res_2 in zip(
                results[i][2],
                results[j][2]
            ):
                np.testing.assert_allclose(res_1.ode_solution.y, res_2.ode_solution.y)
