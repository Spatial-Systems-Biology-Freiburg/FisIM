import pytest
import numpy as np

from FisInMa.model import FisherModelParametrized, FisherResults
from FisInMa.optimization.caller import find_optimal

from test.setUp import default_model_small


class Test_Caller:
    @pytest.mark.parametrize("identical_times", [True, False])
    def test_differential_evolution(self, default_model_small):
        fsm = default_model_small.fsm
        fsm.ode_t0 = 0.0
        fsm.ode_x0 = [np.array([0.05, 0.001])]
        fsm.inputs=[
            np.arange(2, 2+2),
            np.arange(5, 2+5)
        ]
        fsm.times = {"lb":0.0, "ub":10.0, "n":2}
        # Choose very small iteration and population numbers.
        # This is not about convergence, but about if the method will not fail.
        fsr = find_optimal(fsm, "scipy_differential_evolution", workers=1, maxiter=1, popsize=2)
        assert type(fsr) == FisherResults

    @pytest.mark.parametrize("identical_times", [True, False])
    def test_basinhopping(self, default_model_small):
        fsm = default_model_small.fsm
        fsm.ode_t0 = 0.0
        fsm.ode_x0 = [np.array([0.05, 0.001])]
        fsm.inputs=[
            np.arange(2, 2+2),
            np.arange(5, 5+2)
        ]
        fsm.times = {"lb":0.0, "ub":10.0, "n":2}
        # Choose very small iteration and population numbers.
        # This is not about convergence, but about if the method will not fail.
        fsr = find_optimal(fsm, "scipy_basinhopping", niter=1, interval=2)
        assert type(fsr) == FisherResults

    @pytest.mark.parametrize("identical_times", [True, False])
    def test_brute(self, default_model_small):
        fsm = default_model_small.fsm
        fsm.ode_t0 = 0.0
        fsm.ode_x0 = [np.array([0.05, 0.001])]
        fsm.inputs=[
            np.arange(2, 2+2),
            np.arange(5, 5+2)
        ]
        fsm.times = {"lb":0.0, "ub":10.0, "n":2}

        # Choose very small iteration and population numbers.
        # This is not about convergence, but about if the method will not fail.
        fsr = find_optimal(fsm, "scipy_brute", Ns=1, workers=1)
        assert type(fsr) == FisherResults
