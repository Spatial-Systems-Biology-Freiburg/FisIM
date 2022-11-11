import numpy as np

from FisInMa.model import FisherModelParametrized
from FisInMa.solving import *

class Test_Criteria:
    @classmethod
    def define_matrices(self):
        S = np.array([
            [1,2,3],
            [0,1,2],
            [0,0,1]
        ])
        C = np.eye(3)
        F = S @ C @ S.T
        return S, C, F

    def test_fisher_determinant(self):
        S, C, F = self.define_matrices()
        d1 = fisher_determinant(None, S, C)
        d2 = np.linalg.det(F)
        np.testing.assert_almost_equal(d1, d2)

    def test_fisher_sumeigval(self):
        S, C, F = self.define_matrices()
        s1 = fisher_sumeigenval(None, S, C)
        s2 = np.sum(np.linalg.eigvals(F))
        np.testing.assert_almost_equal(s1, s2)

    def test_fisher_mineigenval(self):
        S, C, F = self.define_matrices()
        m1 = fisher_mineigenval(None, S, C)
        m2 = mineigval = np.min(np.linalg.eigvals(F))
        np.testing.assert_almost_equal(m1, m2)

    def test_fisher_ratioeigenval(self):
        S, C, F = self.define_matrices()
        r1 = fisher_ratioeigenval(None, S, C)
        eigvals = np.linalg.eigvals(F)
        r2 = np.min(eigvals) / np.max(eigvals)
        np.testing.assert_almost_equal(r1, r2)
