import unittest
import numpy as np

from FisInMa.model import FischerModelParametrized
from FisInMa.solving import *

from model import Setup_Class

class Test_SolvingMethods(Setup_Class):
    def test_get_S_matrix(self):
        fsmp = self.fsmp
        S, C, solutions = get_S_matrix(fsmp)


class TestCriterions(Setup_Class):
    pass


class Test_CriterionCalculationAutomation(Setup_Class):
    pass