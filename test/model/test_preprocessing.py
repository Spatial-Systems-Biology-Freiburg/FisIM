#!/usr/bin/env python3

import pytest
import numpy as np

from FisInMa import VariableDefinition


class TestVariableDefinition:
    def test_init_no_discretization(self):
        lb = 2.0
        ub = 4.0
        n = 3
        v = VariableDefinition(lb, ub, n)
        np.testing.assert_almost_equal(v.initial_guess, np.linspace(lb, ub, n))

    def test_init_with_discretization(self):
        lb = 2.0
        ub = 4.0
        n = 3
        dx = 0.5
        v = VariableDefinition(lb, ub, n, dx)
        np.testing.assert_almost_equal(v.initial_guess, np.array([2.5, 3.0, 3.5]))

    def test_init_too_many_steps(self):
        lb = 2.0
        ub = 4.0
        n = 10
        dx = 0.5
        unique = True
        with pytest.raises(ValueError):
            VariableDefinition(lb, ub, n, dx, unique=unique)
