#!/usr/bin/env python3

import unittest
import sys, os

sys.path.append(os.getcwd())

from setUp import Setup_Class
from model.test_fisher_model import Test_FisherModelParametrized_Init, Test_FisherModelParametrized_Set_Get
from model.test_preprocessing import TestVariableDefinition
from solving.test_solver import Test_SolvingMethods
from optimization import Test_ScipyCalculateConstraints, Test_ScipyGlobalOptimAlgorithms

if __name__ == "__main__":
    # Create a runner for all tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Create the suite for TestVariableDefinition
    vardef = unittest.TestSuite()
    # Add Tests
    vardef.addTest(unittest.makeSuite(TestVariableDefinition))
    
    # Create the test suite for FisherModelParametrized
    fsmp = unittest.TestSuite()
    # Add Tests
    fsmp.addTest(unittest.makeSuite(Test_FisherModelParametrized_Init))
    fsmp.addTest(unittest.makeSuite(Test_FisherModelParametrized_Set_Get))
    
    # Create the test suite for Solver methods
    solve_fsmp = unittest.TestSuite()
    solve_fsmp.addTest(unittest.makeSuite(Test_SolvingMethods))

    # Test suite für Optimization methods
    optim_fsm = unittest.TestSuite()
    optim_fsm.addTest(unittest.makeSuite(Test_ScipyGlobalOptimAlgorithms))
    optim_fsm.addTest(unittest.makeSuite(Test_ScipyCalculateConstraints))

    # Run the suites
    runner.run(vardef)
    runner.run(fsmp)
    # runner.run(solve_fsmp)
    runner.run(optim_fsm)
