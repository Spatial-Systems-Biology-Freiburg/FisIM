#!/usr/bin/env python3

import unittest

from model import *

if __name__ == "__main__":
    # Create a runner for all tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Create the suite for TestVariableDefinition
    vardef = unittest.TestSuite()
    # Add Tests
    vardef.addTest(unittest.makeSuite(TestVariableDefinition))
    
    # Create the test suite for FischerModelParametrized
    fsmp = unittest.TestSuite()
    # Add Tests
    fsmp.addTest(unittest.makeSuite(Test_FischerModelParametrized_Init))
    fsmp.addTest(unittest.makeSuite(Test_FischerModelParametrized_Set_Get))
    
    # Run the suites
    runner.run(vardef)
    runner.run(fsmp)
