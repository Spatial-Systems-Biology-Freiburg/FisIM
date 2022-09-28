import numpy as np
from dataclasses import dataclass

@dataclass
class FischerResult:
    '''Class to store a single fischer result.
    Use a list of this class to store many results.'''
    observable: np.ndarray
    times: np.ndarray
    parameters: list
    q_arr: list
    constants: list
    y0: np.array

    def to_savedict(self):
        '''Used to store results in database'''
        d = {
            "observable": apply_marks(self.observable),
            "times": apply_marks(self.times),
            "parameters": apply_marks(self.parameters),
            "q_arr": apply_marks(self.q_arr),
            "constants": apply_marks(self.constants),
            "y0": apply_marks(self.y0)
        }
        return d


@dataclass
class FischerModel(FischerResult):
    '''Class derived from FischerResult used to store a full singular model description.
    Compared to the FischerResult class, we additionally provide information to solve the ODE.'''
    rhs: function
    jacobian: function
    initial_values: (np.array, float)