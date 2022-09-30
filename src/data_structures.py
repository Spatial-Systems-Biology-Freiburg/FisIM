import numpy as np
from dataclasses import dataclass
import numpy as np


# Used to store results in mongodb
def apply_marks(ls: list):
    if type(ls) == np.ndarray:
        return ["np.ndarray", ls.tolist()]
    elif type(ls) == list:
        return [apply_marks(l) for l in ls]
    else:
        return ls


# Used to convert back from mongodb stored results
def revert_marks(ls: list):
    if type(ls) == list and len(ls) == 2 and ls[0] == "np.ndarray" and type(ls[1]) == list:
        return np.array(ls[1])
    elif type(ls) == list:
        return [revert_marks(l) for l in ls]
    else:
        return ls


@dataclass
class FischerResult:
    '''Class to store a single fischer result.
    Use a list of this class to store many results.'''
    observable: np.ndarray
    times: np.ndarray
    parameters: list
    q_values: list
    constants: list
    y0_t0: tuple

    def to_savedict(self):
        '''Used to store results in database'''
        d = {
            "observable": apply_marks(self.observable),
            "times": apply_marks(self.times),
            "parameters": apply_marks(self.parameters),
            "q_values": apply_marks(self.q_values),
            "constants": apply_marks(self.constants),
            "y0_t0": apply_marks(self.y0_t0)
        }
        return d


@dataclass
class FischerModel(FischerResult):
    '''Class derived from FischerResult used to store a full singular model description.
    Compared to the FischerResult class, we additionally provide information to solve the ODE.'''
    ode_func: callable
    jacobian: callable
