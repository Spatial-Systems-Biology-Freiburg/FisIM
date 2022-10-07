import numpy as np
from dataclasses import dataclass, field


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
class _FischerModelBase:
    times: np.ndarray
    parameters: list
    q_values: list
    constants: list
    y0_t0: tuple
    ode_func: callable
    observable_func: callable


@dataclass
class _FischerModelOptions():
    '''Class derived from FischerResult used to store a full singular model description.
    Compared to the FischerResult class, we additionally provide information to solve the ODE.'''
    jacobian: callable = None
    relative_sensitivities: bool = False


@dataclass
class _FischerResultBase(_FischerModelBase):
    '''Class to store a single fischer result.
    Use a list of this class to store many results.'''
    observable: np.ndarray
    sensitivity_matrix: np.ndarray
    covariance_matrix: np.ndarray
    ode_solutions: list


@dataclass
class _FischerResultOptions(_FischerModelOptions):
    pass


@dataclass
class FischerModel(_FischerModelOptions, _FischerModelBase):
    def __post_init__(self):
        q_values_shape = tuple(len(q) for q in self.q_values)
        # Store the correct shape for the times variable
        if self.times.ndim == 1:
            self._times_1d = True
            self.__times_shape = q_values_shape + (self.times.shape[0],)
        else:
            self._times_1d = False
            self.__times_shape = self.times.shape

    def set_times(self, t):
        if t.shape == self.__times_shape:
            self.times = t
        elif t.ndim == 1:
            self.times = np.full(self.__times_shape, t)
        else:
            raise ValueError("Array does not have the correct shape")


@dataclass
class FischerResult(_FischerResultOptions, _FischerResultBase):
    def to_savedict(self):
        '''Used to store results in database'''
        d = {
            "times": apply_marks(self.times),
            "parameters": apply_marks(self.parameters),
            "q_values": apply_marks(self.q_values),
            "constants": apply_marks(self.constants),
            "y0_t0": apply_marks(self.y0_t0),
            "observable": apply_marks(self.observable),
            "observable_func": apply_marks(self.observable_func.__name__),
            "sensitivity_matrix": apply_marks(self.sensitivity_matrix),
            "covariance_matrix": apply_marks(self.covariance_matrix),
            "ode_solutions": apply_marks(ode_solutions)
        }
        return d