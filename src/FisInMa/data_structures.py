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
    time_interval: tuple
    n_times: int
    parameters: list
    q_values: list
    constants: list
    y0: np.ndarray
    ode_func: callable
    criterion_func: callable


@dataclass
class _FischerModelOptions():
    '''Class derived from FischerResult used to store a full singular model description.
    Compared to the FischerResult class, we additionally provide information to solve the ODE.'''
    jacobian: callable = None
    relative_sensitivities: bool = False
    individual_times: bool = True


@dataclass 
class _FischerModelParametrizedBase (_FischerModelBase):
    pass


@dataclass
class _FischerModelParametrizedOptions(_FischerModelOptions):
    times: np.ndarray = None


@dataclass
class _FischerResultBase(_FischerModelParametrizedBase):
    '''Class to store a single fischer result.
    Use a list of this class to store many results.'''
    criterion: np.ndarray
    sensitivity_matrix: np.ndarray
    covariance_matrix: np.ndarray
    ode_solutions: list


@dataclass
class _FischerResultOptions(_FischerModelParametrizedOptions):
    pass


@dataclass
class FischerModel(_FischerModelOptions, _FischerModelBase):
    pass


@dataclass
class FischerModelParametrized(_FischerModelParametrizedOptions, _FischerModelParametrizedBase):
    def initialize_from(fsm, times_initial_guess=None):
        args = {key: val for key, val in fsm.__dict__.items()}

        fsmp = FischerModelParametrized(
            **args
        )

        if times_initial_guess==None:
            t = np.linspace(fsmp.time_interval[0], fsmp.time_interval[1], fsmp.n_times+2)[1:-1]
            if fsm.individual_times == True:
                times_initial_guess = np.full(fsmp._times_shape, t)
            else:
                times_initial_guess = t

        fsmp.set_times(times_initial_guess)
        
        return fsmp, times_initial_guess

    def __post_init__(self):
        self._q_values_shape = tuple(len(q) for q in self.q_values)
        # Store the correct shape for the times variable
        # if self.individual_times == True:
        self._times_shape = self._q_values_shape + (self.n_times,)
        # else:
        #     self._times_shape = (self.n_times,)

    def set_times(self, t):
        if t.shape == self._times_shape:
            self.times = t
        elif t.ndim == 1:
            self.times = np.full(self._times_shape, t)
        else:
            raise ValueError("Array does not have the correct shape")


@dataclass
class FischerResult(_FischerResultOptions, _FischerResultBase):
    def to_savedict(self):
        '''Used to store results in database'''
        d = {
            "time_interval": apply_marks(self.time_interval),
            "times": apply_marks(self.times),
            "parameters": apply_marks(self.parameters),
            "q_values": apply_marks(self.q_values),
            "constants": apply_marks(self.constants),
            "y0": apply_marks(self.y0),
            "criterion": apply_marks(self.criterion),
            "criterion_func": apply_marks(self.criterion_func.__name__),
            "sensitivity_matrix": apply_marks(self.sensitivity_matrix),
            "covariance_matrix": apply_marks(self.covariance_matrix),
            "ode_solutions": apply_marks(ode_solutions)
        }
        return d