import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from .preprocessing import VariableDefinition


@dataclass
class _FischerVariablesBase:
    ode_y0: np.ndarray
    ode_t0: float
    times: np.ndarray
    inputs: list
    parameters: tuple
    constants: tuple


@dataclass
class _FischerVariablesOptions:
    identical_times: bool = False


@dataclass
class _FischerOdeFunctions:
    ode_fun: callable
    ode_dfdx: callable
    ode_dfdp: callable


@dataclass
class _FischerObservableFunctionsOptional:
    obs_fun: callable = None
    obs_dfdx: callable = None
    obs_dfdp: callable = None


@dataclass
class FischerVariables(_FischerVariablesOptions, _FischerVariablesBase):
    pass


@dataclass
class _FischerModelBase(_FischerOdeFunctions, _FischerVariablesBase):
    pass


@dataclass
class _FischerModelOptions(_FischerVariablesOptions, _FischerObservableFunctionsOptional):
    pass


@dataclass
class FischerModel(_FischerModelOptions, _FischerModelBase):
    pass


@dataclass
class _FischerModelParametrizedBase(_FischerOdeFunctions):
    _fsm_var_def: FischerVariables
    _fsm_var_vals: FischerVariables


@dataclass
class _FischerModelParametrizedOptions(_FischerModelOptions):
    pass


@dataclass
class FischerModelParametrized(_FischerModelParametrizedOptions, _FischerModelParametrizedBase):
    def init_from(fsm: FischerModel):
        # Create distinct classes to store
        # 1) Initial definition of model (ie. sample over certain variable; specify tuple of (min, max, n, dx, guess_method) or explicitly via np.array([...]))
        # 2) Explicit values together with initial guess such that every variable is parametrized
        _fsm_var_def = FischerVariables(
            fsm.ode_y0,
            fsm.ode_t0,
            fsm.times,
            fsm.inputs,
            fsm.parameters,
            fsm.constants,
            fsm.identical_times,
        )
        _fsm_var_vals = deepcopy(_fsm_var_def)

        # Check which external inputs are being sampled
        _inputs_def = []
        _inputs_vals = []
        for q in fsm.inputs:
            if type(q) == tuple and len(q) >= 3:
                q_def = VariableDefinition(*q)
                _inputs_def.append(q_def)
                _inputs_vals.append(q_def.initial_guess)
            else:
                _inputs_def.append(None)
                _inputs_vals.append(q)
        
        _fsm_var_def.inputs = _inputs_def
        _fsm_var_vals.inputs = _inputs_vals
        inputs_shape = tuple(len(q) for q in _inputs_vals)

        # Check if we want to sample over initial values
        y0_def = []
        y0_vals = []
        if type(fsm.ode_y0)==float:
            y0_def = [np.array([fsm.ode_y0])]
            y0_vals = [np.array([fsm.ode_y0])]
        elif type(fsm.ode_y0)==np.ndarray and fsm.ode_y0.ndim == 1:
            y0_def = [fsm.ode_y0]
            y0_vals = [fsm.ode_y0]
            print("Warning!: Initializing FisherModel with 1d array for y0 is beeing interpreted as one single initial value (no sampling) of size " + str(len(fsm.ode_y0)) + ". To sample over a fixed number of values specify as list of 1d arrays with only a single entry.")
        else:
            for y in fsm.ode_y0:
                if type(y) == tuple and len(y) >= 3:
                    y0 = VariableDefinition(*y)
                    y0_def = y0
                    y0_vals.append(y0.initial_guess)
                else:
                    y0_def = None
                    y0_vals.append(np.array(y, dtype=float))

        _fsm_var_def.ode_y0 = y0_def
        _fsm_var_vals.ode_y0 = y0_vals

        # Check if time values are sampled
        if type(fsm.times) == tuple and len(fsm.times) >= 3:
            t = VariableDefinition(*fsm.times)
            _fsm_var_def.times = t
            _fsm_var_vals.times = t.initial_guess
        else:
            _fsm_var_def.times = None
            _fsm_var_vals.times = fsm.times
        # If non-identical times were chosen, expand initial guess to full array
        if fsm.identical_times==False:
            _fsm_var_vals.times = np.full(inputs_shape + _fsm_var_vals.times.shape, _fsm_var_vals.times)

        # Check if we want to sample over initial time
        if type(fsm.ode_t0) == tuple and len(fsm.ode_t0) >= 3:
            t0 = VariableDefinition(*fsm.ode_t0)
            _fsm_var_def.ode_t0 = t0
            _fsm_var_vals.ode_t0 = t0.initial_guess
        elif type(fsm.ode_t0) == float:
            _fsm_var_def.ode_t0 = None
            _fsm_var_vals.ode_t0 = np.array([fsm.ode_t0])
        else:
            _fsm_var_def.ode_t0 = None
            _fsm_var_vals.ode_t0 = fsm.ode_t0

        # Construct parametrized model class and return it
        fsmp = FischerModelParametrized(
            _fsm_var_def=_fsm_var_def,
            _fsm_var_vals=_fsm_var_vals,
            ode_fun=fsm.ode_fun,
            ode_dfdx=fsm.ode_dfdx,
            ode_dfdp=fsm.ode_dfdp,
            obs_fun=fsm.obs_fun,
            obs_dfdx=fsm.obs_dfdx,
            obs_dfdp=fsm.obs_dfdp,
        )
        return fsmp

    # Define properties of class such that it can be used as a parametrized FischerModel
    # Get every possible numeric quantity that is stored in the model
    @property
    def ode_y0(self) -> np.ndarray:
        return self._fsm_var_vals.ode_y0
    
    @property
    def ode_t0(self) -> float:
        return self._fsm_var_vals.ode_t0
    
    @property
    def times(self) -> np.ndarray:
        return self._fsm_var_vals.times

    @property
    def inputs(self) -> list:
        return self._fsm_var_vals.inputs

    @property
    def parameters(self) -> tuple:
        return self._fsm_var_vals.parameters

    @property
    def constants(self) -> tuple:
        return self._fsm_var_vals.constants
    
    # These methods obtain only mutable quantities
    @property
    def ode_y0_mut(self):
        if self._fsm_var_def.ode_y0 is None:
            return None
        else:
            return self._fsm_var_vals.ode_y0
    
    @property
    def ode_t0_mut(self):
        if self._fsm_var_def.ode_t0 is None:
            return None
        else:
            return self._fsm_var_vals.ode_t0
    
    @property
    def times_mut(self):
        if self._fsm_var_def.times is None:
            return None
        else:
            return self._fsm_var_vals.times
    
    @property
    def inputs_mut(self):
        ret = []
        for q_val, q in enumerate(self._fsm_var_def):
            if q is None:
                ret.append(None)
            else:
                ret.append(self._fsm_var_vals.inputs)
        return ret

    # These methods modify mutable quantities
    @ode_y0.setter
    def ode_y0(self, y0) -> None:
        for i, y in enumerate(y0):
            self._fsm_var_vals.ode_y0[i] = y
            if self._fsm_var_def.ode_y0[i] is None:
                raise AttributeError("Variable ode_y0 is not mutable!")
    
    @ode_t0.setter
    def ode_t0(self, t0) -> None:
        if type(t0) == float:
            self._fsm_var_vals.ode_t0 = np.array([t0])
        else:
            self._fsm_var_vals.ode_t0 = t0
        if self._fsm_var_def.ode_t0 is None:
            raise AttributeError("Variable ode_y0 is not mutable!")
    
    @times.setter
    def times(self, times) -> None:
        self._fsm_var_vals.times = times
        if self._fsm_var_def.times is None:
            raise AttributeError("Variable times is not mutable!")

    @inputs.setter
    def inputs(self, inputs) -> None:
        for i, q in enumerate(inputs):
            if q is not None:
                self._fsm_var_vals.inputs[i] = q
                if self._fsm_var_def.inputs[i] is None:
                    raise AttributeError("Variable inputs at index {} is not mutable!".format(i))


@dataclass
class _FischerResultSingleBase(_FischerVariablesBase):
    ode_solution: list


@dataclass
class _FischerResultSingleOptions(_FischerVariablesOptions):
    pass


@dataclass
class FischerResultSingle(_FischerResultSingleOptions, _FischerResultSingleBase):
    pass


@dataclass
class _FischerResultsBase(_FischerOdeFunctions):
    criterion: float
    S: np.ndarray
    C: np.ndarray
    individual_results: list
    _fsm_var_def: FischerVariables
    

@dataclass
class _FischerResultsOptions(_FischerObservableFunctionsOptional):
    pass


class FischerResults(_FischerResultsOptions, _FischerResultsBase):
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
