import numpy as np
# from dataclasses import dataclass
from copy import deepcopy
from pydantic.dataclasses import dataclass
try:
    from collections.abc import Callable
except:
    from typing import Callable
from typing import Optional, Union, Any

from .preprocessing import VariableDefinition


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class _FisherVariablesBase:
    ode_y0: Union[np.ndarray, list[float], list[list[float]], float, list[np.ndarray]]
    ode_t0: Union[tuple, float, np.ndarray, list]
    times: Union[tuple, list[float], list[list[float]], np.ndarray]
    inputs: list# list[Union[list[float],np.ndarray]]
    parameters: Union[tuple,list]
    ode_args: Optional[Any] 


@dataclass(config=Config)
class _FisherVariablesOptions:
    identical_times: bool = False


@dataclass(config=Config)
class _FisherOdeFunctions:
    ode_fun: Callable
    ode_dfdx: Callable
    ode_dfdp: Callable


@dataclass(config=Config)
class _FisherObservableFunctionsOptional:
    obs_fun: Optional[Callable] = None
    obs_dfdx: Optional[Callable] = None
    obs_dfdp: Optional[Callable] = None


@dataclass(config=Config)
class FisherVariables(_FisherVariablesOptions, _FisherVariablesBase):
    pass


@dataclass(config=Config)
class _FisherModelBase(_FisherOdeFunctions, _FisherVariablesBase):
    pass


@dataclass(config=Config)
class _FisherModelOptions(_FisherVariablesOptions, _FisherObservableFunctionsOptional):
    pass


@dataclass(config=Config)
class FisherModel(_FisherModelOptions, _FisherModelBase):
    pass


@dataclass(config=Config)
class _FisherModelParametrizedBase(_FisherOdeFunctions):
    variable_definitions: FisherVariables
    _fsm_var_vals: FisherVariables


@dataclass(config=Config)
class _FisherModelParametrizedOptions(_FisherModelOptions):
    pass


@dataclass(config=Config)
class FisherModelParametrized(_FisherModelParametrizedOptions, _FisherModelParametrizedBase):
    def init_from(fsm: FisherModel):
        """Initialize a parametrized FisherModel with initial guesses for the sampled variables.

        :param fsm: A user-defined fisher model.
        :type fsm: FisherModel
        :raises TypeError: Currently does not accept sampling over initial values ode_y0.
        :return: Fully parametrized model with initial guesses which can be numerically solved.
        :rtype: FisherModelParametrized
        """
        # Create distinct classes to store
        # 1) Initial definition of model (ie. sample over certain variable; specify tuple of (min, max, n, dx, guess_method) or explicitly via np.array([...]))
        # 2) Explicit values together with initial guess such that every variable is parametrized
        variable_definitions = FisherVariables(
            fsm.ode_y0,
            fsm.ode_t0,
            fsm.times,
            fsm.inputs,
            fsm.parameters,
            fsm.ode_args,
            fsm.identical_times,
        )
        _fsm_var_vals = deepcopy(variable_definitions)

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
                _inputs_vals.append(np.array(q))
        
        variable_definitions.inputs = _inputs_def
        _fsm_var_vals.inputs = _inputs_vals
        inputs_shape = tuple(len(q) for q in _inputs_vals)

        # Check if we want to sample over initial values
        if type(fsm.ode_y0)==float:
            y0_def = None
            y0_vals = [np.array([fsm.ode_y0])]
        elif type(fsm.ode_y0)==np.ndarray and fsm.ode_y0.ndim == 1:
            y0_def = None
            y0_vals = [fsm.ode_y0]
        # TODO currently not working
        elif type(fsm.ode_y0)==tuple and len(fsm.ode_y0)>=3:
            y0 = VariableDefinition(*fsm.ode_y0)
            y0_def = y0
            y0_vals = [y0.initial_guess]
            raise TypeError("Warning! Specifying initial values as tuple enables sampling over initial values. This is currently not implemented!")
        elif type(fsm.ode_y0)==list and np.array(fsm.ode_y0).ndim == 1:
            y0_def = None
            y0_vals = [np.array(fsm.ode_y0)]
            print("Got here!", fsm.ode_y0)
        elif type(fsm.ode_y0)==list[list[float]]:
            y0_def = None
            y0_vals = np.array(fsm.ode_y0)
            print("Got here 2!", fsm.ode_y0)
        else:
            y0_def = None
            y0_vals = np.array(fsm.ode_y0)

        variable_definitions.ode_y0 = y0_def
        _fsm_var_vals.ode_y0 = y0_vals

        # Check if time values are sampled
        if type(fsm.times) == tuple and len(fsm.times) >= 3:
            t = VariableDefinition(*fsm.times)
            variable_definitions.times = t
            _fsm_var_vals.times = t.initial_guess
        elif type(fsm.times)==list:
            variable_definitions.times = None
            _fsm_var_vals.times = np.array(fsm.times)
        else:
            variable_definitions.times = None
            _fsm_var_vals.times = np.array(fsm.times)
        # If non-identical times were chosen, expand initial guess to full array
        if fsm.identical_times==False:
            _fsm_var_vals.times = np.full(inputs_shape + _fsm_var_vals.times.shape, _fsm_var_vals.times)

        # Check if we want to sample over initial time
        if type(fsm.ode_t0) == tuple and len(fsm.ode_t0) >= 3:
            t0 = VariableDefinition(*fsm.ode_t0)
            variable_definitions.ode_t0 = t0
            _fsm_var_vals.ode_t0 = t0.initial_guess
        elif type(fsm.ode_t0) == float:
            variable_definitions.ode_t0 = None
            _fsm_var_vals.ode_t0 = np.array([fsm.ode_t0])
        else:
            variable_definitions.ode_t0 = None
            _fsm_var_vals.ode_t0 = np.array(fsm.ode_t0)

        # Construct parametrized model class and return it
        fsmp = FisherModelParametrized(
            variable_definitions=variable_definitions,
            _fsm_var_vals=_fsm_var_vals,
            ode_fun=fsm.ode_fun,
            ode_dfdx=fsm.ode_dfdx,
            ode_dfdp=fsm.ode_dfdp,
            obs_fun=fsm.obs_fun,
            obs_dfdx=fsm.obs_dfdx,
            obs_dfdp=fsm.obs_dfdp,
            identical_times=fsm.identical_times,
        )
        return fsmp

    # Define properties of class such that it can be used as a parametrized FisherModel
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
    def ode_args(self) -> tuple:
        return self._fsm_var_vals.ode_args
    
    # These methods obtain only mutable quantities.
    # Return None or a list of None and values depending on which quantity is mutable
    @property
    def ode_y0_mut(self):
        if self.variable_definitions.ode_y0 is None:
            return None
        else:
            return self._fsm_var_vals.ode_y0
    
    @property
    def ode_t0_mut(self):
        if self.variable_definitions.ode_t0 is None:
            return None
        else:
            return self._fsm_var_vals.ode_t0
    
    @property
    def times_mut(self):
        if self.variable_definitions.times is None:
            return None
        else:
            return self._fsm_var_vals.times
    
    @property
    def inputs_mut(self):
        ret = []
        for q_val, q in zip(self._fsm_var_vals.inputs, self.variable_definitions.inputs):
            if q is None:
                ret.append(None)
            else:
                ret.append(q_val)
        return ret

    # These methods return the definition or None if the values were picked by hand
    @property
    def ode_y0_def(self):
        return self.variable_definitions.ode_y0
    
    @property
    def ode_t0_def(self):
        return self.variable_definitions.ode_t0
    
    @property
    def times_def(self):
        return self.variable_definitions.times

    @property
    def inputs_def(self):
        return self.variable_definitions.inputs

    # These methods modify mutable quantities
    @ode_y0.setter
    def ode_y0(self, y0) -> None:
        for i, y in enumerate(y0):
            self._fsm_var_vals.ode_y0[i] = y
            if self.variable_definitions.ode_y0[i] is None:
                raise AttributeError("Variable ode_y0 is not mutable!")
    
    @ode_t0.setter
    def ode_t0(self, t0) -> None:
        if type(t0) == float:
            self._fsm_var_vals.ode_t0 = np.array([t0])
        else:
            self._fsm_var_vals.ode_t0 = t0
        if self.variable_definitions.ode_t0 is None:
            raise AttributeError("Variable ode_y0 is not mutable!")
    
    @times.setter
    def times(self, times) -> None:
        self._fsm_var_vals.times = times
        if self.variable_definitions.times is None:
            raise AttributeError("Variable times is not mutable!")

    @inputs.setter
    def inputs(self, inputs) -> None:
        for i, q in enumerate(inputs):
            if q is not None:
                self._fsm_var_vals.inputs[i] = q
                if self.variable_definitions.inputs[i] is None:
                    raise AttributeError("Variable inputs at index {} is not mutable!".format(i))


@dataclass(config=Config)
class _FisherResultSingleBase(_FisherVariablesBase):
    ode_solution: Any#Union[list,np.ndarray]


@dataclass(config=Config)
class _FisherResultSingleOptions(_FisherVariablesOptions):
    pass


@dataclass(config=Config)
class FisherResultSingle(_FisherResultSingleOptions, _FisherResultSingleBase):
    pass


@dataclass(config=Config)
class _FisherResultsBase(_FisherOdeFunctions):
    criterion: float
    S: np.ndarray
    C: np.ndarray
    individual_results: list
    variable_definitions: FisherVariables
    

@dataclass(config=Config)
class _FisherResultsOptions(_FisherModelOptions):
    pass


@dataclass(config=Config)
class FisherResults(_FisherResultsOptions, _FisherResultsBase):
    def to_savedict(self):
        '''Used to store results in database'''
        d = {
            "time_interval": apply_marks(self.time_interval),
            "times": apply_marks(self.times),
            "parameters": apply_marks(self.parameters),
            "q_values": apply_marks(self.q_values),
            "ode_args": apply_marks(self.ode_args),
            "y0": apply_marks(self.y0),
            "criterion": apply_marks(self.criterion),
            "criterion_func": apply_marks(self.criterion_func.__name__),
            "sensitivity_matrix": apply_marks(self.sensitivity_matrix),
            "covariance_matrix": apply_marks(self.covariance_matrix),
            "ode_solutions": apply_marks(ode_solutions)
        }
        return d
