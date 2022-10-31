import numpy as np
# from dataclasses import dataclass
from copy import deepcopy
from pydantic.dataclasses import dataclass
try:
    from collections.abc import Callable
except:
    from typing import Callable
from typing import Optional, Union, Any, List

from .preprocessing import VariableDefinition


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class _FisherVariablesBase:
    ode_x0: Union[np.ndarray, List[float], List[List[float]], float, List[np.ndarray]]
    ode_t0: Union[tuple, float, np.ndarray, List]
    times: Union[tuple, List[float], List[List[float]], np.ndarray]
    inputs: List# list[Union[list[float],np.ndarray]]
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
    variable_values: FisherVariables


@dataclass(config=Config)
class _FisherModelParametrizedOptions(_FisherModelOptions):
    pass


@dataclass(config=Config)
class FisherModelParametrized(_FisherModelParametrizedOptions, _FisherModelParametrizedBase):
    def init_from(fsm: FisherModel):
        """Initialize a parametrized FisherModel with initial guesses for the sampled variables.

        :param fsm: A user-defined fisher model.
        :type fsm: FisherModel
        :raises TypeError: Currently does not accept sampling over initial values ode_x0.
        :return: Fully parametrized model with initial guesses which can be numerically solved.
        :rtype: FisherModelParametrized
        """
        # Create distinct classes to store
        # 1) Initial definition of model (ie. sample over certain variable; specify tuple of (min, max, n, dx, guess_method) or explicitly via np.array([...]))
        # 2) Explicit values together with initial guess such that every variable is parametrized
        variable_definitions = FisherVariables(
            fsm.ode_x0,
            fsm.ode_t0,
            fsm.times,
            fsm.inputs,
            fsm.parameters,
            fsm.ode_args,
            fsm.identical_times,
        )
        variable_values = deepcopy(variable_definitions)

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
        variable_values.inputs = _inputs_vals
        inputs_shape = tuple(len(q) for q in _inputs_vals)

        # Check if we want to sample over initial values
        if type(fsm.ode_x0) is float:
            x0_def = None
            x0_vals = [np.array([fsm.ode_x0])]
        elif type(fsm.ode_x0) is np.ndarray and fsm.ode_x0.ndim == 1:
            x0_def = None
            x0_vals = [fsm.ode_x0]
        elif type(fsm.ode_x0) is np.ndarray and fsm.ode_x0.ndim > 1:
            raise TypeError("Variable ode_x0 should be list of arrays with dimension 1 respectively!")
        # TODO currently not working
        elif type(fsm.ode_x0) is tuple and len(fsm.ode_x0) >= 3:
            x0 = VariableDefinition(*fsm.ode_x0)
            x0_def = x0
            x0_vals = [x0.initial_guess]
            raise TypeError("Warning! Specifying initial values as tuple enables sampling over initial values. This is currently not implemented!")
        elif type(fsm.ode_x0) is list and len(fsm.ode_x0) > 0 and type(fsm.ode_x0[0]) is list and len(fsm.ode_x0[0]) > 0 and type(fsm.ode_x0[0][0]) is float:
            x0_def = None
            x0_vals = [np.array(xi) for xi in fsm.ode_x0]
        elif type(fsm.ode_x0) is list and len(fsm.ode_x0) > 0 and type(fsm.ode_x0[0])==np.ndarray:
            x0_def = None
            x0_vals = fsm.ode_x0
        elif type(fsm.ode_x0) is list and len(fsm.ode_x0) > 0 and type(fsm.ode_x0[0]==float):
            x0_def = None
            x0_vals = [np.array(fsm.ode_x0)]
        else:
            x0_def = None
            x0_vals = np.array(fsm.ode_x0)

        variable_definitions.ode_x0 = x0_def
        variable_values.ode_x0 = x0_vals

        # Check if time values are sampled
        if type(fsm.times) == tuple and len(fsm.times) >= 3:
            t = VariableDefinition(*fsm.times)
            variable_definitions.times = t
            variable_values.times = t.initial_guess
        elif type(fsm.times)==list:
            variable_definitions.times = None
            variable_values.times = np.array(fsm.times)
        else:
            variable_definitions.times = None
            variable_values.times = np.array(fsm.times)
        # If non-identical times were chosen, expand initial guess to full array
        if fsm.identical_times==False:
            variable_values.times = np.full(inputs_shape + variable_values.times.shape, variable_values.times)

        # Check if we want to sample over initial time
        if type(fsm.ode_t0) == tuple and len(fsm.ode_t0) >= 3:
            t0 = VariableDefinition(*fsm.ode_t0)
            variable_definitions.ode_t0 = t0
            variable_values.ode_t0 = t0.initial_guess
        elif type(fsm.ode_t0) == float:
            variable_definitions.ode_t0 = None
            variable_values.ode_t0 = np.array([fsm.ode_t0])
        else:
            variable_definitions.ode_t0 = None
            variable_values.ode_t0 = np.array(fsm.ode_t0)

        # Construct parametrized model class and return it
        fsmp = FisherModelParametrized(
            variable_definitions=variable_definitions,
            variable_values=variable_values,
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
    def ode_x0(self) -> np.ndarray:
        return self.variable_values.ode_x0
    
    @property
    def ode_t0(self) -> float:
        return self.variable_values.ode_t0
    
    @property
    def times(self) -> np.ndarray:
        return self.variable_values.times

    @property
    def inputs(self) -> list:
        return self.variable_values.inputs

    @property
    def parameters(self) -> tuple:
        return self.variable_values.parameters

    @property
    def ode_args(self) -> tuple:
        return self.variable_values.ode_args
    
    # These methods obtain only mutable quantities.
    # Return None or a list of None and values depending on which quantity is mutable
    @property
    def ode_x0_mut(self):
        if self.variable_definitions.ode_x0 is None:
            return None
        else:
            return self.variable_values.ode_x0
    
    @property
    def ode_t0_mut(self):
        if self.variable_definitions.ode_t0 is None:
            return None
        else:
            return self.variable_values.ode_t0
    
    @property
    def times_mut(self):
        if self.variable_definitions.times is None:
            return None
        else:
            return self.variable_values.times
    
    @property
    def inputs_mut(self):
        ret = []
        for q_val, q in zip(self.variable_values.inputs, self.variable_definitions.inputs):
            if q is None:
                ret.append(None)
            else:
                ret.append(q_val)
        return ret

    # These methods return the definition or None if the values were picked by hand
    @property
    def ode_x0_def(self):
        return self.variable_definitions.ode_x0
    
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
    @ode_x0.setter
    def ode_x0(self, x0) -> None:
        for i, y in enumerate(x0):
            self.variable_values.ode_x0[i] = y
            if self.variable_definitions.ode_x0[i] is None:
                raise AttributeError("Variable ode_x0 is not mutable!")
    
    @ode_t0.setter
    def ode_t0(self, t0) -> None:
        if type(t0) == float:
            self.variable_values.ode_t0 = np.array([t0])
        else:
            self.variable_values.ode_t0 = t0
        if self.variable_definitions.ode_t0 is None:
            raise AttributeError("Variable ode_x0 is not mutable!")
    
    @times.setter
    def times(self, times) -> None:
        self.variable_values.times = times
        if self.variable_definitions.times is None:
            raise AttributeError("Variable times is not mutable!")

    @inputs.setter
    def inputs(self, inputs) -> None:
        for i, q in enumerate(inputs):
            if q is not None:
                self.variable_values.inputs[i] = q
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
    relative_sensitivities: bool
    

@dataclass(config=Config)
class _FisherResultsOptions(_FisherModelOptions):
    pass


@dataclass(config=Config)
class FisherResults(_FisherResultsOptions, _FisherResultsBase):
    pass
