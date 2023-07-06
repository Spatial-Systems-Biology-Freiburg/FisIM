import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import root_validator
from typing import Union, List, Any
import itertools


class Config:
    arbitrary_types_allowed = True
    smart_union = True


@dataclass(config=Config)
class VariableDefinition():
    lb: float
    ub: float
    n: int
    discrete: Union[float, List[float], np.ndarray] = None
    min_distance: float = None
    initial_guess = "uniform"
    unique: bool = False

    def __post_init__(self):
        # Create the discretization either from float or list
        if type(self.discrete) == float:
            self.discrete = np.arange(self.lb, self.ub + self.discrete/2, self.discrete)
        elif type(self.discrete) == list:
            # TODO test this statement
            self.discrete = np.array(self.discrete)
        
        # Check if we want to specify more values than possible given the range with discretization)
        if self.unique==True and self.discrete!=None:
            # TODO test this statement
            if self.n > len(self.discrete):
                raise ValueError("Too many steps ({}) in interval [{}, {}] with discretization {}".format(self.n, self.ub, self.lb, self.discrete))

        # Define initial guess for variable
        if self.initial_guess == "uniform":
            if self.discrete is None:
                self.initial_guess = np.linspace(self.lb, self.ub, self.n)
            else:
                # If we sample more points than we have discrete values,
                # we simply iterate over all of them and fill the 
                # initial values this way. Afterwards we will sort them.
                if self.n >= len(self.discrete):
                    # TODO test this statement
                    self.initial_guess = []
                    for i in range(self.n):
                        self.initial_guess.append(self.discrete[i % len(self.discrete)])
                    self.initial_guess = np.array(self.initial_guess)
                else:
                    n_low = round((len(self.discrete)- self.n)/2)
                    self.initial_guess = self.discrete[n_low:n_low+self.n]
            self.initial_guess = np.array(self.initial_guess)
        elif type(self.initial_guess)==np.nparray:
            # TODO test this statement
            self.initial_guess = np.sort(self.initial_guess, axis=-1)
        elif type(self.initial_guess)!=np.ndarray:
            raise ValueError("Unknown input {}: Either specify list of values, numpy ndarray or method to obtain initial guess.".format(self.initial_guess))
        self.bounds = (self.lb, self.ub)


@dataclass(config=Config)
class CovarianceDefinition():
    rel: Union[np.ndarray, List[float], float] = None
    abs: Union[np.ndarray, List[float], float] = None


@dataclass(config=Config)
class MultiVariableDefinition():
    lb: Union[float, List[float]]
    ub: Union[float, List[float]]
    n: int
    discrete: Union[float, List[float], List[np.ndarray]] = None
    min_distance: List[float] = None
    initial_guess = "uniform"
    unique: bool = False

    # TODO test these statements
    @root_validator(pre=True)
    def check_dimensions(cls, values):
        lb = values.get('lb')
        ub = values.get('ub')
        discrete = values.get('discrete')
        unique = values.get('unique')
        n = values.get('n')

        # Check lower and upper bounds
        if type(lb)!=type(ub):
            raise ValueError("lb and ub need to have same types")
        if type(lb)==type(ub)==list:
            if len(lb) != len(ub):
                raise ValueError("lb and ub need to have sime size")
        if type(discrete)==list:
            if type(discrete[0])==float:
                if len(discrete) != len(lb):
                    raise ValueError("discretization needs to be of same size as bounds")
            if type(discrete[0])==list:
                for d in discrete:
                    if len(d)!=len(lb):
                        raise ValueError("Every discretization vector needs to have the same size as bounds! Failing vector: {}".format(d))

        # Discrete needs to be a positive float
        if type(discrete)==float and discrete < 0.0:
            raise ValueError("non-positive numbers not accepted for discretization")

        # Check that discretization has correct form
        if type(discrete) == list and type(discrete[0]) == np.ndarray:
            for i in range(len(lb)):
                if discrete[i].ndim!=1:
                    raise ValueError("Discretization needs to have dimension 1")
            try:
                np.array(discrete)
            except:
                raise ValueError("All entries for discretization need to have the same length")

        # Check if we want to specify more values than possible given the range with discretization)
        if unique==True and discrete!=None:
            if n > len(discrete):
                raise ValueError("Too many steps ({}) in interval [{}, {}] with discretization {}".format(n, ub, lb, discrete))
        return values

    # TODO test these statements
    def __post_init__(self):
        # If we only have a float that was specified, then make it a list
        if type(self.lb) == float:
            self.lb = [self.lb]
        if type(self.ub) == float:
            self.ub = [self.ub]

        # Check that discrete is not an int
        if type(self.discrete)==int:
            self.discrete = float(self.discrete)

        # Create the discretization either from float or list
        if type(self.discrete) == float:
            # Generate all possible discretization vectors
            disc_pre = [np.arange(self.lb[i], self.ub[i] + self.discrete/2, self.discrete) for i in range(len(self.lb))]
            self.discrete = [np.array(a) for a in itertools.product(*disc_pre)]
            # self.discrete = [np.arange(self.lb[i], self.ub[i] + self.discrete/2, self.discrete) for i in range(len(self.lb))]
        elif type(self.discrete) == list and type(self.discrete[0]) == float:
            # Create discretizations for the individual dimensions
            self.discrete = [np.array(self.discrete) for _ in range(len(self.lb))]

        # Define initial guess for variable
        if self.initial_guess == "uniform":
            if self.discrete is None:
                self.initial_guess = np.array([d for d in np.array([np.linspace(self.lb[i], self.ub[i], self.n) for i in range(len(self.lb))]).T])
            else:
                # If we sample more points than we have discrete values,
                # we simply iterate over all of them and fill the
                # initial values this way. Afterwards we will sort them.
                if self.n >= len(self.discrete):
                    self.initial_guess = []
                    for i in range(self.n):
                        self.initial_guess.append(self.discrete[i % len(self.discrete)])
                else:
                    m = int(len(self.discrete)/self.n)
                    m_start = int((len(self.discrete) % m)/2)
                    self.initial_guess = self.discrete[m_start:-1:m]
        else:
            raise ValueError("Unknown input {}: Either specify list of values, numpy ndarray or method to obtain initial guess.".format(self.initial_guess))
        self.bounds = (self.lb, self.ub)
