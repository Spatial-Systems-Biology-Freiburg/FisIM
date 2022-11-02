import numpy as np
from pydantic.dataclasses import dataclass

from typing import Union, List, Any


class Config:
    arbitrary_types_allowed = True


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
            self.discrete = np.array(self.discrete)
        
        # Check if we want to specify more values than possible given the range with discretization)
        if self.unique==True and self.discrete!=None:
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
                    self.initial_guess = []
                    for i in range(self.n):
                        self.initial_guess.append(self.discrete[i % len(self.discrete)])
                    self.initial_guess = np.array(self.initial_guess)
                else:
                    n_low = round((len(self.discrete)- self.n)/2)
                    self.initial_guess = self.discrete[n_low:n_low+self.n]
        elif type(self.initial_guess)==np.nparray:
            self.initial_guess = np.sort(self.initial_guess, axis=-1)
        elif type(self.initial_guess)!=np.ndarray:
            raise ValueError("Unknown input {}: Either specify list of values, numpy ndarray or method to obtain initial guess.".format(self.initial_guess))
        self.bounds = (self.lb, self.ub)
