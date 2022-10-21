import numpy as np
from dataclasses import dataclass


@dataclass
class VariableDefinition:
    lb: float
    ub: float
    n: int
    discrete: list = None
    min_distance = None
    initial_guess = "uniform"
    unique = False

    def __post_init__(self):
        if self.initial_guess == "uniform":
            if self.discrete == None:
                self.initial_guess = np.linspace(self.lb, self.ub, self.n)
            elif type(self.discrete) == float:
                if self.lb + self.n * self.discrete > self.ub:
                    raise ValueError("Too many steps ({}) in interval [{}, {}] with discretization {}".format(self.n, self.ub, self.lb, self.discrete))
                uniform_dist = (self.ub - self.lb - (self.n - 1) * self.discrete) / 2
                self.initial_guess = self.lb + uniform_dist + np.arange(self.n) * self.discrete
            else:
                self.initial_guess = discrete[0:self.n]
        elif type(self.initial_guess)!=np.ndarray:
            raise ValueError("Unknown method to sample initial guess")
        self.bounds = (self.lb, self.ub)
