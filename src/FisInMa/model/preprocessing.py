import numpy as np
from dataclasses import dataclass


@dataclass
class VariableDefinition:
    lb: float
    ub: float
    n: int
    dx: float = None
    initial_guess = "uniform"

    def __post_init__(self):
        if self.initial_guess == "uniform":
            if self.dx == None:
                self.initial_guess = np.linspace(self.lb, self.ub, self.n)
            elif self.lb + self.n * self.dx > self.ub:
                raise ValueError("Too many steps ({}) in interval [{}, {}] with discretization {}".format(self.n, self.ub, self.lb, self.dx))
            else:
                uniform_dist = (self.ub - self.lb - (self.n - 1) * self.dx) / 2
                self.initial_guess = self.lb + uniform_dist + np.arange(self.n) * self.dx
        else:
            raise ValueError("Unknown method to sample initial guess")
        self.bounds = (self.lb, self.ub)
