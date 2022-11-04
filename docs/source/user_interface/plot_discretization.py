#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.getcwd())

from FisInMa.optimization import discrete_penalty_calculator_default

if __name__ == "__main__":
    x_discr = np.linspace(0, 10, 5)
    x_values = np.linspace(0, 10)

    y = [discrete_penalty_calculator_default([val], x_discr) for val in x_values]
    plt.plot(x_values, y)
    plt.savefig("./docs/source/user_interface/discretization.png")
