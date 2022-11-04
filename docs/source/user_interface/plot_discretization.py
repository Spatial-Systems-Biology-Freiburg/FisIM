import matplotlib.pyplot as plt
import numpy as np

from FisInMa.optimization import discrete_penalty_calculator_default

def plot_default_discretization(outdir):
    x_discr = np.linspace(0, 10, 5)
    x_values = np.linspace(0, 10)

    y = [discrete_penalty_calculator_default([val], x_discr) for val in x_values]
    plt.plot(x_values, y)
    plt.savefig(outdir / "discretization.png")
