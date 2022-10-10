import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pathlib import Path

from FisInMa.data_structures import FischerResult


def plot_all_odes(fsr: FischerResult, outdir=Path("out")):
    times_low, times_high = fsr.time_interval
    solutions = fsr.ode_solutions

    for i, (t, q, r) in enumerate(solutions):
        # Create figures and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot solution to ode
        t_values = np.linspace(times_low, times_high)
        res = sp.integrate.odeint(fsr.ode_func, fsr.y0, t_values, args=(q, fsr.parameters, fsr.constants)).T[0]
        ax.plot(t_values, res, color="#21918c", label="Ode Solution")

        # Determine where multiple time points overlap by rounding
        t_round = t.round(1)
        unique, indices, counts = np.unique(t_round, return_index=True, return_counts=True)

        # Plot same time points with different sizes to distinguish
        ax.scatter(unique, r[0][indices], s=counts*80, alpha=0.5, color="#440154", label="Q_values: " + str(q))
        ax.legend()

        fig.savefig(outdir / Path("Result_{}_{:010.0f}.svg".format(fsr.ode_func.__name__, i)))

        # Remove figure to free space
        plt.close(fig)