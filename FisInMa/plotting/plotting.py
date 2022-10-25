import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pathlib import Path

from FisInMa.model import FisherResults


def plot_all_odes(fsr: FisherResults, outdir=Path(".")):
    for i, sol in enumerate(fsr.individual_results):
        # Get ODE solutions
        r = sol.ode_solution

        # Get time interval over which to plot
        times_low = sol.ode_t0
        times_high = np.max(sol.times)

        # Create figures and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot solution to ode
        t_values = np.linspace(times_low, times_high)
        res = sp.integrate.solve_ivp(fsr.ode_fun, (times_low, times_high), sol.ode_y0, t_eval=t_values, args=(sol.inputs, sol.parameters, sol.constants))
        ax.plot(res.t.reshape(res.y.shape).T, res.y.T, color="#21918c", label="Ode Solution")

        # Determine where multiple time points overlap by rounding
        ax.scatter(sol.ode_solution.t, sol.ode_solution.y[:len(sol.ode_y0)], s=160, alpha=0.5, color="#440154", label="Q_values: ")
        ax.legend()
        fig.savefig(outdir / Path("Result_{}_{:03.0f}.svg".format(fsr.ode_fun.__name__, i)))

        # Remove figure to free space
        plt.close(fig)
