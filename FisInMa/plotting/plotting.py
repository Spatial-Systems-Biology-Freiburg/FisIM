import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pathlib import Path
import itertools

from FisInMa.model import FisherResults
from FisInMa.solving import ode_rhs


def plot_all_odes(fsr: FisherResults, outdir=Path(".")):
    for i, sol in enumerate(fsr.individual_results):
        # Get ODE solutions
        r = sol.ode_solution

        n_x = len(sol.ode_x0)

        # Get time interval over which to plot
        times_low = sol.ode_t0
        times_high = fsr.variable_definitions.times.ub if fsr.variable_definitions.times is not None else np.max(sol.times)
        
        # Solve the ODE on a sufficiently filled interval
        t_values = np.linspace(times_low, times_high)
        res = sp.integrate.solve_ivp(fsr.ode_fun, (times_low, times_high), sol.ode_x0, t_eval=t_values, args=(sol.inputs, sol.parameters, sol.ode_args))
        t = np.array(res.t)
        y = np.array(res.y)

        # Plot the solution and store in individual files
        for j in range(n_x):
            # Create figures and axis
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the continuous ODE solution
            ax.plot(t, y[j], color="#21918c", label="Ode Solution")
            
            # Determine where multiple time points overlap by rounding
            ax.scatter(sol.ode_solution.t, sol.ode_solution.y[j], s=160, alpha=0.5, color="#440154", label="Q_values: " + str(sol.inputs))
            ax.legend()
            fig.savefig(outdir / Path("ODE_Result_{}_{:03.0f}_x_{:02.0f}.svg".format(fsr.ode_fun.__name__, i, j)))

            # Remove figure to free space
            plt.close(fig)


def plot_all_sensitivities(fsr: FisherResults, outdir=Path(".")):
    for i, sol in enumerate(fsr.individual_results):
        # Get time interval over which to plot
        times_low = sol.ode_t0
        times_high = fsr.variable_definitions.times.ub if fsr.variable_definitions.times is not None else np.max(sol.times)

        # Plot solution to sensitivities
        t_values = np.linspace(times_low, times_high)
        n_x = len(sol.ode_x0)
        n_p = len(sol.parameters)
        x0_full = np.concatenate((sol.ode_x0, np.zeros(n_x * n_p)))
        res = sp.integrate.solve_ivp(ode_rhs, (times_low, times_high), x0_full, t_eval=t_values, args=(fsr.ode_fun, fsr.ode_dfdx, fsr.ode_dfdp, sol.inputs, sol.parameters, sol.ode_args, n_x, n_p))
        t = np.array(res.t)

        # Iterate over all possible sensitivities
        for j, k in itertools.product(range(n_x), range(n_p)):
            # Get ODE solutions
            if fsr.relative_sensitivities==False:
                norm1 = 1.0
                norm2 = 1.0
            else:
                norm1 = sol.ode_solution.y[:n_x].reshape((n_x, -1))[j]
                norm2 = np.array(res.y[:n_x].reshape((n_x, -1))[j])

            r = sol.ode_solution.y[n_x:].reshape((n_x, n_p, -1))[j, k] / norm1
            

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            y = np.array(res.y[n_x:].reshape((n_x, n_p, -1))[j, k]) / norm2
            ax.plot(t, y, color="#21918c", label="Sensitivities Solution")

            # Plot sampled time points
            ax.scatter(sol.ode_solution.t, r, s=160, alpha=0.5, color="#440154", label="Q_values: " + str(sol.inputs))
            ax.legend()
            fig.savefig(outdir / Path("Sensitivities_Results_{}_{:03.0f}_x_{:02.0f}_p_{:02.0f}.svg".format(fsr.ode_fun.__name__, i, j, k)))

            # Remove figure to free space
            plt.close(fig)


def plot_all_solutions(fsr: FisherResults, outdir=Path(".")):
    plot_all_odes(fsr, outdir)
    plot_all_sensitivities(fsr, outdir)
