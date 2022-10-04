#!/usr/bin/env python3

import numpy as np
import itertools as iter
import multiprocessing as mp
import time

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom functions for optimization
from src.solving import fischer_determinant, calculate_fischer_observable
from src.database import generate_new_collection, insert_fischer_dataclasses, drop_all_collections
from src.data_structures import FischerModel


# System of equation for pool-model and sensitivities
def pool_model_sensitivity(y, t, Q, P, Const):
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    (n, sa, sb, sc) = y
    return [
        (a*Temp + c) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max),
        (  Temp    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sa,
        (a*Temp + c) * (    n0*t*Temp * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sb,
        (     1    ) * (n -        n0 * np.exp(-b*Temp*t))*(1-n/n_max) + (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) * sc
    ]


def jacobi(y, t, Q, P, Const):
    (n, sa, sb, sc) = y
    (a, b, c) = P
    (Temp,) = Q
    (n0, n_max) = Const
    dfdn = (a*Temp + c) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t))
    return np.array([
        [   dfdn,                                                                                             0,    0,    0   ],
        [(  Temp    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sa, dfdn, 0,    0   ],
        [(a*Temp + c) * (  -  n0/n_max * t * Temp * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sb, 0,    dfdn, 0   ],
        [(     1    ) * (1 - 2*n/n_max + n0/n_max * np.exp(-b*Temp*t)) + (a*Temp + c) * (1 - 2 / n_max) * sc, 0,    0,    dfdn]
    ])


def sorting_key(x):
    '''Contents of x are typically results of calculate_Fischer_determinant (see above)
    Thus x = (obs, times, P, Q_arr, Const, Y0)'''
    #norm = max(x[1].size, 1.0)**0.5
    norm = len(x[1].parameters) * x[1].times.size
    # norm = 1.0
    seperate_times = 1.0
    for t in x[1].times:
        if len(np.unique(t)) != len(t) or len(np.unique(x[1].q_values[0])) != len(x[1].q_values[0]):
            seperate_times = 0.0
    return x[0] * seperate_times /norm


def factorize_reduced(M):
    res = []
    for i in range(2, M):
        if (M % i == 0):
            res.append((i, round(M/i)))
    return res


def get_best_fischer_results(n_time_temp, fischer_results, sorting_key, N_best):
    (n_times, n_temp) = n_time_temp
    # TODO use partial sort or some other efficient alrogithm to obtain O(n) scaling behvaiour
    # for best result retrieval
    return sorted(filter(lambda x: x[1].times.shape[-1]==n_times and len(x[1].q_values[0])==n_temp, fischer_results), key=sorting_key, reverse=True)[:N_best]


def get_new_combinations_from_best(best, N_spawn, temp_low, temp_high, dtemp, times_low, times_high, dtimes):
    combinations = []
    for (det, fsm, S, C, r) in best:
        # Also depend old result in case its better
        combinations.append(fsm)
        # Now spawn new results via next neighbors of current results
        for _ in range(0, N_spawn):
            temps_new = np.array(
                [np.random.choice([max(temp_low, T-dtemp), T, min(temp_high, T+dtemp)]) for T in fsm.q_values[0]]
            )
            times_new = np.array(
                [
                    np.sort(np.array([np.random.choice(
                        [max(times_low, t-dtimes), t, min(times_high, t+dtimes)]
                    ) for t in fsm.times[i]]))
                    for i in range(len(fsm.q_values[0]))
                ]
            )
            fsm2 = fsm
            fsm2.times = times_new
            fsm2.q_values = [temps_new]
            combinations.append(fsm2)
    return combinations


if __name__ == "__main__":
    # Define constants for the simulation duration
    n0 = 0.25
    n_max = 2e4
    effort_low = 2
    effort = 11
    effort_max = 20
    Const = (n0, n_max)

    # Define initial parameter guesses
    a = 0.065
    b = 0.01
    c = 1.31

    #2nd choice of parameters:
    #a = 0.0673
    #b = 0.01
    #c = 1.314

    P = (a, b, c)

    # Initial values for complete ODE (with S-Terms)
    y0 = [n0, 0, 0, 0]

    # Define bounds for sampling
    temp_low = 2.0
    temp_high = 16.0
    dtemp = 1.0
    n_temp_max = int((temp_high - temp_low) / dtemp + 1) # effort+1
    temp_total = np.linspace(temp_low, temp_low + dtemp * (n_temp_max - 1) , n_temp_max)

    times_low = 0.0
    times_high = 15.0
    dtimes = 1.0
    n_times_max = int((times_high-times_low) / dtimes + 1) # effort+1
    times_total = np.linspace(times_low, times_low + dtimes * (n_times_max - 1), n_times_max)

    # Initial conditions with initial time
    y0_t0 = (y0, times_low)

    # How often should we choose a sample with same number of temperatures and times
    N_mult = 3
    # How many optimization runs should we do
    N_opt = 2
    # How many best results should be propagated forward?
    N_best = 2
    # How many new combinations should an old result spawn?
    N_spawn = 3
    # How many processes will be run in parallel
    N_parallel = 48

    # Begin sampling of time and temperature values
    combinations = []

    # Iterate over every combination for total effort eg. for effort=16 we get combinations (2,8), (4,4), (8,2)
    # We exclude the "boundary" cases of (1,16) and (16,1)
    # Generate pairs of temperature and time and put everything in a large list
    for _ in range(N_mult):
        # Sample only over combinatins of both
        # for (n_temp, n_times) in factorize_reduced(effort):
        for (n_times, n_temp) in iter.product(range(effort_low, min(effort, n_times_max - 2)), range(effort_low, min(effort, n_temp_max - 2))):
            temperatures = np.random.choice(temp_total, n_temp, replace=False)
            #temperatures = np.linspace(temp_low, temp_low + dtemp * (n_temp - 1) , n_temp)
            times = np.array([np.sort(np.random.choice(times_total, n_times, replace=False)) for _ in range(len(temperatures))])
            fsm = FischerModel(
                observable=fischer_determinant,
                times=times,
                parameters=P,
                q_values=[temperatures],
                constants=Const,
                y0_t0=(y0, times_low),
                ode_func=pool_model_sensitivity,
                jacobian=jacobi
            )
            combinations.append(fsm)

    # Create pool we will later use
    p = mp.Pool(N_parallel)

    # Begin optimization scheme
    start_time = time.time()
    print_line = "[Time: {:> 8.3f} Run: {:> " +str(len(str(N_opt))) + "}] Optimizing"
    for opt_run in range(0, N_opt):
        print(print_line.format(time.time()-start_time, opt_run+1), end="\r")
        # Calculate new results
        # fischer_results will have entries of the form
        # (obs, times, P, Q_arr, Const, Y0)
        fischer_results = p.map(calculate_fischer_observable, combinations)

        # Do not optimize further if we are in the last run
        if opt_run != N_opt-1:
            # Delete old combinations
            combinations.clear()
            # Choose the N_best results with the largest objective function value from fischer_results
            fisses = p.starmap(get_best_fischer_results, zip(
                    iter.product(range(effort_low, min(effort, n_times_max - 2)), range(effort_low, min(effort, n_temp_max - 2))),
                    iter.repeat(fischer_results),
                    iter.repeat(sorting_key),
                    iter.repeat(N_best)
            ), chunksize=100)
            # Calculate new combinations parallelized
            combinations = p.starmap(get_new_combinations_from_best, zip(
                fisses,
                iter.repeat(N_spawn),
                iter.repeat(temp_low),
                iter.repeat(temp_high),
                iter.repeat(dtemp),
                iter.repeat(times_low),
                iter.repeat(times_high),
                iter.repeat(dtimes)
            ))
            combinations = [x for comb_list in combinations for x in comb_list]
    # Choose 1 best result for each (n_times, n_temp) combination
    fisses = p.starmap(get_best_fischer_results, zip(
        iter.product(range(effort_low, min(effort, n_times_max - 2)), range(effort_low, min(effort, n_temp_max - 2))),
        iter.repeat(fischer_results),
        iter.repeat(sorting_key),
        iter.repeat(1)
    ), chunksize=100)
    print(print_line.format(time.time()-start_time, opt_run+1), "done")

    # Database part
    coll = generate_new_collection("pool_model_random_grid_determinant_div_m")
    insert_fischer_dataclasses([f[0][1] for f in fisses], coll)
