import dataclasses
import pprint
import shutil
from dataclasses import fields
import numpy as np
import itertools

from FisInMa.model import FisherModelParametrized, FisherResults, FisherResultSingle


def display_heading(string, terminal_size=shutil.get_terminal_size((80, 20))):
    string = " " + string + " "
    heading_fmt = "{:=^" + str(terminal_size[0]) + "}"
    print(heading_fmt.format(string))


def display_entries(table, terminal_size, caption=None):
    # Calculate how large the column may be
    n_cols = len(table[0])
    col_sizes = [max([len(str(c[i])) for c in table]) for i in range(n_cols)]

    entry_fmt = "  ".join(["{:<" + str(k) + "}" for k in col_sizes][0:-1]) + "  {:^}"
    entry_fmt_sta = lambda row: print(("┌─" + entry_fmt).format(*[str(r) for r in row]))
    entry_fmt_sto = lambda row: print(("│ " + entry_fmt).format(*[str(r) for r in row]))
    entry_fmt_mid = lambda row: print(("├─" + entry_fmt).format(*[str(r) for r in row])) if len(str(row[0])) > 0 else entry_fmt_non(row)
    entry_fmt_non = lambda row: print(("│ " + entry_fmt).format(*[str(r) for r in row]))
    entry_fmt_end = lambda row: print(("└─" + entry_fmt).format(*[str(r) for r in row])) if len(str(row[0])) > 0 else entry_fmt_sto(row)

    if caption is not None:
        print(caption)
    if len(table) > 1:
        entry_fmt_mid(table[0])
    else:
        entry_fmt_end(table[0])
    
    for row in table[1:-1]:
        entry_fmt_mid(row)
    if len(table) > 1:
        entry_fmt_end(table[-1])


def display_fsmp_details(fsmp: FisherModelParametrized, pp=pprint.PrettyPrinter(indent=2, width=shutil.get_terminal_size((80, 20))[0]), terminal_size=shutil.get_terminal_size((80, 20))):
    display_heading("ODE FUNCTIONS", terminal_size)
    cols = [
        ("ode_fun", getattr(fsmp.ode_fun, '__name__', 'unknown')),
        ("ode_dfdx", getattr(fsmp.ode_dfdx, '__name__', 'unknown')),
        ("ode_dfdp", getattr(fsmp.ode_dfdp, '__name__', 'unknown')),
    ]
    if callable(fsmp.ode_dfdx0):
        cols.append(("ode_dfdx0", getattr(fsmp.ode_dfdx0, '__name__', 'unknown')))
    if callable(fsmp.obs_fun):
        cols.append(("obs_fun", getattr(fsmp.obs_fun, '__name__', 'unknown')))
        cols.append(("obs_dgdx", getattr(fsmp.obs_dgdx, '__name__', 'unknown')))
        cols.append(("obs_dgdp", getattr(fsmp.obs_dgdp, '__name__', 'unknown')))
    if callable(fsmp.obs_dgdx0):
        cols.append(("obs_dgdx0", getattr(fsmp.obs_dgdx0, '__name__', 'unknown')))
    display_entries(cols, terminal_size)

    display_heading("INITIAL GUESS", terminal_size)
    cols = [(field.name, getattr(fsmp.variable_values, field.name)) for field in fields(fsmp.variable_values)]
    display_entries(cols, terminal_size)
    # pp.pprint(fsmp.variable_values)


    display_heading("VARIABLE DEFINITIONS", terminal_size)
    cols = [(field.name, getattr(fsmp.variable_definitions, field.name)) for field in fields(fsmp.variable_definitions)]
    display_entries(cols, terminal_size)

    display_heading("VARIABLE VALUES", terminal_size)
    cols = [(field.name, getattr(fsmp.variable_values, field.name)) for field in fields(fsmp.variable_values)]
    display_entries(cols, terminal_size)

    display_heading("OTHER OPTIONS", terminal_size)
    cols = [
        ("identical_times", fsmp.identical_times),
    ]
    display_entries(cols, terminal_size)


def _generate_matrix_cols(M, name, terminal_size=shutil.get_terminal_size((80, 20))):
    M_strings = np.array2string(M, max_line_width=terminal_size[0]).split("\n")
    M_total = [(name, M_strings[0])]
    M_total += list(zip(itertools.repeat(""), M_strings[1:]))
    return M_total


def _generate_fsrs_cols(fsrs: FisherResultSingle, terminal_size=shutil.get_terminal_size((80, 20))):
    cols = [
        ("ode_x0", fsrs.ode_x0),
        ("ode_t0", fsrs.ode_t0)
    ]
    cols += _generate_matrix_cols(fsrs.times, "times", terminal_size)

    cols+= [
        ("inputs", fsrs.inputs),
        ("parameters", fsrs.parameters),
    ]
    return cols


def display_fsr_details(fsr: FisherResults, pp=pprint.PrettyPrinter(indent=2, width=shutil.get_terminal_size((80, 20))[0]), terminal_size=shutil.get_terminal_size((80, 20))):
    display_heading("OPTIMIZED RESULTS")

    display_heading("CRITERION")
    cols = [
        (getattr(fsr.criterion_fun, '__name__', 'unknown'), fsr.criterion),
    ]
    cols += _generate_matrix_cols(fsr.S.T, "sensitivity matrix", terminal_size)
    cols += _generate_matrix_cols(fsr.C.T, "inverse covariance matrix", terminal_size)
    display_entries(cols, terminal_size)

    display_heading("INDIVIDUAL RESULTS")

    for i in range(len(fsr.individual_results)):
        cols = _generate_fsrs_cols(fsr.individual_results[i])#) for i in range(len(fsr.individual_results))]
        cap = ("Result_{:0" + str(len(str(len(fsr.individual_results)))) + "}").format(i)
        display_entries(cols, caption=cap, terminal_size=terminal_size)


    display_heading("DISCRETIZATION PENALTY SUMMARY")
    cols = [(field.name, getattr(fsr.penalty_discrete_summary, field.name)) for field in fields(fsr.penalty_discrete_summary)]
    display_entries(cols, terminal_size)
