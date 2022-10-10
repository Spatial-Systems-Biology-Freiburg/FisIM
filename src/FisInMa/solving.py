import numpy as np
from scipy.integrate import odeint
import itertools

from FisInMa.data_structures import FischerModelParametrized, FischerResult


def get_S_matrix(fsmp: FischerModelParametrized, relative_sensitivities=False):
    """now we calculate the derivative with respect to the parameters
    The matrix S has the form
    i   -->  index of parameter
    jk  -->  index of kth variable
    t   -->  index of time
    S[i, j1, j2, ..., t] = (dO/dp_i(v_j1, v_j2, v_j3, ..., t))"""
    S = np.zeros((len(fsmp.parameters), fsmp.times.shape[-1],) + tuple(len(x) for x in fsmp.q_values))
    error_n = np.zeros((fsmp.times.shape[-1],) + tuple(len(x) for x in fsmp.q_values))

    # Iterate over all combinations of Q-Values
    solutions = []
    for index in itertools.product(*[range(len(q)) for q in fsmp.q_values]):
        # Store the results of the respective ODE solution
        Q = [fsmp.q_values[i][j] for i, j in enumerate(index)]
        t = fsmp.times[index]
        t_init = np.insert(t, 0, fsmp.time_interval[0])

        # Actually solve the ODE for the selected parameter values
        #r = solve_ivp(ODE_func, [t0, t.max()], y0, method='Radau', t_eval=t,  args=(Q, P, Const), jac=jacobian).y.T[1:,:]
        r = odeint(fsmp.ode_func, fsmp.y0, t_init, args=(Q, fsmp.parameters, fsmp.constants), Dfun=fsmp.jacobian).T[:, 1:]

        # Calculate the S-Matrix with the supplied jacobian
        # Depending on if we want to calculate the relative sensitivities
        if relative_sensitivities==True:
            a = (r[1:]/r[0])
            for i in range(len(fsmp.parameters)):
                a[i] *= fsmp.parameters[i]
            S[(slice(None), slice(None)) + index] = a
        else:
            S[(slice(None), slice(None)) + index] = r[1:]

        # Assume that the error of the measurement is 25% from the measured value r[0] n 
        # (use for covariance matrix calculation)
        error_n[(slice(None),) + index] = r[0] * 0.25
        solutions.append((t, Q, r))
    
    # Reshape to 2D Form (len(P),:)
    S = S.reshape((len(fsmp.parameters),np.prod(S.shape[1:])))
    error_n = error_n.flatten()
    cov_matrix = np.eye(len(error_n), len(error_n)) * error_n**2
    C = np.linalg.inv(cov_matrix)
    return S, C, solutions


def fischer_determinant(fsmp: FischerModelParametrized, S, C):
    # Calculate Fisher Matrix
    F = (S.dot(C)).dot(S.T)

    # Calculate Determinant
    det = np.linalg.det(F)
    return det


def fischer_sumeigenval(fsmp: FischerModelParametrized, S, C):
    # Calculate Fisher Matrix
    F = S.dot(C).dot(S.T)

    # Calculate sum eigenvals
    sumeigval = np.sum(np.linalg.eigvals(F))
    return sumeigval


def fischer_mineigenval(fsmp: FischerModelParametrized, S, C):
    # Calculate Fisher Matrix
    F = S.dot(C).dot(S.T)

    # Calculate sum eigenvals
    mineigval = np.min(np.linalg.eigvals(F))
    return mineigval


def calculate_fischer_criterion(fsmp: FischerModelParametrized, covar=False, relative_sensitivities=False):
    S, C, solutions = get_S_matrix(fsmp, relative_sensitivities)
    if covar == False:
        C = np.eye(S.shape[1])
    crit = fsmp.criterion_func(fsmp, S, C)

    args = {key:value for key, value in fsmp.__dict__.items() if not key.startswith('_')}

    fsr = FischerResult(
        **args,
        criterion=crit,
        sensitivity_matrix=S,
        covariance_matrix=C,
        ode_solutions=solutions
    )
    return fsr
