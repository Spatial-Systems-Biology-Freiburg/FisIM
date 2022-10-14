import numpy as np
from scipy.integrate import odeint
import itertools

from FisInMa.model import FischerModelParametrized, FischerResults, FischerResultSingle


def get_S_matrix(fsmp: FischerModelParametrized, relative_sensitivities=False):
    """now we calculate the derivative with respect to the parameters
    The matrix S has the form
    i   -->  index of parameter
    jk  -->  index of kth variable
    t   -->  index of time
    S[i, j1, j2, ..., t] = (dO/dp_i(v_j1, v_j2, v_j3, ..., t))"""
    S = np.zeros((len(fsmp.parameters), fsmp.times.shape[-1],) + tuple(len(x) for x in fsmp.inputs))
    error_n = np.zeros((fsmp.times.shape[-1],) + tuple(len(x) for x in fsmp.inputs))

    # Iterate over all combinations of Q-Values
    solutions = []
    for index in itertools.product(*[range(len(q)) for q in fsmp.inputs]):
        # Store the results of the respective ODE solution
        Q = [fsmp.inputs[i][j] for i, j in enumerate(index)]
        if fsmp.identical_times==True:
            t = fsmp.times
        else:
            t = fsmp.times[index]
        t_init = np.insert(t, 0, fsmp.ode_t0)

        # Actually solve the ODE for the selected parameter values
        #r = solve_ivp(ODE_func, [t0, t.max()], y0, method='Radau', t_eval=t,  args=(Q, P, Const), jac=jacobian).y.T[1:,:]
        # TODO use solve_ivp in future
        # TODO this is NOT the correct function with which to solve
        # we need to mash in ode_dfdx and ode_dfdp to obtain the complete system
        # set the initial values for all non-core ode components to 0
        res = odeint(fsmp.ode_fun, fsmp.ode_y0, t_init, args=(Q, fsmp.parameters, fsmp.constants))#, Dfun=fsmp.ode_dfdx)
        # TODO this in turn is also not the correct result!
        r = res.T[:, 1:]

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
        # TODO
        # TODO This is unaceptable!
        error_n[(slice(None),) + index] = r[0] * 0.25
        # TODO
        # TODO
        fsrs = FischerResultSingle(
            ode_y0=fsmp.ode_y0,
            ode_t0=fsmp.ode_t0,
            times=fsmp.times,
            inputs=fsmp.inputs,
            parameters=fsmp.parameters,
            constants=fsmp.constants,
            ode_solution=res,
            identical_times=fsmp.identical_times
        )
        solutions.append(fsrs)
    
    # Reshape to 2D Form (len(P),:)
    S = S.reshape((len(fsmp.parameters),np.prod(S.shape[1:])))
    error_n = error_n.flatten()
    # cov_matrix = np.eye(len(error_n), len(error_n)) * error_n**2
    # C = np.linalg.inv(cov_matrix)
    C = np.eye(len(error_n))
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

    fsr = FischerResults(
        **args,
        criterion=crit,
        sensitivity_matrix=S,
        covariance_matrix=C,
        ode_solutions=solutions
    )
    return fsr
