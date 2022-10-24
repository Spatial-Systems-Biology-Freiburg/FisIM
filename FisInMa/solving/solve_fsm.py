import numpy as np
import scipy.integrate as integrate
import itertools

from FisInMa.model import FisherModelParametrized, FisherResults, FisherResultSingle


def ode_rhs(t, x, ode_fun, ode_dfdx, ode_dfdp, inputs, parameters, constants, n_x, n_p):
    r"""Calculate the right-hand side of the ODEs system, containing the model definition with state variables :math:`\dot x = f(x, t, u, u, c)` 
    and the equations for the local sensitivities :math:`\dot s = \frac{\partial f}{\partial x} s + \frac{\partial f}{\partial p}`.
    
    :param t: The measurement times :math:`t`.
    :type t: np.ndarray
    :param x: The array containing the state variables :math:`x` and sensitivities :math:`s`.
    :type x: np.ndarray
    :param ode_fun: The ODEs right-hand side function :math:`f` for the state variables :math:`x`.
    :type ode_fun: callable
    :param ode_dfdx: The derivative of the ODEs function with respect to state variables :math:`x`.
    :type ode_dfdx: callable
    :param ode_dfdp: The derivative of the ODEs function with respect to parameters :math:`p`.
    :type ode_dfdp: callable
    :param inputs: The inputs of the system.
    :type inputs: list
    :param parameters: The estimated parameters of the system :math:`p`.
    :type params: tuple
    :param constants: The constants of the system :math:`c`.
    :type constants: tuple
    :param n_x: The number of the state variables of the system.
    :type n_x: int
    :param n_p: The number of the estimated parameters of the system.
    :type n_p: int

    :return: The right-hand side of the ODEs system for sensitivities calculation.
    :rtype: np.ndarray
    """   
    x_fun, s, rest = lists = np.split(x, [n_x, n_x + n_x*n_p])
    s = s.reshape((n_x, n_p))
    dx_f = ode_fun(t, x_fun, inputs, parameters, constants)
    dfdx = ode_dfdx(t, x_fun, inputs, parameters, constants)
    dfdp = ode_dfdp(t, x_fun, inputs, parameters, constants)
    # Calculate the rhs of the sensitivities
    # TODO validate these equations!
    ds = np.dot(dfdx, s) + dfdp
    x_tot = np.concatenate((dx_f, *ds))
    return x_tot


def get_S_matrix(fsmp: FisherModelParametrized, relative_sensitivities=False):
    r"""Calculate the sensitivity matrix for a Fisher Model.

    :param fsmp: The parametrized FisherModel with a chosen values for the sampled variables.
    :type fsmp: FisherModelParametrized
    :param relative_sensitivities: Use relative local sensitivities :math:`s_{ij} = \frac{\partial y_i}{\partial p_j} \frac{p_j}{y_i}` instead of absolute. Defaults to False.
    :type relative_sensitivities: bool, optional

    :return: The sensitivity matrix S, the cobvariance matrix C, the object of type FisherResultSingle with ODEs solutions.
    :rtype: np.ndarray, np.ndarray, FisherResultSingle
    """   
    # Helper variables
    # How many parameters are in the system?
    n_p = len(fsmp.parameters)
    # How many initial times do we have?
    n_t0 = len(fsmp.ode_t0)
    # How large is the vector of one initial value? (ie. dimensionality of the ODE)
    n_y0 = len(fsmp.ode_y0[0])
    # How many different initial values do we have?
    N_y0 = len(fsmp.ode_y0)
    # The lengths of the individual input variables stored as tuple
    inputs_shape = tuple(len(q) for q in fsmp.inputs)

    # The shape of the initial S matrix is given by
    # (n_p, n_t0, n_y0, n_q0, ..., n_ql, n_times)
    # S = np.zeros((n_x * n_p, fsmp.times.shape[-1],) + tuple(len(x) for x in fsmp.inputs))
    S = np.zeros((n_p, n_t0, N_y0, n_y0) + inputs_shape + (fsmp.times.shape[-1],))
    error_n = np.zeros((fsmp.times.shape[-1],) + tuple(len(x) for x in fsmp.inputs))

    # Iterate over all combinations of Q-Values
    solutions = []
    for (i_y0, y0), (i_t0, t0), index in itertools.product(
        enumerate(fsmp.ode_y0),
        enumerate(fsmp.ode_t0),
        itertools.product(*[range(len(q)) for q in fsmp.inputs])
    ):
        # pick one pair of input values
        Q = [fsmp.inputs[i][j] for i, j in enumerate(index)]
        # Check if identical times are being used
        if fsmp.identical_times==True:
            t = fsmp.times
        else:
            t = fsmp.times[index]
        # t_init = np.insert(t, 0, fsmp.ode_t0)

        # Define initial values for ode
        y0_full = np.concatenate((y0, np.zeros(n_y0 * n_p)))

        # Actually solve the ODE for the selected parameter values
        res = integrate.solve_ivp(fun=ode_rhs, t_span=(t0, np.max(t)), y0=y0_full, t_eval=t, args=(fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, Q, fsmp.parameters, fsmp.constants, n_y0, n_p), method="Radau")#, jac=fsmp.ode_dfdx)
        
        # Obtain sensitivities dg/dp from the last components of the ode
        r = np.array(res.y[n_y0:])
        
        s = np.swapaxes(r.reshape((n_y0, n_p, -1)), 0, 1)

        # Calculate the S-Matrix from the sensitivities
        # Depending on if we want to calculate the relative sensitivities
        if relative_sensitivities==True:
            # Multiply by parameter
            for i, p in enumerate(fsmp.parameters):
                s[i] *= p

            # Divide by observable
            for i, o in enumerate(res.y[:n_y0]):
                s[(slice(None), i)] /= o
            
            # Fill S-Matrix
            S[(slice(None), i_t0, i_y0, slice(None)) + index] = s
        else:
            S[(slice(None), i_t0, i_y0, slice(None)) + index] = s

        # Assume that the error of the measurement is 25% from the measured value r[0] n 
        # (use for covariance matrix calculation)
        # TODO
        # TODO This is unaceptable!
        error_n[(slice(None),) + index] = r[0] * 0.25
        # TODO
        # TODO
        fsrs = FisherResultSingle(
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
    S = S.reshape((n_p,-1))
    # TODO fix this covariance matrix stuff!
    error_n = error_n.flatten()
    # cov_matrix = np.eye(len(error_n), len(error_n)) * error_n**2
    # C = np.linalg.inv(cov_matrix)
    C = np.eye(len(error_n))
    return S, C, solutions


def fisher_determinant(fsmp: FisherModelParametrized, S, C):
    """Calculate the determinant of the Fisher information matrix using the sensitivity matrix.

    :param fsmp: The parametrized FisherModel with a chosen values for the sampled variables.
    :type fsmp: FisherModelParametrized
    :param S: The sensitivity matrix.
    :type S: np.ndarray
    :param C: The covariance matrix of the measurement errors.
    :type C: np.ndarray

    :return: The determinant of the Fisher information matrix.
    :rtype: float
    """
    # Calculate Fisher Matrix
    F = (S.dot(C)).dot(S.T)

    # Calculate Determinant
    det = np.linalg.det(F)
    return det


def fisher_sumeigenval(fsmp: FisherModelParametrized, S, C):
    """Calculate the sum of the all eigenvalues of the Fisher information matrix using the sensitivity matrix.

    :param fsmp: The parametrized FisherModel with a chosen values for the sampled variables.
    :type fsmp: FisherModelParametrized
    :param S: The sensitivity matrix.
    :type S: np.ndarray
    :param C: The covariance matrix of the measurement errors.
    :type C: np.ndarray

    :return: The sum of the eigenvalues of the Fisher information matrix.
    :rtype: float
    """
    # Calculate Fisher Matrix
    F = S.dot(C).dot(S.T)

    # Calculate sum eigenvals
    sumeigval = np.sum(np.linalg.eigvals(F))
    return sumeigval


def fisher_mineigenval(fsmp: FisherModelParametrized, S, C):
    """Calculate the minimal eigenvalue of the Fisher information matrix using the sensitivity matrix.

    :param fsmp: The parametrized FisherModel with a chosen values for the sampled variables.
    :type fsmp: FisherModelParametrized
    :param S: The sensitivity matrix.
    :type S: np.ndarray
    :param C: The covariance matrix of the measurement errors.
    :type C: np.ndarray

    :return: The sum of the eigenvalues of the Fisher information matrix.
    :rtype: float
    """
    # Calculate Fisher Matrix
    F = S.dot(C).dot(S.T)

    # Calculate sum eigenvals
    mineigval = np.min(np.linalg.eigvals(F))
    return mineigval


def calculate_fisher_criterion(fsmp: FisherModelParametrized, criterion=fisher_determinant, covar=False, relative_sensitivities=False):
    r"""Calculate the Fisher information optimality criterion for a chosen Fisher model.

    :param fsmp: The parametrized FisherModel with a chosen values for the sampled variables.
    :type fsmp: FisherModelParametrized
    :param covar: Use the covariance matrix of error measurements. Defaults to False.
    :type covar: bool, optional
    :param relative_sensitivities: Use relative local sensitivities :math:`s_{ij} = \frac{\partial y_i}{\partial p_j} \frac{p_j}{y_i}` instead of absolute. Defaults to False.
    :type relative_sensitivities: bool, optional

    :return: The result of the Fisher information optimality criterion represented as a FisherResults object.
    :rtype: FisherResults
    """
    S, C, solutions = get_S_matrix(fsmp, relative_sensitivities)
    if covar == False:
        C = np.eye(S.shape[1])
    crit = criterion(fsmp, S, C)

    args = {key:value for key, value in fsmp.__dict__.items() if not key.startswith('_')}

    fsr = FisherResults(
        criterion=crit,
        S=S,
        C=C,
        individual_results=solutions,
        _fsm_var_def=fsmp._fsm_var_def,
        **args,
    )
    return fsr
