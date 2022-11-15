import numpy as np
import scipy.integrate as integrate
import itertools

from FisInMa.model import FisherModelParametrized, FisherResults, FisherResultSingle
from .criteria import fisher_determinant


def ode_rhs(t, x, ode_fun, ode_dfdx, ode_dfdp, ode_dfdx0, inputs, parameters, ode_args, n_x, n_p):
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
    :param ode_args: The ode_args of the system :math:`c`.
    :type ode_args: tuple
    :param n_x: The number of the state variables of the system.
    :type n_x: int
    :param n_p: The number of the estimated parameters of the system.
    :type n_p: int

    :return: The right-hand side of the ODEs system for sensitivities calculation.
    :rtype: np.ndarray
    """
    if callable(ode_dfdx0):
        n_s_x0 = n_x
    else:
        n_s_x0 = 0
    x_fun, s, s_x0, rest = np.split(x, [n_x, n_x + n_x*n_p, n_x + n_x*n_p + n_s_x0])
    s = s.reshape((n_x, n_p))
    s_x0 = s_x0.reshape((n_s_x0, n_s_x0))
    dx_f = np.asarray(ode_fun(t, x_fun, inputs, parameters, ode_args), dtype=float).reshape((n_x))
    dfdx = np.asarray(ode_dfdx(t, x_fun, inputs, parameters, ode_args), dtype=float).reshape((n_x, n_x))
    dfdp = np.asarray(ode_dfdp(t, x_fun, inputs, parameters, ode_args), dtype=float).reshape((n_x, n_p))

    # Calculate the rhs of the sensitivities
    ds = np.dot(dfdx, s) + dfdp
    if callable(ode_dfdx0):
        dfdx0 = np.asarray(ode_dfdx0(t, x_fun, inputs, parameters, ode_args)).reshape((n_x, n_x))
        ds_x0 = np.dot(dfdx, s_x0) + dfdx0
        x_tot = np.concatenate((dx_f, *ds, *ds_x0))
    else:
        x_tot = np.concatenate((dx_f, *ds))
    return x_tot


def _calculate_sensitivities_with_observable(fsmp: FisherModelParametrized, t: np.ndarray, x: np.ndarray, s: np.ndarray, Q: np.ndarray, n_x:int, n_obs: int, n_p: int, relative_sensitivities=False, **kwargs):
    # Shape annotations:
    # t: (n_t)
    # x: (n_x, n_t)
    # s: (n_p_full, n_x, n_t)
    # term1: (n_p, n_obs, n_t)
    # term2: (n_p, n_obs, n_t)
    # term1_full: (n_p_full, n_obs, n_t)
    # term2_full: (n_p_full, n_obs, n_t)
    n_t = t.size
    if callable(fsmp.obs_fun) and callable(fsmp.obs_dgdp) and callable(fsmp.obs_dgdx):
        # Calculate the first term of the equation
        term1 = np.array([fsmp.obs_dgdp(ti, x[:,i_t], Q, fsmp.parameters, fsmp.ode_args) for i_t, ti in enumerate(t)]).reshape((n_t, n_obs, n_p)).swapaxes(0, 2)

        # Calculate the second term of the equation and add them
        term2 = np.array([
            # This has shape (n_x, n_obs)
            np.array(fsmp.obs_dgdx(ti, x[:,i], Q, fsmp.parameters, fsmp.ode_args), dtype=float)
                .reshape((n_x, n_obs))
                .T
                .dot(s[:n_p,:,i].T)
            for i, ti in enumerate(t)
        ]).reshape((n_t, n_obs, n_p)).swapaxes(0, 2)

        if callable(fsmp.ode_dfdx0) and callable(fsmp.obs_dgdx0):
            # Calculate term1_add which is dgdx0_ik for every time point in t
            term1_add = np.array([
                fsmp.obs_dgdx0(ti, x[:,i_t], Q, fsmp.parameters, fsmp.ode_args)
                for i_t, ti in enumerate(t)
            ]).reshape((-1, n_obs, n_x)).swapaxes(0, 2)

            # Calculate term2_add which is dgdx0_ij * s_jk
            term2_add = np.array([
                np.array(fsmp.obs_dgdx(ti, x[:,i], Q, fsmp.parameters, fsmp.ode_args), dtype=float)
                    .reshape((n_x, n_obs))
                    .T
                    .dot(s[n_p:,:,i].T)
                for i, ti in enumerate(t)
            ]).reshape((n_t, n_obs, n_x)).swapaxes(0, 2)

            term1_full = np.concatenate([term1, term1_add], axis=0)
            term2_full = np.concatenate([term2, term2_add], axis=0)

            s = term1_full + term2_full
        else:
            s = term1 + term2

        # Also calculate the results for the observable which can be used later for relative sensitivities
        obs = np.array([fsmp.obs_fun(ti, x[:,i_t], Q, fsmp.parameters, fsmp.ode_args) for i_t, ti in enumerate(t)]).reshape((-1, n_obs)).T
        return s, obs
    else:
        return s, x


def get_S_matrix(fsmp: FisherModelParametrized, relative_sensitivities=False, **kwargs):
    r"""Calculate the sensitivity matrix for a Fisher Model.

    :param fsmp: The parametrized FisherModel with a chosen values for the sampled variables.
    :type fsmp: FisherModelParametrized
    :param relative_sensitivities: Use relative local sensitivities :math:`s_{ij} = \frac{\partial y_i}{\partial p_j} \frac{p_j}{y_i}` instead of absolute. Defaults to False.
    :type relative_sensitivities: bool, optional

    :return: The sensitivity matrix S, the cobvariance matrix C, the object of type FisherResultSingle with ODEs solutions.
    :rtype: np.ndarray, np.ndarray, FisherResultSingle
    """   
    # Helper variables
    # How many initial times do we have?
    n_t0 = len(fsmp.ode_t0)
    # How large is the vector of one initial value? (ie. dimensionality of the ODE)
    n_x0 = len(fsmp.ode_x0[0])
    # How many parameters are in the system?
    n_p = len(fsmp.parameters)
    n_p_full = len(fsmp.parameters) + n_x0 if callable(fsmp.ode_dfdx0) else len(fsmp.parameters)
    # How many different initial values do we have?
    N_x0 = len(fsmp.ode_x0)
    # The lengths of the individual input variables stored as tuple
    inputs_shape = tuple(len(q) for q in fsmp.inputs)

    # Determine the number of components the observable has by evaluating at
    if callable(fsmp.obs_fun) and callable(fsmp.obs_dgdp) and callable(fsmp.obs_dgdx):
        n_obs = np.array(fsmp.obs_fun(fsmp.ode_t0[0], fsmp.ode_x0[0], [q[0] for q in fsmp.inputs], fsmp.parameters, fsmp.ode_args)).size
    else:
        n_obs = n_x0

    # The shape of the initial S matrix is given by
    # (n_p, n_t0, n_x0, n_q0, ..., n_ql, n_times)
    S = np.zeros((n_p_full, n_t0, N_x0, n_obs) + inputs_shape + (fsmp.times.shape[-1],))

    # Do not calculate the covariance any further if both are None
    calculate_covar = fsmp.covariance.abs is not None or fsmp.covariance.rel is not None
    if calculate_covar:
        uncertainty = np.zeros((n_t0, N_x0, n_obs) + inputs_shape + (fsmp.times.shape[-1],))

    # Iterate over all combinations of input-Values and initial values
    solutions = []
    for (i_x0, x0), (i_t0, t0), index in itertools.product(
        enumerate(fsmp.ode_x0),
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
        
        # solve_ivp cannot cope with repeating values.
        # Thus we will filter for them and in post multiply them again
        t_red, counts = np.unique(t, return_counts=True)

        # Define initial values for ode
        if callable(fsmp.ode_dfdx0):
            x0_full = np.concatenate((x0, np.zeros(n_x0 * n_p), np.ones(n_x0)))
        else:
            x0_full = np.concatenate((x0, np.zeros(n_x0 * n_p)))

        # Make sure that the t_span interval is actually not empty (only for python 3.7)
        t_max = np.max(t) if np.max(t)>t0 else t0+1e-30

        # Actually solve the ODE for the selected parameter values
        res = integrate.solve_ivp(fun=ode_rhs, t_span=(t0, t_max), y0=x0_full, t_eval=t_red, args=(fsmp.ode_fun, fsmp.ode_dfdx, fsmp.ode_dfdp, fsmp.ode_dfdx0, Q, fsmp.parameters, fsmp.ode_args, n_x0, n_p), method="LSODA", rtol=1e-4)
        # Obtain sensitivities dg/dp from the last components of the ode
        # Check if t_red is made up of only initial values

        # If time values were only made up of initial time,
        # we simply set everything to zero, since these are the initial values for the sensitivities
        if np.all(t_red == t0):
            s = np.zeros((n_p_full, n_x0, counts[0]))
        else:
            r = np.array(res.y[n_x0:])
            s = np.swapaxes(r.reshape((n_x0, n_p_full, -1)), 0, 1)

        # If the observable was specified we will transform the result with
        # dgdp = dgdp + dxdp * dgdx
        x = res.y[:n_x0].reshape((n_x0, -1))
        s, obs = _calculate_sensitivities_with_observable(fsmp, t_red, x, s, Q, n_x0, n_obs, n_p, relative_sensitivities, **kwargs)

        # Multiply the values again to obtain desired shape for sensitivity matrix
        s = np.repeat(s, counts, axis=2)
        obs = np.repeat(obs, counts, axis=1)

        # Define constants for covariance calculation
        if calculate_covar:
            c_abs = 0.0 if fsmp.covariance.abs is None else fsmp.covariance.abs
            c_rel = 0.0 if fsmp.covariance.rel is None else fsmp.covariance.rel

        # Calculate the S-Matrix from the sensitivities
        # Depending on if we want to calculate the relative sensitivities
        if relative_sensitivities==True:
            # Multiply by parameter
            if callable(fsmp.ode_dfdx0):
                params = fsmp.parameters + (*fsmp.ode_x0[0],)
            else:
                params = fsmp.parameters
            for i, p in enumerate(params):
                s[i] *= p

            # Divide by observable
            for i, o in enumerate(obs):
                s[(slice(None), i)] /= o

            # Fill S-Matrix
            S[(slice(None), i_t0, i_x0, slice(None)) + index] = s

            # Calculate the uncertainty
            if calculate_covar:
                uncertainty[(i_t0, i_x0, slice(None)) + index] = c_rel + c_abs/obs
        else:
            S[(slice(None), i_t0, i_x0, slice(None)) + index] = s

            # Calculate the uncertainty
            if calculate_covar:
                uncertainty[(i_t0, i_x0, slice(None)) + index] = c_rel*obs + c_abs

        # Assume that the error of the measurement is 25% from the measured value r[0] n 
        # (use for covariance matrix calculation)
        fsrs = FisherResultSingle(
            ode_x0=x0,
            ode_t0=t0,
            times=t,
            inputs=Q,
            parameters=fsmp.parameters,
            ode_args=fsmp.ode_args,
            ode_solution=res,
            sensitivities=s,
            identical_times=fsmp.identical_times,
            observables = obs
        )
        solutions.append(fsrs)
    
    # Calculate the covariance matrix
    if calculate_covar==True:
        uncertainty = uncertainty.flatten()
        cov_matrix = np.eye(len(uncertainty), len(uncertainty)) * uncertainty**2
        try:
            C = np.linalg.inv(cov_matrix)
        except:
            C = np.full((len(uncertainty), len(uncertainty)), np.nan)
    else:
        n_datapoints = np.product(S.shape[1:])
        C = np.eye(n_datapoints)

    # Reshape to 2D Form (len(P),:)
    S = S.reshape((n_p_full,-1))
    
    return S, C, solutions


def calculate_fisher_criterion(fsmp: FisherModelParametrized, criterion=fisher_determinant, relative_sensitivities=False, verbose=False):
    """Calculate the Fisher information optimality criterion for a chosen Fisher model.

    :param fsmp: The parametrized FisherModel with a chosen values for the sampled variables.
    :param criterion: Choose the optimality criterion to determine the objective function and quantify the Experimental Design. The default is "fisher_determinant".

        - fisher_determinant
            Use the D-optimality criterion that maximizes the determinant of the Fisher Information matrix.
        - fisher_mineigenval
            Use the E-optimality criterion that maximizes the minimal eigenvalue of the Fisher Information matrix.
        - fisher_sumeigenval
            Use the A-optimality criterion that maximizes the sum of all eigenvalues of the Fisher Information matrix.
        - fisher_ratioeigenval
            Use the modified E-optimality criterion that maximizes the ratio of the minimal and maximal eigenvalues of the Fisher Information matrix.

    :type criterion: callable
    :type fsmp: FisherModelParametrized
    :param relative_sensitivities: Use relative local sensitivities :math:`s_{ij} = \dfrac{\partial y_i}{\partial p_j} \dfrac{p_j}{y_i}` instead of absolute. Defaults to False.
    :type relative_sensitivities: bool, optional

    :return: The result of the Fisher information optimality criterion represented as a FisherResults object.
    :rtype: FisherResults
    """
    S, C, solutions = get_S_matrix(fsmp, relative_sensitivities)
    crit = criterion(fsmp, S, C)

    fsmp_args = {key:value for key, value in fsmp.__dict__.items() if not key.startswith('_')}

    fsr = FisherResults(
        criterion=crit,
        S=S,
        C=C,
        individual_results=solutions,
        criterion_fun=criterion,
        relative_sensitivities=relative_sensitivities,
        **fsmp_args,
    )
    return fsr
