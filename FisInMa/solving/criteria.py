import numpy as np

from FisInMa.model import FisherModelParametrized


def fisher_determinant(fsmp: FisherModelParametrized, S, C):
    """Calculate the determinant of the Fisher information matrix (the D-optimality criterion) using the sensitivity matrix.

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
    F = S @ C @ S.T

    # Calculate Determinant
    det = np.linalg.det(F)
    return det


def fisher_sumeigenval(fsmp: FisherModelParametrized, S, C):
    """Calculate the sum of the all eigenvalues of the Fisher information matrix (the A-optimality criterion) using the sensitivity matrix.

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
    F = S @ C @ S.T

    # Calculate sum eigenvals
    sumeigval = np.sum(np.linalg.eigvals(F))
    return sumeigval


def fisher_mineigenval(fsmp: FisherModelParametrized, S, C):
    """Calculate the minimal eigenvalue of the Fisher information matrix (the E-optimality criterion) using the sensitivity matrix.

    :param fsmp: The parametrized FisherModel with a chosen values for the sampled variables.
    :type fsmp: FisherModelParametrized
    :param S: The sensitivity matrix.
    :type S: np.ndarray
    :param C: The covariance matrix of the measurement errors.
    :type C: np.ndarray

    :return: The minimal eigenvalue of the Fisher information matrix.
    :rtype: float
    """
    # Calculate Fisher Matrix
    F = S @ C @ S.T

    # Calculate sum eigenvals
    try:
        mineigval = np.min(np.linalg.eigvals(F))
    except:
        mineigval = 0.0
    return mineigval


def fisher_ratioeigenval(fsmp: FisherModelParametrized, S, C):
    """Calculate the ratio of the minimal and maximal eigenvalues of the Fisher information matrix (the modified E-optimality criterion) using the sensitivity matrix.

    :param fsmp: The parametrized FisherModel with a chosen values for the sampled variables.
    :type fsmp: FisherModelParametrized
    :param S: The sensitivity matrix.
    :type S: np.ndarray
    :param C: The covariance matrix of the measurement errors.
    :type C: np.ndarray

    :return: The ratio of the minimal and maximal eigenvalues of the Fisher information matrix.
    :rtype: float
    """
    # Calculate Fisher Matrix
    F = S @ C @ S.T

    # Calculate sum eigenvals
    try:
        eigvals = np.linalg.eigvals(F)
        ratioeigval = np.min(eigvals) / np.max(eigvals)
    except:
        ratioeigval = 0.0
    return ratioeigval