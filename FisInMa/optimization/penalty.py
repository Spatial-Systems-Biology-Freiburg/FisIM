import numpy as np
import itertools

from pydantic.dataclasses import dataclass

from FisInMa.model import FisherModel, FisherModelParametrized, VariableDefinition


class PenaltyConfig:
    arbitrary_types_allowed = True


@dataclass(config=PenaltyConfig)
class PenaltyInformation:
    penalty: float
    penalty_ode_t0: float
    # TODO - add penalty for ode_x0 when sampling is done
    # penalty_ode_x0: List[List[float]]
    penalty_inputs: float
    penalty_times: float
    penalty_summary: dict


def penalty_structure_zigzag(v, dv):
    r"""Define the zigzag structure of the penalty potential between two allowed discrete values described by equation 

    .. math::

        U_1(v) = \bigg|1 - \frac{2 v}{dv}\bigg|,

    where :math:`v` is the distance between the optimized value and the smaller neighboring discrete value and :math:`dv` is the distance between smaller and larger neighboring discrete values.    
    The function is used as an argument in function :py:meth:`discrete_penalty_individual_template` and dtermines the shape of it.

    :param v: The distance between the optimized value and the smaller neighboring discrete value :math:`v`.
    :type v: float
    :param dv: The distance between smaller and larger neighboring discrete values :math:`dv`.
    :type dv: float

    :return: The value of the penalty potential.
    :rtype: float
    """
    return np.abs(1 - 2 * v / dv)


def penalty_structure_cos(v, dv):
    r"""Define the cosine structure of the penalty potential between two allowed discrete values described by equation 

    .. math::

        U_1(v) = \frac{1}{2} (1 + \cos{2 \pi v}),

    where :math:`v` is the distance between the optimized value and the smaller neighboring discrete value and :math:`dv` is the distance between smaller and larger neighboring discrete values.    
    The function is used as an argument in function :py:meth:`discrete_penalty_individual_template` and dtermines the shape of it.

    :param v: The distance between the optimized value and the smaller neighboring discrete value :math:`v`.
    :type v: float
    :param dv: The distance between smaller and larger neighboring discrete values :math:`dv`.
    :type dv: float

    :return: The value of the penalty potential.
    :rtype: float
    """
    return 0.5 * (1 + np.cos(2*np.pi * v / dv))


def penalty_structure_gauss(v, dv):
    r"""Define the gaussian structure of the penalty potential between two allowed discrete values described by equation 

    .. math::

        U_1(v) = e^{-\frac{v^2}{2\sigma^2}} + e^{-\frac{(v-dv)^2}{2\sigma^2}}

    where :math:`v` is the distance between the optimized value and the smaller neighboring discrete value, :math:`dv` is the distance between smaller and larger neighboring discrete values
    and the variance is :math:`\sigma = 0.1dv`.    
    The function is used as an argument in function :py:meth:`discrete_penalty_individual_template` and dtermines the shape of it.

    :param v: The distance between the optimized value and the smaller neighboring discrete value :math:`v`.
    :type v: float
    :param dv: The distance between smaller and larger neighboring discrete values :math:`dv`.
    :type dv: float

    :return: The value of the penalty potential.
    :rtype: float
    """
    sigma = dv / 10
    return np.exp(- 0.5 * v**2 / sigma**2) +  np.exp(- 0.5 * (v - dv)**2 / sigma**2)


def discrete_penalty_individual_template(vals, vals_discr, pen_structure):
    r"""The discretization penalty function template.
    If there is no penalty, a function gives 1 and 0 in case of the maximum penalty for data points that do not sit on the desired discretization points.

    The resulting contribution of the penalty function is calculated as a product of all penalty values  for each value :math:`v`.
    
    .. math::

      U = \prod_{i=1} U_1(v_i).


    :param vals: The array of values to optimize :math:`v`.
    :type vals: np.ndarary
    :param vals_discr: The array of allowed discrete values :math:`v^{\text{discr}}`.
    :type vals_discr: np.ndarary
    :param pen_structure: Define the structure of the template.

        - :py:meth:`penalty_structure_zigzag`
            Use zigzag structure.

        - :py:meth:`penalty_structure_cos`
            Use cosine function for potential.
            
        - :py:meth:`penalty_structure_gauss`
            Use two Gaussian functions.

    .. figure:: discretization_template.png
      :align: center
      :width: 450

      The discretization penalty function for discrete values :math:`v^{\text{discr}} = [1, 2, 3, 6, 8, 9]` for different penalty structures.

    :type pen_structure: Callable
    
    :return: The array of the penalty potential values for *vals*. The resulting contribution (product) of the penalty function.
    :rtype: np.ndarary, float
    """
    prod = []
    for v in vals:
        for i in range (len(vals_discr)-1):
            if vals_discr[i] <= v <= vals_discr[i+1]:
                dx = vals_discr[i+1] - vals_discr[i]
                prod.append(pen_structure(v - vals_discr[i], dx))
    pen = np.product(prod)
    return pen, prod


def discrete_penalty_calculator_default(vals, vals_discr):
    r"""The discretization penalty function taken as a product of all differences between the optimized value :math:`v` and all possible discrete values allowed :math:`v^{\text{discr}}`.
    If there is no penalty, a function gives 1 and 0 in case of the maximum penalty for data points that do not sit on the desired discretization points.
    
    .. math::

      U_1(v) = 1 - \sqrt[N]{\prod_{k=1}^N (v - v^{\text{discr}}_{k})} \frac{1}{\max(v^{\text{discr}}) - \min(v^{\text{discr}})},

    where :math:`N` is the size of the vector of the allowed discrete values :math:`v^{\text{discr}}`. 
    The resulting contribution of the penalty function is a product of the potential values for each value :math:`v`.
    
    .. figure:: discretization_product.png
      :align: center
      :width: 400

      The discretization penalty function for discrete values :math:`v^{\text{discr}} = [1, 2, 3, 6, 8, 9]`.

    The resulting contribution of the penalty function is calculated as a product of all penalty values for each value :math:`v`.
    
    .. math::

      U = \prod_{i=1} U_1(v_i).


    :param vals: The array of values to optimize :math:`v`.
    :type vals: np.ndarary
    :param vals_discr: The array of allowed discrete values :math:`v^{\text{discr}}`.
    :type vals_discr: np.ndarary

    :return: The array of the penalty potential values for *vals*. The resulting contribution (product) of the penalty function.
    :rtype: np.ndarary, float
    """
    # TODO - document this function
    # TODO - should be specifiable as parameter in optimization routine
    # Calculate the penalty for provided values
    prod = np.array([1 - (np.abs(np.prod((vals_discr - v))))**(1.0 / len(vals_discr)) / (np.max(vals_discr) - np.min(vals_discr)) for v in vals])
    pen = np.product(prod)
    # Return the penalty and the output per inserted variable
    return pen, prod


DISCRETE_PENALTY_FUNCTIONS = {
    "default": discrete_penalty_calculator_default,
    "product_difference": discrete_penalty_calculator_default,
    "individual_zigzag": lambda vals, vals_discr: discrete_penalty_individual_template(vals, vals_discr, penalty_structure_zigzag),
    "individual_cos": lambda vals, vals_discr: discrete_penalty_individual_template(vals, vals_discr, penalty_structure_cos),
    "individual_gauss": lambda vals, vals_discr: discrete_penalty_individual_template(vals, vals_discr, penalty_structure_gauss),
}


def _discrete_penalizer(fsmp: FisherModelParametrized, penalizer_name="default"):
    penalizer = DISCRETE_PENALTY_FUNCTIONS[penalizer_name]
    # Penalty contribution from initial times
    pen_ode_t0 = 1
    pen_ode_t0_full = []
    if type(fsmp.ode_t0_def) is VariableDefinition:
        # Now we can expect that this parameter was sampled
        # thus we want to look for possible discretization values
        discr = fsmp.ode_t0_def.discrete
        if type(discr) is np.ndarray:
            values = fsmp.ode_t0
            pen_ode_t0, pen_ode_t0_full = penalizer(values, discr)

    # Penalty contribution from inputs
    pen_inputs = 1
    pen_inputs_full = []
    for var_def, var_val in zip(fsmp.inputs_def, fsmp.inputs):
        if type(var_def) == VariableDefinition:
            discr = var_def.discrete
            if type(discr) is np.ndarray:
                values = var_val
                p, p_full = penalizer(values, discr)
                pen_inputs *= p
                pen_inputs_full.append(p_full)

    # Penalty contribution from times
    pen_times = 1
    pen_times_full = []
    if type(fsmp.times_def) is VariableDefinition:
        discr = fsmp.times_def.discrete
        if type(discr) is np.ndarray:
            if fsmp.identical_times==True:
                # TODO test these statements
                values = fsmp.times
                pen_times, pen_times_full = penalizer(values, discr)
            else:
                pen_times_full = []
                for index in itertools.product(*[range(len(q)) for q in fsmp.inputs]):
                    if fsmp.identical_times==True:
                        # TODO test this statement
                        values = fsmp.times
                    else:
                        values = fsmp.times[index]
                    p, p_full = penalizer(values, discr)
                    pen_times *= p
                    pen_times_full.append(p_full)

    # Calculate the total penalty
    pen = pen_ode_t0 * pen_inputs * pen_times

    # Create a summary
    pen_summary = {
        "ode_t0": pen_ode_t0_full,
        "inputs": pen_inputs_full,
        "times": pen_times_full
    }

    # Store values in class
    ret = PenaltyInformation(
        penalty=pen,
        penalty_ode_t0=pen_ode_t0,
        penalty_inputs=pen_inputs,
        penalty_times=pen_times,
        penalty_summary=pen_summary,
    )

    # Store all results and calculate total penalty
    return pen, ret
