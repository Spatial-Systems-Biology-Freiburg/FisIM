import numpy as np
import scipy as sp
import sympy as sym
import time
import matplotlib.pyplot as plt


def f(x, t, inputs, params, consts):
    A, B = x
    (Q,) = inputs
    (p, q) = params
    return [
        p * B,
        - q * Q * A
    ]


def dfdx(x, t, inputs, params, consts):
    A, B = x
    (Q,) = inputs
    (p, q) = params
    return [
        [0, p],
        [- q * Q, 0]
    ]


def dfdp(x, t, inputs, params, consts):
    A, B = x
    (Q,) = inputs
    (p, q) = params
    return [
        [B, 0],
        [0, - Q * A]
    ]


def calculate_dfdx(f, x_len, inputs_len, params_len, consts_len):
    x_symbols = [sym.Symbol('x'+str(i)) for i in range(x_len)]
    t_symbol = sym.Symbol('t')
    inputs_symbols = [sym.Symbol('I'+str(i)) for i in range(inputs_len)]
    params_symbols = [sym.Symbol('P'+str(i)) for i in range(params_len)]
    consts_symbols = [sym.Symbol('C'+str(i)) for i in range(consts_len)]

    symbols = [x_symbols, t_symbol, inputs_symbols, params_symbols, consts_symbols]

    dfdx = [
        [sym.diff(f(x_symbols, t_symbol, inputs_symbols, params_symbols, consts_symbols)[i], x_symbols[j]) for i in range(x_len)] for j in range(x_len)
    ]

    d = [
        [sym.lambdify(symbols, dfdx[i][j], "numpy") for i in range(x_len)]
    for j in range(x_len)]

    dfdx_deriv = lambda x, t, inputs, params, consts: np.array([
        [d[i][j](x, t, inputs, params, consts) for i in range(x_len)]
    for j in range(x_len)])

    return dfdx_deriv


if __name__ == "__main__":
    # Only temporary helper variables
    x_len = 2
    inputs_len = 1
    params_len = 2
    consts_len = 1

    # Symbolically differentiate the function f to obtain dfdx
    dfdx_deriv = calculate_dfdx(f, x_len, inputs_len, params_len, consts_len)

    # Define initial values of ODE
    x0 = np.array([1.0, 2.0])
    t0 = 0.0

    # Define external input values
    inputs = [
        np.array([1.0, 2.0, 3.0])
    ]

    # Define ode_args and parameters
    params = np.array([2.3, 1.1])
    consts = (2.0,)

    # Define number of iterations
    N = 1000

    # Time symbolic differentiation
    print("Comparing as helper functions in ode solver")
    start = time.time()
    for _ in range(N):
        r = sp.integrate.odeint(f, x0, t=np.linspace(t0, 10.0), args=([inputs[0][0]], params, consts), Dfun=dfdx_deriv)
    end = time.time()
    t_sym_1 = end-start
    print("Sympy:   ", t_sym_1)

    # Time explicit method
    start = time.time()
    for _ in range(N):
        r = sp.integrate.odeint(f, x0, t=np.linspace(t0, 10.0), args=([inputs[0][0]], params, consts), Dfun=dfdx)
    end = time.time()
    t_expl_1 = end - start
    print("Explicit: ", t_expl_1)
    print("Ratio:    ", t_sym_1/t_expl_1)

    # Create new functions to dummy test
    dfdx_deriv_flat = lambda x, t, inputs, params, consts: np.array(dfdx_deriv(x, t, inputs, params, consts)).flatten()[1:3]
    dfdx_deriv_flat0 = np.array(dfdx_deriv(x0, t0, [inputs[0][0]], params, consts)).flatten()[1:3]

    dfdx_flat = lambda x, t, inputs, params, consts: np.array(dfdx(x, t, inputs, params, consts)).flatten()[1:3]
    dfdx_flat0 = np.array(dfdx(x0, t0, [inputs[0][0]], params, consts)).flatten()[1:3]

    # Time symbolic differentiation
    print("\nComparing as main targets in ode solver")
    start = time.time()
    for _ in range(N):
        r = sp.integrate.odeint(dfdx_deriv_flat, dfdx_deriv_flat0, t=np.linspace(t0, 10.0), args=([inputs[0][0]], params, consts))
    end = time.time()
    t_sym_2 = end-start
    print("Sympy:    ", t_sym_2)

    # Time explicit
    start = time.time()
    for _ in range(N):
        r = sp.integrate.odeint(dfdx_flat, dfdx_flat0, t=np.linspace(t0, 10.0), args=([inputs[0][0]], params, consts))
    end = time.time()
    t_expl_2 = end - start
    print("Explicit: ", t_expl_2)
    print("Ratio:    ", t_sym_2/t_expl_2)