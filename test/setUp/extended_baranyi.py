import numpy as np
import pytest

from FisInMa import *


def baranyi_roberts_ode(t, x, u, p, ode_args):
    x1, x2, y1, y2 = x
    (Temp, ) = u
    (xy_max, b_x, Temp_min_x, b_y, Temp_min_y, ) = p
    # Define the maximum growth rate
    mu_max_x = b_x**2 * (Temp - Temp_min_x)**2
    mu_max_y = b_y**2 * (Temp - Temp_min_y)**2
    return [
        mu_max_x * (x2/(x2 + 1)) * x1 * (1 - x1/xy_max - y1 / xy_max), # f1
        mu_max_x * x2,                                                 # f2
        mu_max_y * (y2/(y2 + 1)) * y1 * (1 - y1/xy_max - x1 / xy_max), # f3
        mu_max_y * y2,                                                 # f4
    ]

def ode_dfdp(t, x, u, p, ode_args):
    x1, x2, y1, y2 = x
    (Temp, ) = u
    (xy_max, b_x, Temp_min_x, b_y, Temp_min_y, ) = p
    # Define the maximum growth rate
    mu_max_x = b_x**2 * (Temp - Temp_min_x)**2
    mu_max_y = b_y**2 * (Temp - Temp_min_y)**2
    return [
        [
            mu_max_x * (x2/(x2 + 1)) * x1 * (x1 + y1) / (xy_max**2),                         # df1/dxy_max
             2 * b_x * (Temp - Temp_min_x)**2 * (x2/(x2 + 1)) * x1 * (1 - (x1 + y1)/xy_max), # df1/db_x
            -2 * b_x**2 * (Temp - Temp_min_x) * (x2/(x2 + 1)) * x1 * (1 - (x1 + y1)/xy_max), # df1/dTemp_min_x
            0,                                                                               # df1/db_y
            0                                                                                # df1/dTemp_min_y
        ],
        [
            0,                                      # df2/dxy_max
            2 * b_x * (Temp - Temp_min_x)**2 * x2,  # df2/dbx
            -2 * b_x**2 * (Temp - Temp_min_x) * x2, # df2/dTemp_minx
            0,
            0
        ],
        [
            mu_max_y * (y2/(y2 + 1)) * y1 * (x1 + y1) / (xy_max**2), # df3/dxy_max
            0,
            0,
            2 * b_y * (Temp - Temp_min_y)**2 * (y2/(y2 + 1)) * y1 * (1 - (x1 + y1)/xy_max),
            -2 * b_y**2 * (Temp - Temp_min_y) * (y2/(y2 + 1)) * y1 * (1 - (x1 + y1)/xy_max),
        ],
        [
            0,
            0,
            0,
            2 * b_y * (Temp - Temp_min_y)**2 * y2,
            -2 * b_y**2 * (Temp - Temp_min_y) * y2,
        ]
    ]

def ode_dfdx(t, x, u, p, ode_args):
    x1, x2, y1, y2 = x
    (Temp, ) = u
    (xy_max, b_x, Temp_min_x, b_y, Temp_min_y, ) = p
    # Define the maximum growth rate
    mu_max_x = b_x**2 * (Temp - Temp_min_x)**2
    mu_max_y = b_y**2 * (Temp - Temp_min_y)**2
    return [
        [
            mu_max_x * (x2/(x2 + 1)) * (1 - (2*x1 + y1)/xy_max),  # df1/dx1
            mu_max_x * 1/(x2 + 1)**2 * (1 - (x1 + y1)/xy_max)*x1, # df1/dx2
            -mu_max_x * x1 / xy_max,                              # df1/dy1
            0,                                                    # df1/dy2
        ],
        [
            0,                                                    # df2/dx1
            mu_max_x,                                             # df2/dx2
            0,
            0
        ],
        [
            -mu_max_y * y1 / xy_max,                              # df3/dx1
            0,
            mu_max_y * (y2/(y2 + 1)) * (1 - (2*y1 + x1)/xy_max),  # df3/dy1
            mu_max_y * 1/(y2 + 1)**2 * (1 - (x1 + y1)/xy_max)*y1, # df3/dy2
        ],
        [
            0,                                                                    # df4/dx1
            0,                                                                    # df4/dx2
            0,
            mu_max_y
        ]
    ]


class ModelExtendedBaranyi:
    def __init__(self, obs=[0], N_x0=2, n_t0=13, n_times=5, n_inputs=7, identical_times=False):
        # Use prime numbers for sampled parameters to
        # show errors in code where reshaping is done
        self.x0 = [np.array([1e4, 0.1 / i, 1e2, 0.1 / i]) for i in range(1, N_x0+1)]
        self.t0=np.linspace(0.0, 0.01, n_t0)
        self.times=np.linspace(0.1, 10.0, n_times)
        self.inputs=[np.arange(3, n_inputs + 3)]

        self.parameters = (1e8, 0.2, 1.0, 0.2, 1.0)
        self.fsm = FisherModel(
            ode_fun=baranyi_roberts_ode,
            ode_dfdx=ode_dfdx,
            ode_dfdp=ode_dfdp,
            ode_initial=self.x0,
            ode_x0=self.x0,
            ode_t0=self.t0,
            times=self.times,
            inputs=self.inputs,
            parameters=self.parameters,
            obs_fun=obs,
            identical_times=identical_times,
        )
        self.fsmp = FisherModelParametrized.init_from(self.fsm)


@pytest.fixture()
def extended_baranyi_model():
    return ModelExtendedBaranyi()

@pytest.fixture()
def extended_baranyi_model_parametrized(N_x0, n_t0, n_times, n_inputs, identical_times):
    return ModelExtendedBaranyi(N_x0, n_t0, n_times, n_inputs, identical_times)

@pytest.fixture()
def extended_baranyi_model_small(identical_times):
    return ModelExtendedBaranyi(N_x0=1, n_t0=1, n_times=1, n_inputs=1, identical_times=identical_times)

