Sensitivity Calculation
=======================

.. note::

   In-depth information about the theoretical underlying and the calculation methods will be described in a book chapter releasing in the near future.

The Fisher information matrix (FIM) can be easily calculated via the sensitivity matrix :math:`S`:

.. math::
    \begin{alignat}{3}
    F = S^T C^{-1} S,
    \end{alignat}

where :math:`C` is the covariance matrix of measurement error.

As an example, the mentioned sensitivity matrix for two observables :math:`y = (y_1, y_2)`, two different inputs :math:`u = (u_1, u_2)`, :math:`N` different time and :math:`N_p` parameterscan be built in the following way:

.. math::
    S =
    \begin{bmatrix}
    s_{11} (t_1, u_1) & ... & s_{1 N_p}(t_1, u_1) \\
    \vdots  &   & \vdots  \\
    s_{11} (t_{N}, u_1) & ... & s_{1 N_p} (t_{N}, u_1)\\
    s_{11} (t_1, u_2) & ... & s_{1 N_p}(t_1, u_2) \\
    \vdots  &   & \vdots  \\
    s_{11} (t_N, u_2) & ... & s_{1 N_p} (t_N, u_2)\\
    s_{21} (t_1, u_1) & ... & s_{2 N_p}(t_1, u_1) \\
    \vdots  &   & \vdots  \\
    s_{21} (t_{N}, u_1) & ... & s_{2 N_p} (t_{N}, u_1)\\
    s_{21} (t_1, u_2) & ... & s_{2 N_p}(t_1, u_2) \\
    \vdots  &   & \vdots  \\
    s_{21} (t_N, u_2) & ... & s_{2 N_p} (t_N, u_2)
    \end{bmatrix}

Here the elements of this matrix are the local sensitivity coefficients

.. math::
    \begin{alignat}{3}
    s_{ij} (t_m, u_n) = \frac{\mathrm{d} y_i}{\mathrm{d} p_j}
    \end{alignat}


