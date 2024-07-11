# !/usr/bin/env python3

import numpy as np


def _gqsp_reduce_laurent_polynomials(F, G, theta, phi, tol=1e-8):
    """
    Reduce Laurent polynomials F and G
    by decreasing their degrees by at least 1 using theta and phi.

    Args:
        F (LaurentPolynomial): Laurent polynomial F.
        G (LaurentPolynomial): Laurent polynomial G.
        theta (float): The theta angle for reduction.
        phi (float): The phi angle for reduction.
        tol (float, optional): Tolerance for numerical stability. Default is 1e-8.

    Returns:
        Tuple[LaurentPolynomial, LaurentPolynomial]: Reduced Laurent polynomials F_tilde
                                                     and G_tilde.
    """
    exp = np.exp(-1.j * phi)
    cos = np.cos(theta)
    sin = np.sin(theta)

    F_tilde = (exp * cos * F + sin * G) >> 1
    G_tilde = exp * sin * F - cos * G
    F_tilde = F_tilde.resize(0, F.d_max - 1).trim(tol=tol)
    G_tilde = G_tilde.resize(0, G.d_max - 1).trim(tol=tol)
    return F_tilde, G_tilde


def gqsp_decompose_via_carving(F, G, d_min, d_max, tol=1e-8):
    """
    Decompose Laurent polynomials F and G into phase angles
    for Generalized Quantum Signal Processing (GQSP) via carving algorithm.

    Args:
        F (LaurentPolynomial): Laurent polynomial F.
        G (LaurentPolynomial): Laurent polynomial G.
        d_min (int): Minimum degree.
        d_max (int): Maximum degree.
        tol (float, optional): Tolerance for numerical stability. Default is 1e-8.

    Returns:
        dict: The phase angles dictionary containing:
            - 'd_min' (int): Minimum degree.
            - 'theta' (np.ndarray): List of theta angles.
            - 'phi' (np.ndarray): List of phi angles.
            - 'lambda' (float): Lambda angle.
    """
    theta_list = np.zeros(d_max - d_min + 1, dtype=float)
    phi_list = np.zeros(d_max - d_min + 1, dtype=float)

    F = (F >> F.d_min).trim(tol=tol)
    G = (G >> G.d_min).trim(tol=tol)

    for i in range(d_max - d_min, -1, -1):
        F_leading_coef = F[F.d_nonzero_max]
        G_leading_coef = G[G.d_nonzero_max]
        if F.d_nonzero_max == 0 and G.d_nonzero_max == 0:
            break
        elif i < 0:
            raise ValueError

        if F.is_almost_zero(tol=tol):
            theta_list[i] = 0
            phi_list[i] = -np.angle(G_leading_coef)
        elif G.is_almost_zero(tol=tol):
            theta_list[i] = np.pi / 2
            phi_list[i] = np.angle(F_leading_coef)
        elif F.d_nonzero_max > G.d_nonzero_max:
            theta_list[i] = 0
            phi_list[i] = np.angle(F_leading_coef)
        elif F.d_nonzero_max < G.d_nonzero_max:
            theta_list[i] = np.pi
            phi_list[i] = -np.angle(G_leading_coef)
        else:
            theta_list[i] = np.arctan2(np.abs(G_leading_coef), np.abs(F_leading_coef))
            phi_list[i] = np.angle(F_leading_coef) - np.angle(G_leading_coef)

        F, G = _gqsp_reduce_laurent_polynomials(F, G, theta_list[i], phi_list[i], tol=tol)

    F_leading_coef = F[F.d_nonzero_max]
    G_leading_coef = G[G.d_nonzero_max]
    theta_list[i] = np.arctan2(np.abs(G_leading_coef), np.abs(F_leading_coef))
    lambda_ = np.angle(G_leading_coef)
    phi_list[i] = np.angle(F_leading_coef) - lambda_

    if i > 0:
        theta_list[0] = -np.pi / 2
        theta_list[1:i] = np.pi
        theta_list[i] -= np.pi / 2
        phi_list[0] = 0
        phi_list[1:i] = np.pi

    phase_angles = dict()
    phase_angles['d_min'] = d_min
    phase_angles['theta'] = theta_list
    phase_angles['phi'] = phi_list
    phase_angles['lambda'] = lambda_
    return phase_angles
