# !/usr/bin/env python3

import numpy as np
from scipy.special import jv

from laurent_polynomial import LaurentPolynomial


def truncate_exp(d, tau):
    """
    Truncate an exponential function of the form exp(-1.j * tau * cos(theta))
    using Chebyshev approximation.

    Args:
        d (int): Degree of the Laurent polynomial.
        tau (float): Parameter for the exponential function.

    Returns:
        tuple: A tuple containing the truncated Laurent polynomial and the exponential function.
            - LaurentPolynomial: Truncated Laurent polynomial.
            - function: Exponential function evaluable at given angles.
    """
    f_cheb = np.zeros(d + 1, dtype=complex)
    for k in range(d + 1):
        f_cheb[k] = 2 * jv(k, tau) * [1, - 1.j, -1, 1.j][k % 4]
    f_cheb[0] /= 2
    f_w_coef = np.hstack((f_cheb[-1:0:-1] / 2, f_cheb[0], f_cheb[1:] / 2))
    f_w = LaurentPolynomial(f_w_coef, -d, d)
    return f_w, lambda theta: np.exp(-1.j * tau * np.cos(theta))


def truncate_cos(d, cos_scale, noise_scale):
    """
    Truncate a cosine function of the form cos_scale * cos(d * theta) + noise.

    Args:
        d (int): Degree of the Laurent polynomial.
        cos_scale (float): Scaling factor for the coefficients.
        noise_scale (float): Amplitude of random noise.

    Returns:
        tuple: A tuple containing the truncated Laurent polynomial and the cosine function.
            - LaurentPolynomial: Truncated Laurent polynomial.
            - function: Cosine function evaluable at given angles.
    """
    f_w_coef = np.random.rand(d * 2 + 1) * noise_scale
    f_w_coef[0] += cos_scale
    f_w_coef[-1] += cos_scale
    f_w = LaurentPolynomial(f_w_coef, -d, d)
    return f_w, lambda theta: f_w.eval(theta)


def truncate_exp_plus_cos(d, tau, cos_scale, noise_scale):
    """
    Truncate a combination of exponential and cosine functions.

    Args:
        d (int): Degree of the Laurent polynomials.
        tau (float): Parameter for the exponential function.
        cos_scale (float): Scaling factor for the cosine function coefficients.
        noise_scale (float): Amplitude of random noise for the cosine function.

    Returns:
        tuple:
            A tuple containing the combined truncated Laurent polynomial and the combined function.
            - LaurentPolynomial: Combined truncated Laurent polynomial.
            - function: Combined function evaluable at given angles.
    """
    p_exp, exp = truncate_exp(d, tau)
    p_cos, cos = truncate_cos(d, cos_scale, noise_scale)
    return p_exp + p_cos, lambda theta: exp(theta) + cos(theta)
