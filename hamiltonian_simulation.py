# !/usr/bin/env python3

import numpy as np
from scipy.special import jv

from laurent_polynomial import LaurentPolynomial
from gqsp_phase_angle_finder import GQSPPhaseAngleFinderViaRootFindingAndCarving
from gqsp_phase_angle_finder import GQSPPhaseAngleFinderViaPronyAndCarving


def truncate_exp(d, tau):
    """
    Truncate an exponential function of the form scale * exp(-1.j * tau * cos(theta))
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


def truncate_cos(d, scale, eps):
    """
    Truncate a cosine function of the form scale * cos(d * theta) + noise.

    Args:
        d (int): Degree of the Laurent polynomial.
        scale (float): Scaling factor for the coefficients.
        eps (float): Amplitude of random noise.

    Returns:
        tuple: A tuple containing the truncated Laurent polynomial and the cosine function.
            - LaurentPolynomial: Truncated Laurent polynomial.
            - function: Cosine function evaluable at given angles.
    """
    f_w_coef = np.random.rand(d * 2 + 1) * eps
    f_w_coef[0] += 1 / 2
    f_w_coef[-1] += 1 / 2
    f_w_coef *= scale
    f_w = LaurentPolynomial(f_w_coef, -d, d)
    return f_w, lambda theta: f_w.eval(theta)


def truncate_exp_plus_cos(d, tau, scale, eps):
    """
    Truncate a combination of exponential and cosine functions.

    Args:
        d (int): Degree of the Laurent polynomials.
        tau (float): Parameter for the exponential function.
        scale (float): Scaling factor for the cosine function coefficients.
        eps (float): Amplitude of random noise for the cosine function.

    Returns:
        tuple:
            A tuple containing the combined truncated Laurent polynomial and the combined function.
            - LaurentPolynomial: Combined truncated Laurent polynomial.
            - function: Combined function evaluable at given angles.
    """
    p_exp, exp = truncate_exp(d, tau)
    p_cos, cos = truncate_cos(d, scale, eps)
    return p_exp + p_cos, lambda theta: exp(theta) + cos(theta)


if __name__ == '__main__':
    tau = 10.0
    d = 40
    scale = 1 / 2
    eps = 1e-3

    def truncate_func1(d):
        return truncate_exp_plus_cos(d[1], tau=tau, scale=scale, eps=eps)

    def truncate_func2(d):
        return truncate_cos(d[1], scale=scale, eps=eps)

    solver = GQSPPhaseAngleFinderViaRootFindingAndCarving(truncate_func=truncate_func1)
    solver.angle_finding(d=(-d, d), scale=0.5, measure_error=True)
    print(solver.info)
    solver = GQSPPhaseAngleFinderViaRootFindingAndCarving(truncate_func=truncate_func2)
    solver.angle_finding(d=(-d, d), scale=0.5, measure_error=True)
    print(solver.info)

    solver = GQSPPhaseAngleFinderViaPronyAndCarving(truncate_func=truncate_func1)
    solver.angle_finding(d=(-d, d), scale=0.5, measure_error=True)
    print(solver.info)
    solver = GQSPPhaseAngleFinderViaPronyAndCarving(truncate_func=truncate_func2)
    solver.angle_finding(d=(-d, d), scale=0.5, measure_error=True)
    print(solver.info)
