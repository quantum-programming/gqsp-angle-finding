# !/usr/bin/env python3

import warnings
import numpy as np
from numpy.fft import fft, ifft
from scipy.linalg import hankel
from numpy.linalg import svd

from utils import is_zero, ANGLE_SAMPLES
from laurent_polynomial import LaurentPolynomial


def gqsp_complete_via_root_finding(F, seed=None, tol=1e-8):
    """
    Compute the completion part of the phase angle finding algorithm 
    for Generalized Quantum Signal Processing (GQSP) via root finding.

    Args:
        F (LaurentPolynomial): Input Laurent polynomial F(w).
        seed (list, optional): Seed values for controlling computation, default is None.
        tol (float, optional): Tolerance value for numerical stability, default is 1e-8.

    Returns:
        LaurentPolynomial: The resulting Laurent polynomial G(w) after the completion part.
    """
    if seed is None:
        seed = [0] * (len(F) * 2)

    completion_poly = 1 - F * ~F
    roots = completion_poly.roots()
    i = 0
    fft_size = 1 << (len(F) - 1).bit_length()
    G_fft = fft([1], fft_size)
    G_degree = 0
    while i < len(roots):
        s = roots[i]
        if np.abs(s) <= 1 + tol:
            if seed[i] == 1:
                s = 1 / np.conj(s)
            G_fft *= fft([- s, 1], fft_size)
            G_degree += 1
            if is_zero(np.abs(s) - 1, tol=tol):
                warnings.warn("Potential instability due to roots too close to the unit circle"
                              "in the complex plane. Please reconsider the normalization factor.",
                              UserWarning)
                i += 1
        i += 1

    G_coef = ifft(G_fft, fft_size)[: G_degree + 1]
    G = LaurentPolynomial(G_coef, F.d_min, F.d_max)
    C = np.sqrt(np.mean(completion_poly.eval(ANGLE_SAMPLES) / (G * ~G).eval(ANGLE_SAMPLES)))
    G *= C
    return G


def gqsp_completion_via_prony(F):
    """
    Compute the completion part of the phase angle finding algorithm 
    for Generalized Quantum Signal Processing (GQSP) via Prony's method.

    Args:
        F (LaurentPolynomial): Input Laurent polynomial F(w).

    Returns:
        LaurentPolynomial: The resulting Laurent polynomial G(w) after the completion part.
    """
    G_degree = len(F) - 1
    fft_size = 1 << (G_degree.bit_length() + 5)

    F_w = F.eval(2 * np.arange(fft_size) * np.pi / fft_size)
    completion_poly_inv_w = 1 / (1 - F_w * np.conj(F_w))

    completion_poly_inv_fft = np.flip(fft(completion_poly_inv_w))
    c = completion_poly_inv_fft[: G_degree + 2]
    r = completion_poly_inv_fft[G_degree + 1 : G_degree + len(F) + 1]
    M = hankel(c, r)
    _, _, vh = svd(M)

    G_coef = np.conj(vh[-1])
    G = LaurentPolynomial(G_coef, F.d_min, F.d_max)
    C = np.sqrt(np.mean((1 - F * ~F).eval(ANGLE_SAMPLES) / (G * ~G).eval(ANGLE_SAMPLES)))
    G *= C
    return G
