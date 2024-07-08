# !/usr/bin/env python3

import numpy as np

ANGLE_SAMPLES = np.linspace(- np.pi, np.pi, 1000)
X_FOR_EVAL = np.cos(ANGLE_SAMPLES)
W_FOR_EVAL = np.exp(1.j * ANGLE_SAMPLES)


def is_zero(x, tol=1e-8):
    """
    Check if a number or array of numbers is close to zero within a given tolerance.

    Args:
        x (float or np.ndarray): Input number or array.
        tol (float, optional): Tolerance value. Default is 1e-8.

    Returns:
        bool or np.ndarray: True if input is within tolerance of zero, False otherwise.
    """
    return np.abs(x) < tol


def abs_max(x):
    """
    Compute the maximum absolute value of an array.

    Args:
        x (np.ndarray): Input array.

    Returns:
        float: Maximum absolute value of the input array.
    """
    return np.max(np.abs(x))
