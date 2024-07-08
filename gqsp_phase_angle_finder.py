# !/usr/bin/env python3

import numpy as np
from functools import reduce

from abstract_phase_angle_finder import AbstractPhaseAngleFinder
from gqsp_completion import gqsp_complete_via_root_finding, gqsp_completion_via_prony
from gqsp_decomposition import gqsp_decompose_via_carving
from utils import ANGLE_SAMPLES


def gqsp_generate_signal_operator(angle):
    """
    Generate a signal operator for GQSP from a given angle or array of angles.

    Args:
        angle (float or np.ndarray): The angle(s) for the signal operator.

    Returns:
        np.ndarray: The signal operator matrix or an array of such matrices.
    """
    w = np.zeros(angle.shape + (2, 2), dtype=complex)
    w[..., 0, 0] = np.exp(1.j * angle)
    w[..., 1, 1] = 1
    return w


def gqsp_generate_signal_processing_operator(theta_list, phi_list, lambda_):
    """
    Generate signal processing operators for GQSP from lists of theta, phi, and lambda.

    Args:
        theta_list (np.ndarray): List of theta angles.
        phi_list (np.ndarray): List of phi angles.
        lambda_ (float): Lambda angle.

    Returns:
        np.ndarray: Array of signal processing operator matrices.
    """
    lambda_list = np.zeros_like(phi_list, dtype=float)
    if lambda_list.ndim == 0:
        lambda_list[()] = lambda_
    else:
        lambda_list[..., 0] = lambda_

    s = np.zeros(theta_list.shape + (2, 2), dtype=complex)
    s[..., 0, 0] = np.exp(1.j * (lambda_list + phi_list)) * np.cos(theta_list)
    s[..., 0, 1] = np.exp(1.j * lambda_list) * np.sin(theta_list)
    s[..., 1, 0] = np.exp(1.j * phi_list) * np.sin(theta_list)
    s[..., 1, 1] = -np.cos(theta_list)
    return s


def gqsp_compose_unitary_matrix(d_min, theta_list, phi_list, lambda_, angle):
    """
    Compose a unitary matrix from phase angles for given angles in GQSP.

    Args:
        d_min (int): Minimum degree.
        theta_list (list of float): List of theta angles.
        phi_list (list of float): List of phi angles.
        lambda_ (float): Lambda angle.
        angle (float or np.ndarray): The angle(s) for the unitary composition.

    Returns:
        np.ndarray: The composed unitary matrix.
    """
    angle = np.array(angle)
    if len(phi_list) == 0:
        identity = np.eye(2, dtype=complex).reshape((1,) * angle.ndim + (2, 2))
        identity = np.broadcast_to(identity, angle.shape + (2, 2))
        return identity

    w = gqsp_generate_signal_operator(angle)
    s = gqsp_generate_signal_processing_operator(theta_list, phi_list, lambda_)
    u_phi = reduce(lambda x, y: x @ w @ y, s)
    u_phi *= np.exp(1.j * angle * d_min).reshape(angle.shape + (1, 1))
    return u_phi


class GQSPPhaseAngleFinder(AbstractPhaseAngleFinder):
    """
    A class for finding phase angles for GQSP.

    Attributes:
        truncate_func (callable): Function to truncate the target function.
        seed (int or None): Random seed for algorithms.
        tol (float): Tolerance parameter for numerical computations.
    """

    def __init__(self, seed=None, tol=1e-8, **kwargs):
        """
        Initialize the GQSPPhaseAngleFinder instance.

        Args:
            seed (int or None): Random seed for algorithms.
            tol (float): Tolerance parameter for numerical computations.
            **kwargs: Additional keyword arguments.
                      'truncate_func': Function to truncate the target function.
        """
        self.truncate_func = kwargs['truncate_func']
        self.seed = seed
        self.tol = tol
        super().__init__()

    def depth(self, d):
        """
        Calculate the depth or degree range.

        Args:
            d (Tuple[int, int]): Tuple containing minimum and maximum degrees.

        Returns:
            int: Depth or degree range.
        """
        return d[1] - d[0]

    def measure(self, phase_angles):
        U_phi = gqsp_compose_unitary_matrix(*phase_angles, ANGLE_SAMPLES)
        F_w = U_phi[..., 0, 0]
        return F_w


class GQSPPhaseAngleFinderViaRootFindingAndCarving(GQSPPhaseAngleFinder):
    def complete(self, F):
        """
        Compute the completion part via root finding.

        Args:
            F (LaurentPolynomial): Input Laurent polynomial.

        Returns:
            LaurentPolynomial or None:
                The resulting Laurent polynomial or None if completion fails.
        """
        try:
            G = gqsp_complete_via_root_finding(F, seed=self.seed, tol=self.tol)
        except np.linalg.LinAlgError:
            return None
        return G

    def decompose(self, F, G, d):
        """
        Compute the decomposition part via carving algorithm.

        Args:
            F (LaurentPolynomial): First input Laurent polynomial.
            G (LaurentPolynomial): Second input Laurent polynomial.
            d (Tuple[int, int]): Tuple containing minimum and maximum degrees.

        Returns:
            tuple or None: Phase angles or None if decomposition fails.
        """
        try:
            phase_angles = gqsp_decompose_via_carving(F, G, *d, tol=self.tol)
        except np.linalg.LinAlgError:
            return None
        return phase_angles


class GQSPPhaseAngleFinderViaPronyAndCarving(GQSPPhaseAngleFinder):
    def complete(self, F):
        """
        Compute the completion part via Prony's method.

        Args:
            F (LaurentPolynomial): Input Laurent polynomial.

        Returns:
            LaurentPolynomial or None:
                The resulting Laurent polynomial or None if completion fails.
        """
        try:
            G = gqsp_completion_via_prony(F)
        except np.linalg.LinAlgError:
            return None
        return G

    def decompose(self, F, G, d):
        """
        Compute the decomposition part via carving algorithm.

        Args:
            F (LaurentPolynomial): First input Laurent polynomial.
            G (LaurentPolynomial): Second input Laurent polynomial.
            d (Tuple[int, int]): Tuple containing minimum and maximum degrees.

        Returns:
            tuple or None: Phase angles or None if decomposition fails.
        """
        try:
            phase_angles = gqsp_decompose_via_carving(F, G, *d, tol=self.tol)
        except np.linalg.LinAlgError:
            return None
        return phase_angles
