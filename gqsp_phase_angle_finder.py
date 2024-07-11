# !/usr/bin/env python3

import numpy as np
from functools import reduce

from abstract_phase_angle_finder import AbstractPhaseAngleFinder
from gqsp_completion import gqsp_complete_via_root_finding, gqsp_completion_via_prony
from gqsp_decomposition import gqsp_decompose_via_carving
from utils import ANGLE_SAMPLES


r"""
In GQSP:
Signal operators, denoted as W_0(w) and W_1(w), are defined as:
- W_0(w) = [[w, 0], [0, 1]]
- W_1(w) = [[1, 0], [0, w^-1]]

Signal processing operators, denoted as R(theta, phi, lambda), are defined as:
- R(\theta, \phi, \lambda) = [[e^(i (\lambda + \phi)) * cos(\theta), e^(i \lambda) * sin(\theta)],
                             [e^(i \phi) * sin(\theta), -cos(\theta)]]

These operators form the basis for constructing the GQSP operation sequence U, where:
U = R(\theta_{d_min}, \phi_{d_min}, \lambda) * W_1(w)
    * \prod_{k=d_{min}+1}^{0} W_1(w) * R(\theta_k, \phi_k, 0)
    * \prod_{k=1}^{d_{max}} W_0(w) * R(\theta_k, \phi_k, 0).
"""


def generate_gqsp_signal_operator(angle, is_inverse):
    """
    Generate a signal operator matrix for GQSP from a given angle.

    Args:
        angle (float or np.ndarray): The angle(s) for the signal operator.
        is_inverse (bool): Flag indicating if the operator is inverse.

    Returns:
        np.ndarray: The signal operator matrix or an array of such matrices.
    """
    w = np.zeros(angle.shape + (2, 2), dtype=complex)
    if is_inverse:
        w[..., 0, 0] = 1
        w[..., 1, 1] = np.exp(-1.j * angle)
    else:
        w[..., 0, 0] = np.exp(1.j * angle)
        w[..., 1, 1] = 1
    return w


def stringify_gqsp_signal_operator(is_inverse):
    """
    Generate a string representation of a GQSP signal operator.

    Args:
        is_inverse (bool): Flag indicating if the operator is inverse.

    Returns:
        str: String representation of the GQSP signal operator.
    """
    w = "W_1(w)" if is_inverse else "W_0(w)"
    return w


def generate_gqsp_signal_processing_operator(theta_list, phi_list, lambda_list):
    """
    Generate signal processing operators for GQSP from lists of theta, phi, and lambda.

    Args:
        theta_list (np.ndarray): List of theta angles.
        phi_list (np.ndarray): List of phi angles.
        lambda_list (np.ndarray): List of lambda angles.

    Returns:
        np.ndarray: Array of signal processing operator matrices.
    """

    s = np.zeros(theta_list.shape + (2, 2), dtype=complex)
    s[..., 0, 0] = np.exp(1.j * (lambda_list + phi_list)) * np.cos(theta_list)
    s[..., 0, 1] = np.exp(1.j * lambda_list) * np.sin(theta_list)
    s[..., 1, 0] = np.exp(1.j * phi_list) * np.sin(theta_list)
    s[..., 1, 1] = -np.cos(theta_list)
    return s


def stringify_gqsp_signal_processing_operator(theta_list, phi_list, lambda_list):
    """
    Generate a string representation of a GQSP signal processing operator.

    Args:
        theta_list (np.ndarray): List of theta angles.
        phi_list (np.ndarray): List of phi angles.
        lambda_list (np.ndarray): List of lambda angles.

    Returns:
        np.ndarray: Array of strings representing the GQSP signal processing operators.
    """
    theta_list_str = theta_list.astype(str)
    phi_list_str = phi_list.astype(str)
    lambda_list_str = lambda_list.astype(str)
    s = reduce(np.char.add, ["R(", theta_list_str, ",", phi_list_str, ",", lambda_list_str, ")"])
    return s


def compose_gqsp_operation_sequence(phase_angles, angle):
    """
    Compose a unitary matrix (GQSP operation sequence) from phase angles for given angles.

    Args:
        phase_angles (dict): phase angles required for computation:
            - 'd_min' (int): Minimum degree.
            - 'theta' (np.ndarray): List of theta angles.
            - 'phi' (np.ndarray): List of phi angles.
            - 'lambda' (float): Lambda angle.
        angle (float or np.ndarray): The angle(s) for the unitary composition.

    Returns:
        np.ndarray: The composed unitary matrix.
    """
    d_min = phase_angles['d_min']
    theta_list = phase_angles['theta']
    phi_list = phase_angles['phi']
    lambda_ = phase_angles['lambda']
    if d_min > 0:
        raise ValueError("Minimum degree must be 0 or negative.")

    lambda_list = np.zeros_like(phi_list, dtype=float)
    if lambda_list.ndim == 0:
        lambda_list[()] = lambda_
    else:
        lambda_list[..., 0] = lambda_
    angle = np.array(angle)
    if len(phi_list) == 0:
        identity = np.eye(2, dtype=complex).reshape((1,) * angle.ndim + (2, 2))
        identity = np.broadcast_to(identity, angle.shape + (2, 2))
        return identity

    W_1 = generate_gqsp_signal_operator(angle, is_inverse=True)
    R = generate_gqsp_signal_processing_operator(theta_list, phi_list, lambda_list)
    U_qsp = reduce(lambda x, y: x @ W_1 @ y, R[: -d_min + 1])
    W_0 = generate_gqsp_signal_operator(angle, is_inverse=False)
    U_qsp = reduce(lambda x, y: x @ W_0 @ y, [U_qsp, *R[-d_min + 1 :]])
    return U_qsp


def stringify_gqsp_operation_sequence(phase_angles):
    """
    Generate a string representation of the GQSP operation sequence based on phase angles.

    Args:
        phase_angles (dict): Dictionary containing phase angles:
            - 'd_min' (int): Minimum degree.
            - 'theta' (np.ndarray): List of theta angles.
            - 'phi' (np.ndarray): List of phi angles.
            - 'lambda' (float): Lambda angle.

    Returns:
        str: String representation of the GQSP operation sequence.
    """
    d_min = phase_angles['d_min']
    theta_list = phase_angles['theta']
    phi_list = phase_angles['phi']
    lambda_ = phase_angles['lambda']
    if d_min > 0:
        raise ValueError("Minimum degree must be 0 or negative.")

    lambda_list = np.zeros_like(phi_list, dtype=float)
    if lambda_list.ndim == 0:
        lambda_list[()] = lambda_
    else:
        lambda_list[..., 0] = lambda_
    if len(phi_list) == 0:
        identity = "Identity"
        return identity

    W_1 = stringify_gqsp_signal_operator(is_inverse=True)
    R = stringify_gqsp_signal_processing_operator(theta_list, phi_list, lambda_list)
    U_qsp = reduce(lambda x, y: "\n * ".join([x, W_1, y]), R[: -d_min + 1])
    W_0 = stringify_gqsp_signal_operator(is_inverse=False)
    U_qsp = reduce(lambda x, y: "\n * ".join([x, W_0, y]), [U_qsp, *R[-d_min + 1 :]])
    return U_qsp


class GQSPPhaseAngleFinder(AbstractPhaseAngleFinder):
    """
    A class for finding phase angles for GQSP.

    Attributes:
        truncate_func (callable): Function to truncate the target function.
        seed (list, optional): Random seed for algorithms.
        tol (float, optional): Tolerance parameter for numerical computations.
    """

    def __init__(self, truncate_func, seed=None, tol=1e-8):
        """
        Initialize the GQSPPhaseAngleFinder instance.

        Args:
            truncate_func (callable): Function to truncate the target function.
            seed (list, optional): Random seed for algorithms. Default is None.
            tol (float, optional): Tolerance parameter for numerical computations. Default is 1e-8.
        """
        self.seed = seed
        self.tol = tol
        super().__init__(truncate_func)

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
        """
        Measure the unitary matrix (GQSP operation sequence) computed from phase angles.

        Args:
            phase_angles (dict): Dictionary containing phase angles:
                - 'd_min' (int): Minimum degree.
                - 'theta' (np.ndarray): List of theta angles.
                - 'phi' (np.ndarray): List of phi angles.
                - 'lambda' (float): Lambda angle.

        Returns:
            np.ndarray: Matrix elements corresponding to F(w).
        """
        U_qsp = compose_gqsp_operation_sequence(phase_angles, ANGLE_SAMPLES)
        F_w = U_qsp[..., 0, 0]
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
            dict or None: Dictionary containing phase angles ('d_min', 'theta', 'phi', 'lambda')
                          or None if decomposition fails.
        """
        try:
            phase_angles = gqsp_decompose_via_carving(F, G, *d, tol=self.tol)
        except np.linalg.LinAlgError:
            return None
        return phase_angles


class GQSPPhaseAngleFinderViaPronyAndCarving(GQSPPhaseAngleFinder):

    def __init__(self, truncate_func, tol=1e-8):
        """
        Initialize the GQSPPhaseAngleFinder instance.

        Args:
            truncate_func (callable): Function to truncate the target function.
            tol (float, optional): Tolerance parameter for numerical computations. Default is 1e-8.
        """
        super().__init__(truncate_func, seed=None, tol=tol)

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
            dict or None: Dictionary containing phase angles ('d_min', 'theta', 'phi', 'lambda')
                          or None if decomposition fails.
        """
        try:
            phase_angles = gqsp_decompose_via_carving(F, G, *d, tol=self.tol)
        except np.linalg.LinAlgError:
            return None
        return phase_angles
