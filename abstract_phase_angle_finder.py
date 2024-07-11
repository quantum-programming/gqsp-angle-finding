# !/usr/bin/env python3

from abc import ABC, abstractmethod
from collections import defaultdict
import time
import numpy as np
from utils import ANGLE_SAMPLES, abs_max


class AbstractPhaseAngleFinder(ABC):
    """
    Abstract base class for phase angle finding algorithms.

    Attributes:
        truncate_func (callable): Function to truncate the target function.
        info (defaultdict): Default dictionary for storing algorithm information.
    """

    def __init__(self, truncate_func):
        """
        Initializes the AbstractPhaseAngleFinder instance with a defaultdict for info.

        Args:
            truncate_func (callable): Function to truncate the target function.
        """
        self.truncate_func = truncate_func
        self.info = defaultdict(lambda: np.nan)

    @abstractmethod
    def depth(self, d):
        """
        Abstract method to calculate the depth or degree range.

        Args:
            d (Tuple[int, int] or int): Tuple containing minimum and maximum degrees
                                        or maximum degree as an integer.

        Returns:
            int: Depth or degree range.
        """
        pass

    @abstractmethod
    def measure(self, phase_angles):
        """
        Abstract method to measure the unitary matrix (QSP operation sequence) computed
        from phase angles.

        Args:
            phase_angles (dict): phase angles required for computation.

        Returns:
            np.ndarray: Matrix elements corresponding to F(w).
        """
        pass

    def truncate(self, d):
        """
        Truncate a target function using a specified degree.

        Args:
            d (Tuple[int, int] or int): Tuple containing minimum and maximum degrees
                                        or maximum degree as an integer.

        Returns:
            Tuple[Polynomial, function]: Tuple containing truncated function and target function.
        """
        return self.truncate_func(d)

    def truncation_error(self, trun_f, target_f):
        """
        Calculates the truncation error between truncated and target functions.

        Args:
            trun_f (Polynomial): Truncated function.
            target_f (function): Target function.

        Returns:
            float: Truncation error.

        Notes:
            Truncation error is defined as ||trun_f(w) - target_f(w)||_inf.
        """
        trun_f_w = trun_f.eval(ANGLE_SAMPLES)
        target_f_w = target_f(ANGLE_SAMPLES)
        return abs_max(trun_f_w - target_f_w)

    @abstractmethod
    def complete(self, F):
        """
        Abstract method to compute the completion part.

        Args:
            F (Polynomial): Input polynomial.

        Returns:
            Polynomial or None: The resulting polynomial or None if completion fails.
        """
        pass

    def comp_error(self, F, G):
        """
        Calculates the completion error between F, G, and the identity operator.

        Args:
            F (Polynomial): First input polynomial.
            G (Polynomial): Second input polynomial.

        Returns:
            float: Completion error.

        Notes:
            Completion error is defined as ||1 - F(w)F^*(w^-1) - G(w)G^*(w^-1)||_inf.
        """
        res_w = (1 - F * ~F - G * ~G).eval(ANGLE_SAMPLES)
        return abs_max(res_w)

    @abstractmethod
    def decompose(self, F, G, d):
        """
        Abstract method to compute the decomposition part.

        Args:
            F (Polynomial): First input polynomial.
            G (Polynomial): Second input polynomial.
            d (Tuple[int, int] or int): Tuple containing minimum and maximum degrees
                                        or maximum degree as an integer.

        Returns:
            dict or None: Dictionary containing phase angles or None if decomposition fails.
        """
        pass

    def angle_finding(self, d, scale, measure_error=False):
        """
        Main method to find phase angles based on truncation, completion, and decomposition.

        Args:
            d (Tuple[int, int] or int): Tuple containing minimum and maximum degrees
                                        or maximum degree as an integer.
            scale (float): Scaling factor.
            measure_error (bool, optional): Flag to measure errors. Default is False.

        Returns:
            dict or None: Dictionary containing phase angles or None if completion
                          or decomposition fails.
        """
        self.info = defaultdict(lambda: np.nan)
        if measure_error:
            start_time = time.perf_counter()

        trun_f, target_f = self.truncate(d)
        if measure_error:
            self.info['truncation_error'] = self.truncation_error(trun_f, target_f)
        F = scale * trun_f

        G = self.complete(F)
        if measure_error:
            self.info['completion_error'] = self.comp_error(F, G)
        if G is None:
            if measure_error:
                end_time = time.perf_counter()
                self.info['running_time'] = end_time - start_time
            return None

        phase_angles = self.decompose(F, G, d)
        if phase_angles is None:
            if measure_error:
                end_time = time.perf_counter()
                self.info['running_time'] = end_time - start_time
            return None

        if measure_error:
            tmp = self.angle_finding_error(trun_f, phase_angles, scale)
            self.info['angle_finding_error'] = tmp
            self.info['total_error'] = self.total_error(target_f, phase_angles, scale)
            tmp = self.info['truncation_error'] / self.info['total_error']
            self.info['truncation_error_per_total'] = tmp
            tmp = self.info['angle_finding_error'] / self.info['total_error']
            self.info['angle_finding_error_per_total'] = tmp
            end_time = time.perf_counter()
            self.info['running_time'] = end_time - start_time

        return phase_angles

    def angle_finding_error(self, trun_f, phase_angles, scale):
        """
        Calculates the angle finding error based on truncated function and phase angles.

        Args:
            trun_f (Polynomial): Truncated function.
            phase_angles (dict): phase angles required for computation.
            scale (float): Scaling factor.

        Returns:
            float: Angle finding error.

        Notes:
            Angle finding error is defined as ||(1/scale)*f_qsp(w) - trun_f(w)||_inf.
        """
        trun_f_w = trun_f.eval(ANGLE_SAMPLES)
        U_qsp_w = self.measure(phase_angles)
        return abs_max(1 / scale * U_qsp_w - trun_f_w)

    def total_error(self, target_f, phase_angles, scale):
        """
        Calculates the total error based on target function and phase angles.

        Args:
            target_f (function): Target function.
            phase_angles (dict): phase angles required for computation.
            scale (float): Scaling factor.

        Returns:
            float: Total error.

        Notes:
            Total error is defined as ||(1/scale)*f_qsp(w) - target_f(w)||_inf.
        """
        target_f_w = target_f(ANGLE_SAMPLES)
        U_qsp_w = self.measure(phase_angles)
        return abs_max(1 / scale * U_qsp_w - target_f_w)
