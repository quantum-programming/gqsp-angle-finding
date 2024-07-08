# !/usr/bin/env python3

import numbers
import numpy as np

from utils import is_zero


class LaurentPolynomial():
    """
    A class to represent Laurent polynomials.

    Attributes:
        coef (array): Coefficients of the polynomial.
        d_min (int): Minimum degree of the polynomial.
        d_max (int): Maximum degree of the polynomial.
    """

    def __init__(self, coef, d_min, d_max):
        """
        Initialize a Laurent polynomial.

        Args:
            coef (array or number): Coefficients of the polynomial.
            d_min (int): Minimum degree.
            d_max (int): Maximum degree.

        Raises:
            ValueError: If d_min is greater than d_max.
            ValueError: If coef is not a one-dimensional array
                        or its length is incompatible with the specified range of d_min and d_max.
        """
        if d_min > d_max:
            raise ValueError("Invalid input: 'd_min' should be less than or equal to 'd_max'. "
                             "Please check the values of 'd_min' and 'd_max'.")
        if isinstance(coef, numbers.Number):
            self.__coef = np.array([coef], dtype=complex)
        else:
            self.__coef = np.array(coef, dtype=complex)
        self.__d_min = d_min
        self.__d_max = d_max
        if len(self) == 0:
            self.__is_zero = True
            self.__coef = np.array([0], dtype=complex)
            self.__d_nonzero_max = self.__d_min
        else:
            self.__d_nonzero_max = len(self) + self.__d_min - 1
            if len(self.__coef.shape) != 1 or self.__d_nonzero_max > self.__d_max:
                raise ValueError("Invalid coefficient array: 'coef' must be a one-dimensional "
                                 "array and its length must be compatible with the range "
                                 "specified by 'd_min' and 'd_max'.")

            self.__is_zero = False

    @property
    def coef(self):
        """
        Get the coefficients of the Laurent polynomial.

        Returns:
            numpy.ndarray: The coefficients of the Laurent polynomial.
        """
        return self.__coef

    @property
    def d_min(self):
        """
        Get the minimum degree (d_min) of the Laurent polynomial.

        Returns:
            int: The minimum degree of the Laurent polynomial.
        """
        return self.__d_min

    @property
    def d_nonzero_max(self):
        """
        Get the highest nonzero degree (d_nonzero_max) of the Laurent polynomial.

        Returns:
            int: The highest nonzero degree of the Laurent polynomial.
        """
        return self.__d_nonzero_max

    @property
    def d_max(self):
        """
        Get the maximum degree (d_max) of the Laurent polynomial.

        Returns:
            int: The maximum degree of the Laurent polynomial.
        """
        return self.__d_max

    @d_max.setter
    def d_max(self, d_max):
        """
        Set the maximum degree (d_max) of the Laurent polynomial.

        Args:
            d_max (int): The maximum degree to set.

        Raises:
            ValueError: If the highest nonzero degree of the polynomial exceeds d_max.
        """
        self.__d_max = d_max
        if self.__d_nonzero_max > self.__d_max:
            raise ValueError("Invalid degree range: The highest nonzero degree of the polynomial "
                             "must not exceed 'd_max'. Please adjust the degree range "
                             "accordingly.")

    @property
    def is_zero(self):
        """
        Check if the Laurent polynomial is zero.

        Returns:
            bool: True if the polynomial is zero, False otherwise.
        """
        return self.__is_zero

    def is_almost_zero(self, tol=1e-8):
        """
        Check if the Laurent polynomial is almost zero within a given tolerance.

        Args:
            tol (float, optional): Tolerance value for determining 'almost zero'. Defaults to 1e-8.

        Returns:
            bool: True if the polynomial is almost zero (within the specified tolerance), False otherwise.
        """
        return self.is_zero or self.trim(tol=tol).is_zero

    def copy(self):
        """
        Create a copy of the Laurent polynomial.

        Returns:
            LaurentPolynomial: A copy of the current polynomial.
        """
        return self.__class__(self.__coef.copy(), self.__d_min, self.__d_max)

    def resize(self, d_min, d_max):
        """
        Resize the polynomial to fit within the specified range of degrees.

        Args:
            d_min (int): Minimum degree after resizing.
            d_max (int): Maximum degree after resizing.

        Returns:
            LaurentPolynomial: Resized polynomial within the specified degree range.

        Raises:
            ValueError:
                If d_min is greater than d_max.
                Ensure that 'd_min' is less than or equal to 'd_max' when resizing the polynomial.

        """
        if d_min > d_max:
            raise ValueError("Invalid padding range: 'd_min' should be less than or equal to "
                             "'d_max'. Ensure that 'd_min' and 'd_max' are "
                             "correctly specified.")
        if self.__is_zero or d_max < self.__d_min or d_min > self.__d_nonzero_max:
            coef = np.zeros(d_max - d_min + 1, dtype=complex)
        else:
            coef = self.__coef
            if d_max <= self.__d_nonzero_max:
                coef = coef[: d_max - self.__d_min + 1]
            else:
                coef = np.pad(coef, (0, d_max - self.__d_nonzero_max), 'constant')
            if d_min >= self.__d_min:
                coef = coef[d_min - self.__d_min :]
            else:
                coef = np.pad(coef, (self.__d_min - d_min, 0), 'constant')
        return LaurentPolynomial(coef, d_min, d_max)

    def __getitem__(self, idx):
        """
        Get the coefficient of the term with degree idx.

        Args:
            idx (int): The degree of the term.

        Returns:
            complex: The coefficient of the term with degree idx.
        """
        if idx >= 100:
            raise IndexError
        pos = idx - self.__d_min
        if pos >= 0 and pos < len(self):
            return self.__coef[pos]
        else:
            return 0

    def __len__(self):
        """
        Get the number of coefficients in the Laurent polynomial.

        Returns:
            int: The number of coefficients.
        """
        return len(self.__coef)

    def __repr__(self):
        """
        Return a string representation of the LaurentPolynomial object.

        Returns:
            str: A string representation showing the coefficients,
                 minimum degree, and maximum degree.
        """
        return f"LaurentPolynomial(coef={self.__coef}, d_min={self.__d_min}, d_max={self.__d_max})"

    def __str__(self):
        """
        String representation of the Laurent polynomial.

        Returns:
            str: The string representation of the polynomial.
        """
        indices = range(self.__d_min, self.__d_max + 1)
        return " + ".join([f"{self[i]} * w^({i})" for i in indices])

    def __array__(self):
        """
        Convert the LaurentPolynomial object into a numpy array.

        Returns:
            numpy.ndarray: A numpy array containing the LaurentPolynomial object.
        """
        array = np.array(None, dtype=object)
        array[()] = self
        return array

    def __add__(self, other):
        """
        Add another Laurent polynomial or a number to this polynomial.

        Args:
            other (LaurentPolynomial or number): The polynomial or number to add.

        Returns:
            LaurentPolynomial: The resulting polynomial after addition.

        Raises:
            TypeError: If other is not a LaurentPolynomial or a number.
        """
        if isinstance(other, numbers.Number):
            return self + self.__class__(other, 0, 0)
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.__is_zero:
            return other.copy()
        if other.__is_zero:
            return self.copy()
        d_min = min(self.__d_min, other.d_min)
        d_nonzero_max = max(self.__d_nonzero_max, other.d_nonzero_max)
        d_max = max(self.__d_max, other.d_max)
        coef = self.resize(d_min, d_nonzero_max).coef + other.resize(d_min, d_nonzero_max).coef
        return self.__class__(coef, d_min, d_max)

    def __radd__(self, other):
        """
        Reverse addition.

        Args:
            other (LaurentPolynomial or number): The polynomial or number to add.

        Returns:
            LaurentPolynomial: The resulting polynomial after addition.
        """
        return self + other

    def __neg__(self):
        """
        Negate the Laurent polynomial.

        Returns:
            LaurentPolynomial: The negated polynomial.
        """
        return self.__class__(-self.__coef, self.__d_min, self.__d_max)

    def __mul__(self, other):
        """
        Multiply by another Laurent polynomial or a number.

        Args:
            other (LaurentPolynomial or number): The polynomial or number to multiply by.

        Returns:
            LaurentPolynomial: The resulting polynomial after multiplication.

        Raises:
            TypeError: If other is not a LaurentPolynomial or a number.
        """
        if isinstance(other, numbers.Number):
            return self.__class__(other * self.__coef, self.__d_min, self.__d_max)
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.__is_zero or other.__is_zero:
            return self.__class__([], self.__d_min, self.__d_max)
        coef = np.convolve(self.__coef, other.coef)
        return self.__class__(coef, self.__d_min + other.d_min, self.__d_max + other.d_max)

    def __rmul__(self, other):
        """
        Reverse multiplication.

        Args:
            other (LaurentPolynomial or number): The polynomial or number to multiply by.

        Returns:
            LaurentPolynomial: The resulting polynomial after multiplication.
        """
        if isinstance(other, numbers.Number):
            return self.__class__(other * self.__coef, self.__d_min, self.__d_max)
        if not isinstance(other, self.__class__):
            return NotImplemented
        else:
            return self * other

    def __sub__(self, other):
        """
        Subtract another Laurent polynomial or a number.

        Args:
            other (LaurentPolynomial or number): The polynomial or number to subtract.

        Returns:
            LaurentPolynomial: The resulting polynomial after subtraction.
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Reverse subtraction.

        Args:
            other (LaurentPolynomial or number): The polynomial or number to subtract.

        Returns:
            LaurentPolynomial: The resulting polynomial after subtraction.
        """
        return other + (-self)

    def __ilshift__(self, d):
        """
        In-place left shift of the polynomial degrees.

        Args:
            d (int): Amount to shift the degrees by.

        Returns:
            LaurentPolynomial: The resulting polynomial after in-place left shift.
        """
        self.__d_min += d
        self.__d_max += d
        self.__d_nonzero_max += d
        return self

    def __lshift__(self, d):
        """
        Left shift of the polynomial degrees.

        Args:
            d (int): Amount to shift the degrees by.

        Returns:
            LaurentPolynomial: The resulting polynomial after left shift.
        """
        copy = self.copy()
        copy <<= d
        return copy

    def __irshift__(self, d):
        """
        In-place right shift of the polynomial degrees.

        Args:
            d (int): Amount to shift the degrees by.

        Returns:
            LaurentPolynomial: The resulting polynomial after in-place right shift.
        """
        self.__d_min -= d
        self.__d_max -= d
        self.__d_nonzero_max -= d
        return self

    def __rshift__(self, d):
        """
        Right shift of the polynomial degrees.

        Args:
            d (int): Amount to shift the degrees by.

        Returns:
            LaurentPolynomial: The resulting polynomial after right shift.
        """
        copy = self.copy()
        copy >>= d
        return copy

    def recip(self):
        """
        Return the reciprocal of the Laurent polynomial.

        Returns:
            LaurentPolynomial: The reciprocal polynomial, obtained by reversing the coefficients.

        Notes:
            The reciprocal of a Laurent polynomial f(w) is obtained by reversing its coefficients.
            Mathematically, it represents f(w) -> f(w^-1).
        """
        coef = self.resize(self.__d_min, self.__d_max).coef[::-1]
        return self.__class__(coef, -self.__d_max, -self.__d_min)

    def conj(self):
        """
        Complex conjugate of the Laurent polynomial.

        Returns:
            LaurentPolynomial: The complex conjugated polynomial.
        """
        return self.__class__(np.conj(self.__coef), self.__d_min, self.__d_max)

    def __invert__(self):
        """
        Return the conjugate reciprocal of the Laurent polynomial.

        Returns:
            LaurentPolynomial: The conjugate reciprocal polynomial, obtained by reversing
            the coefficients and taking the complex conjugate.

        Notes:
            The conjugate reciprocal of a Laurent polynomial f(w) is obtained by reversing its
            coefficients and then taking the complex conjugate of each coefficient.
            Mathematically, it represents f(w) -> f^*(w^-1).
        """
        return self.recip().conj()

    def eval(self, angles):
        """
        Evaluate the polynomial at given angles.

        Args:
            angles (array): Angles at which to evaluate the polynomial.

        Returns:
            array: Evaluated values of the polynomial.
        """
        if self.__is_zero:
            return np.zeros_like(angles, dtype=complex)
        indices = range(self.__d_min, self.__d_nonzero_max + 1)
        exp_i_angle = np.exp(1.j * np.multiply.outer(angles, indices))
        return np.array(exp_i_angle @ self.__coef)

    def trim(self, tol=1e-8):
        """
        Trim the Laurent polynomial by removing zero coefficients at the higher degrees.

        Args:
            tol (float, optional):
                Tolerance for considering a coefficient as zero. Default is 1e-8.

        Returns:
            LaurentPolynomial: Trimmed polynomial where coefficients at higher degrees are zero.

        Notes:
            The trimming process adjusts the degree range such that the resulting polynomial has
            non-zero coefficients between `d_min` and `d_nonzero_max_new`.
            If all coefficients are zero within the tolerance `tol`, returns a zero polynomial.
        """

        d_nonzero_max_new = self.__d_nonzero_max
        while d_nonzero_max_new >= self.__d_min:
            if not is_zero(self[d_nonzero_max_new], tol=tol):
                break
            d_nonzero_max_new -= 1

        if d_nonzero_max_new < self.__d_min:
            return self.__class__([], 0, 0)
        return self.resize(self.__d_min, d_nonzero_max_new)

    def roots(self):
        """
        Compute the roots of the Laurent polynomial.

        Returns:
            numpy.ndarray: Array containing the roots of the polynomial.
        """
        return np.roots(np.flip(self.__coef))
