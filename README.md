# Generalized Quantum Signal Processing (GQSP)

This repository contains algorithms for angle finding in Generalized Quantum Signal Processing (GQSP).

## Definition of GQSP:

### Signal Operator
In GQSP, the following signal operators are defined:

$$ W_0(w) = \begin{pmatrix} w & 0 \\\ 0 & 1 \end{pmatrix}, $$

$$ W_1(w) = \begin{pmatrix} 1 & 0 \\\ 0 & w^{-1} \end{pmatrix}. $$

### Signal Processing Operator
The signal processing operator $R(\theta, \phi, \lambda)$ is defined as:

$$ R(\theta, \phi, \lambda) = \begin{pmatrix} e^{i(\lambda + \phi)} \cos(\theta) & e^{i\lambda} \sin(\theta) \\\ e^{i\phi} \sin(\theta) & -\cos(\theta) \end{pmatrix}. $$

### QSP Operation Sequence
The QSP operation sequence $U_{\Theta \Phi \lambda}$ is represented as:

$$ U_{\Theta \Phi \lambda} = R(\theta_{d_\text{min}}, \phi_{d_\text{min}}, \lambda) * W_1(w) * \prod_{k=d_\text{min}+1}^{0} W_1(w) * R(\theta_k, \phi_k, 0) * \prod_{k=1}^{d_\text{max}} W_0(w) * R(\theta_k, \phi_k, 0). $$

### Angle Finding Process

1. **Initialization**:
   - Initializes the angle finder instance and sets up necessary data structures.

2. **Truncation**:
   - Truncates the target function using a specified degree range.

3. **Completion**:
   - Computes the completion part of the polynomial using methods such as Prony's method or root finding.

4. **Decomposition**:
   - Decomposes the polynomial into phase angles using the completed polynomial.

5. **Error Calculation**:
   - Measures various errors including truncation error, completion error, angle finding error, and total error.

### AbstractPhaseAngleFinder Class

AbstractPhaseAngleFinder is an abstract base class defining methods for angle finding algorithms:

- `depth(d)`: Calculates the depth or degree range. Takes a tuple containing minimum and maximum degrees or maximum degree as an integer and returns the depth or degree range.

- `measure(phase_angles)`: Abstract method to measure the unitary matrix (QSP operation sequence) computed from phase angles. Takes a dictionary containing phase angles ('d_min', 'theta', 'phi', 'lambda') and returns matrix elements corresponding to F(w).

- `truncate(d)`: Truncates a target function using a specified degree. Takes a tuple containing minimum and maximum degrees or maximum degree as an integer and returns a tuple containing truncated function and target function.

- `truncation_error(trun_f, target_f)`: Calculates the truncation error between truncated and target functions. Takes a truncated function (`trun_f`) and a target function (`target_f`) and returns the truncation error.

- `complete(F)`: Abstract method to compute the completion part. Takes an input polynomial (`F`) and returns the resulting polynomial or `None` if completion fails.

- `comp_error(F, G)`: Calculates the completion error between `F`, `G`, and the identity operator. Takes two input polynomials (`F`, `G`) and returns the completion error.

- `decompose(F, G, d)`: Abstract method to compute the decomposition part. Takes two input polynomials (`F`, `G`) and a tuple containing minimum and maximum degrees or maximum degree as an integer (`d`), and returns a dictionary containing phase angles ('d_min', 'theta', 'phi', 'lambda') or `None` if decomposition fails.

- `angle_finding(d, scale, measure_error=False)`: Main method to find phase angles based on truncation, completion, and decomposition. Takes a tuple containing minimum and maximum degrees or maximum degree as an integer (`d`), a scaling factor (`scale`), and an optional flag (`measure_error`) to measure errors. Returns a dictionary containing phase angles or `None` if completion or decomposition fails.

- `angle_finding_error(trun_f, phase_angles, scale)`: Calculates the angle finding error based on truncated function and phase angles. Takes a truncated function (`trun_f`), a dictionary containing phase angles ('d_min', 'theta', 'phi', 'lambda') (`phase_angles`), and a scaling factor (`scale`) and returns the angle finding error.

- `total_error(target_f, phase_angles, scale)`: Calculates the total error based on target function and phase angles. Takes a target function (`target_f`), a dictionary containing phase angles ('d_min', 'theta', 'phi', 'lambda') (`phase_angles`), and a scaling factor (`scale`) and returns the total error.

## Description of main.py

`main.py` implements the GQSP angle finding algorithm using different completion methods.

- `GQSPPhaseAngleFinderViaRootFindingAndCarving`: Implements angle finding using root finding and carving methods.

  - **Initialization**: Initializes the instance with a truncation function, optional random seed, and tolerance for numerical computations. The `truncate_func` is used to truncate the target function, `seed` is used for random number generation (default is `None`), and `tol` is the tolerance for numerical calculations (default is `1e-8`).

- `GQSPPhaseAngleFinderViaPronyAndCarving`: Implements angle finding using Prony's method and carving methods.

  - **Initialization**: Initializes the instance with a truncation function and tolerance for numerical computations. The `truncate_func` is used to truncate the target function, and `tol` is the tolerance for numerical calculations (default is `1e-8`).

The `angle_finding` method returns phase angles in the form of a dictionary containing 'd_min', 'theta', 'phi', 'lambda'. The `info` attribute includes various errors and execution time.
The function `compose_gqsp_operation_sequence` generates the GQSP operator sequence evaluated at angles.
Utilizing `stringify_gqsp_operation_sequence`, you can obtain a string representation of the GQSP operator sequence.

## Reference

- Shuntaro Yamamoto and Nobuyuki Yoshioka. "Robust Angle Finding for Generalized Quantum Signal Processing." arXiv preprint arXiv:2402.03016 (2024). [Link](https://arxiv.org/abs/2402.03016)
