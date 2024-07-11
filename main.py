# !/usr/bin/env python3

import numpy as np

from hamiltonian_simulation import truncate_exp, truncate_cos, truncate_exp_plus_cos
from gqsp_phase_angle_finder import GQSPPhaseAngleFinderViaRootFindingAndCarving
from gqsp_phase_angle_finder import GQSPPhaseAngleFinderViaPronyAndCarving
from gqsp_phase_angle_finder import compose_gqsp_operation_sequence
from gqsp_phase_angle_finder import stringify_gqsp_operation_sequence

if __name__ == '__main__':
    # Parameters for simulation and phase angle finding
    tau = 10.0
    d = 10
    scale = 1 / 2
    cos_scale = 1 / 2
    noise_scale = 1e-3

    angles = [0, np.pi / 4, - np.pi / 3]

    # Truncation functions for different types of simulations
    def truncate_func1(d):
        # exp(-1.j * tau * cos(theta))
        return truncate_exp(d[1], tau=tau)

    def truncate_func2(d):
        # cos_scale * cos(d * theta) + noise
        return truncate_cos(d[1], cos_scale=cos_scale, noise_scale=noise_scale)

    def truncate_func3(d):
        # a combination of exponential and cosine functions
        return truncate_exp_plus_cos(d[1], tau=tau, cos_scale=cos_scale, noise_scale=noise_scale)

    # Example using GQSPPhaseAngleFinderViaRootFindingAndCarving
    solver = GQSPPhaseAngleFinderViaRootFindingAndCarving(truncate_func=truncate_func1)
    # Compute phase angles
    phase_angles = solver.angle_finding(d=(-d, d), scale=0.5, measure_error=True)
    print(f"{phase_angles=}")
    # Extract solver information
    info = dict(solver.info)
    print(f"{info=}")
    # Compose the GQSP operation sequence
    U = compose_gqsp_operation_sequence(phase_angles, angles)
    print(f"U({angles=})={U}")
    # Generate string representation of the GQSP operation sequence
    U_str = stringify_gqsp_operation_sequence(phase_angles)
    print(f"U=\n{U_str}")
    print()

    # Example using GQSPPhaseAngleFinderViaPronyAndCarving
    solver = GQSPPhaseAngleFinderViaPronyAndCarving(truncate_func=truncate_func2)
    phase_angles = solver.angle_finding(d=(-d, d), scale=0.5)
    print(phase_angles)
    print()
    solver = GQSPPhaseAngleFinderViaPronyAndCarving(truncate_func=truncate_func3)
    phase_angles = solver.angle_finding(d=(-d, d), scale=0.5)
    print(phase_angles)
    print()
