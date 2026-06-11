"""Bearing-only SO(3) EqF problem, simulations, and plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os

import casadi as ca
import cyecca.lie as lie
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from equivariant_filter_casadi import EqFProblem, derive_equivariant_filter
from so3_example_utils import (
    angle_between_directions_deg,
    exp_so3_np,
    hat_np,
    normalize_np,
    omega_profile,
    project_to_so3_np,
    symmetrize_psd,
)

@dataclass(frozen=True)
class BearingEqFSymbolics:
    derived: object
    A: ca.Function
    B: ca.Function
    C: ca.Function
    C_star: ca.Function | None
    yhat: ca.Function
    normal_coord: ca.Function
    normal_coord_inv: ca.Function
    ekf_h: ca.Function
    ekf_H: ca.Function
    ekf_g: ca.Function
    ekf_G: ca.Function
    output_linearization_error: ca.Function

def make_bearing_eqf_symbolics():
    Rhat = ca.SX.sym("Rhat", 3, 3)
    Omega = ca.SX.sym("Omega", 3)
    y = ca.SX.sym("y", 3)
    eta = ca.SX.sym("eta", 3)
    eps = ca.SX.sym("eps", 2)
    c_m = ca.SX.sym("c_m")

    e1 = ca.SX([1.0, 0.0, 0.0])

    def eps_wedge(error):
        return ca.vertcat(0, error[0], error[1])

    def phi(R, eta_arg):
        return R.T @ eta_arg

    def psi(R, omega):
        return R.T @ omega

    def h(eta_arg):
        return c_m * eta_arg

    def rho(R, y_arg):
        return R.T @ y_arg

    def chart_inv(error):
        return lie.so3.elem(eps_wedge(error)).exp(lie.SO3Dcm).to_Matrix().T @ e1

    bearing_problem = EqFProblem(
        name="bearing",
        state=eta,
        group=Rhat,
        input=Omega,
        error=eps,
        output_measurement=y,
        lie_group=lie.SO3Dcm,
        origin=e1,
        parameters=(c_m,),
        action=phi,
        input_action=psi,
        output=h,
        output_action=rho,
        lift=lambda _eta, omega: omega,
        chart_inv=chart_inv,
        algebra_from_error=eps_wedge,
    )
    derived = derive_equivariant_filter(bearing_problem)

    e1_cross_eta = lie.so3.elem(e1).to_Matrix() @ eta
    cross_norm = ca.norm_2(e1_cross_eta)
    theta = ca.atan2(cross_norm, e1.T @ eta)
    tangent_part = ca.vertcat(e1_cross_eta[1], e1_cross_eta[2])
    normal_coord = ca.if_else(
        cross_norm < 1e-9,
        ca.SX.zeros(2, 1),
        -theta * tangent_part / cross_norm,
    )

    eta_norm = ca.norm_2(eta)
    h_ekf = c_m * eta / eta_norm
    H_ekf = ca.jacobian(h_ekf, eta)
    g_ekf = ca.dot(eta, eta)
    G_ekf = ca.jacobian(g_ekf, eta)

    normal_coord_inv = chart_inv(eps)
    true_residual = h(normal_coord_inv) - h(e1)
    C_id = ca.substitute(
        ca.reshape(ca.jacobian(ca.reshape(true_residual, 3, 1), eps), 6, 1),
        eps,
        ca.SX.zeros(eps.shape),
    ).reshape((3, 2))

    tau = ca.SX.sym("bearing_residual_error_tau")
    y_for_error = ca.SX.sym("bearing_y_for_error", 3)
    output_curve = lie.so3.elem(tau * eps_wedge(eps)).exp(lie.SO3Dcm).to_Matrix()
    dy_at_y = ca.substitute(
        ca.jacobian(ca.reshape(rho(output_curve, y_for_error), 3, 1), tau),
        tau,
        ca.SX(0),
    )
    dy_at_yhat = ca.substitute(
        ca.jacobian(ca.reshape(rho(output_curve, h(e1)), 3, 1), tau),
        tau,
        ca.SX(0),
    )
    C_star_id_independent_y = ca.jacobian(0.5 * (dy_at_y + dy_at_yhat), eps)
    C_star_id = ca.substitute(C_star_id_independent_y, y_for_error, h(normal_coord_inv))
    eqf_residual = C_id @ eps
    eqf_star_residual = C_star_id @ eps
    residual_error = ca.horzcat(
        true_residual - eqf_residual,
        true_residual - eqf_star_residual,
    )

    return BearingEqFSymbolics(
        derived=derived,
        A=derived.A,
        B=derived.B,
        C=derived.C,
        C_star=derived.C_star,
        yhat=derived.yhat,
        normal_coord=ca.Function("bearing_normal_coord", [eta], [normal_coord]),
        normal_coord_inv=ca.Function("bearing_normal_coord_inv", [eps], [normal_coord_inv]),
        ekf_h=ca.Function("bearing_ekf_h", [eta, c_m], [h_ekf]),
        ekf_H=ca.Function("bearing_ekf_H", [eta, c_m], [H_ekf]),
        ekf_g=ca.Function("bearing_ekf_g", [eta], [g_ekf]),
        ekf_G=ca.Function("bearing_ekf_G", [eta], [G_ekf]),
        output_linearization_error=ca.Function(
            "bearing_output_linearization_error", [eps, c_m], [residual_error]
        ),
    )

def bearing_lyapunov(sym, R, eta_true, Sigma):
    error_coordinates = np.array(sym.normal_coord(R @ eta_true), dtype=float).reshape(2)
    return float(error_coordinates.T @ np.linalg.solve(Sigma, error_coordinates))

def bearing_secant_output_matrix(sym, R, y_m, yhat, c_m=1.0, epsilon_floor=1e-8):
    """Measurement-dependent secant output matrix for the bearing residual.

    This is not the paper's EqF* matrix. It chooses a rank-one matrix that maps
    the measured normal-coordinate displacement exactly onto the measured
    residual, falling back to EqF* when the displacement is too small.
    """

    measurement_direction = normalize_np(np.asarray(y_m, dtype=float).reshape(3))
    eps_meas = np.array(sym.normal_coord(R @ measurement_direction), dtype=float).reshape(2)
    eps_norm_sq = float(eps_meas @ eps_meas)
    if eps_norm_sq < epsilon_floor**2:
        if sym.C_star is None:
            raise RuntimeError("Bearing problem should have an EqF* matrix")
        return np.array(sym.C_star(R, y_m, c_m), dtype=float)

    residual = np.asarray(y_m, dtype=float).reshape(3) - np.asarray(yhat, dtype=float).reshape(3)
    return np.outer(residual, eps_meas) / eps_norm_sq

def bearing_secant_blend_key(alpha):
    return f"eqf_secant_blend_{int(round(100.0 * float(alpha))):02d}"

def bearing_secant_adaptive_key(k):
    return f"eqf_secant_adaptive_k{int(round(float(k))):02d}"

def bearing_ekf_adaptive_key(k):
    return f"ekf_adaptive_k{int(round(float(k))):02d}"

def simulate_official_auto_eqf_sphere_example(
    seed=0,
    max_t=5.0,
    dt=0.01,
    init_noise=0.5,
    gyr_noise=0.01,
    meas_noise=0.05,
    apply_gm_noise=True,
    sym=None,
):
    """Reproduce the public ``auto_eqf/sphere_example.py`` simulation.

    The source repository referenced by the paper contains a single automatic
    EqF sphere example, not the EKF/EqF/EqF* comparison script used for the
    TAC figures. This mirrors that public example's seed, noise values,
    angular velocity, discrete propagation, and discrete update.
    """

    if sym is None:
        sym = make_bearing_eqf_symbolics()

    rng = np.random.RandomState(seed)
    e1 = np.array([1.0, 0.0, 0.0])
    eta = normalize_np(e1 + init_noise * rng.randn(3))
    R_hat = np.eye(3)
    Sigma = (init_noise**2) * np.eye(2)

    n_steps = int(max_t / dt)
    times = np.arange(n_steps, dtype=float) * dt
    hist = {
        "t": times,
        "err_deg": np.zeros(n_steps),
        "lyapunov": np.zeros(n_steps),
        "trace": np.zeros(n_steps),
    }

    noise_scale = 1.0 if apply_gm_noise else 0.0
    gyro_cov = (gyr_noise**2) * np.eye(3)
    meas_cov = (meas_noise**2) * np.eye(3)

    for step, t_now in enumerate(times):
        omega = np.array([0.1 * np.cos(2.0 * t_now), 0.2 * np.sin(t_now), 0.0])
        eta = exp_so3_np(-dt * omega) @ eta

        gyr = omega + noise_scale * gyr_noise * rng.randn(3)
        meas = eta + noise_scale * meas_noise * rng.randn(3)

        B = np.array(sym.B(R_hat, gyr, 1.0), dtype=float)
        R_hat = project_to_so3_np(R_hat @ exp_so3_np(dt * gyr))
        Sigma = symmetrize_psd(Sigma + dt * (B @ gyro_cov @ B.T))

        yhat = R_hat.T @ e1
        if sym.C_star is None:
            raise RuntimeError("Bearing problem should have an EqF* matrix")
        C = np.array(sym.C_star(R_hat, meas, 1.0), dtype=float)
        S_inv = np.linalg.inv(C @ Sigma @ C.T + meas_cov)
        delta = Sigma @ C.T @ S_inv @ (meas - yhat)
        R_hat = project_to_so3_np(exp_so3_np(np.array([0.0, delta[0], delta[1]])) @ R_hat)
        Sigma = symmetrize_psd(Sigma - Sigma @ C.T @ S_inv @ C @ Sigma)

        eta_hat = R_hat.T @ e1
        hist["err_deg"][step] = float(
            np.degrees(np.arccos(np.clip(float(eta @ eta_hat), -1.0, 1.0)))
        )
        hist["lyapunov"][step] = bearing_lyapunov(sym, R_hat, eta, Sigma)
        hist["trace"][step] = np.trace(Sigma)

    return hist

def simulate_bearing_only_eqf(
    seed=7,
    T=5.0,
    dt=0.01,
    sigma_init=2.0,
    sigma_omega=0.01,
    sigma_y=0.05,
    filter_sigma_omega=0.01,
    filter_sigma_y=0.05,
    ekf_filter_sigma_omega=None,
    ekf_filter_sigma_y=None,
    ekf_n_epsilon=None,
    ekf_update="continuous",
    ekf_constraint_jacobian_sign=1.0,
    m_epsilon=1e-2,
    n_epsilon=2e-2,
    sigma_constraint=1.0,
    initial_covariance=None,
    initial_eta=None,
    c_m=1.0,
    secant_blend_weights=(0.25, 0.5, 0.75),
    secant_adaptive_k_values=(4.0, 8.0, 16.0),
    secant_adaptive_alpha_max=1.0,
    ekf_adaptive_k_values=(4.0, 8.0, 16.0),
    ekf_adaptive_alpha_max=1.0,
    include_ekf_flipped=True,
    sym=None,
    rng=None,
):
    """Run one Section VII bearing-estimation trial.

    The stochastic setup follows Eq. (50) of the paper.  The EqF observers
    follow the continuous-time equations (36)--(38) with Euler integration.
    The paper states that the EKF was also integrated with Euler, so the
    default EKF update is a continuous Kalman-Bucy-style correction.  Set
    ``ekf_update="discrete"`` to compare against a sampled-measurement EKF.
    The Section VI-A linearization-error tuning terms M_epsilon and N_epsilon
    are exposed because the paper states they are used but does not publish
    their exact numerical values.
    """

    if sym is None:
        sym = make_bearing_eqf_symbolics()
    if rng is None:
        rng = np.random.default_rng(seed)
    times = np.arange(0.0, T + dt, dt)

    e1 = np.array([1.0, 0.0, 0.0])
    if initial_eta is None:
        eta_true = normalize_np(e1 + sigma_init * rng.normal(size=3))
    else:
        eta_true = normalize_np(initial_eta)
    if initial_covariance is None:
        initial_covariance = sigma_init**2
    if ekf_filter_sigma_omega is None:
        ekf_filter_sigma_omega = filter_sigma_omega
    if ekf_filter_sigma_y is None:
        ekf_filter_sigma_y = filter_sigma_y
    if ekf_n_epsilon is None:
        ekf_n_epsilon = n_epsilon
    if ekf_update not in {"continuous", "discrete"}:
        raise ValueError('ekf_update must be "continuous" or "discrete"')

    R_eqf = np.eye(3)
    R_eqf_star = np.eye(3)
    R_eqf_secant = np.eye(3)
    secant_blend_weights = tuple(float(alpha) for alpha in secant_blend_weights)
    R_secant_blend = {
        bearing_secant_blend_key(alpha): np.eye(3) for alpha in secant_blend_weights
    }
    secant_adaptive_k_values = tuple(float(k) for k in secant_adaptive_k_values)
    R_secant_adaptive = {
        bearing_secant_adaptive_key(k): np.eye(3) for k in secant_adaptive_k_values
    }
    eta_ekf = e1.copy()
    Sigma_eqf = initial_covariance * np.eye(2)
    Sigma_eqf_star = initial_covariance * np.eye(2)
    Sigma_eqf_secant = initial_covariance * np.eye(2)
    Sigma_secant_blend = {
        bearing_secant_blend_key(alpha): initial_covariance * np.eye(2)
        for alpha in secant_blend_weights
    }
    Sigma_secant_adaptive = {
        bearing_secant_adaptive_key(k): initial_covariance * np.eye(2)
        for k in secant_adaptive_k_values
    }
    P_ekf = initial_covariance * np.eye(3)
    eta_ekf_flipped = e1.copy()
    P_ekf_flipped = initial_covariance * np.eye(3)
    ekf_flipped_alive = include_ekf_flipped
    ekf_adaptive_k_values = tuple(float(k) for k in ekf_adaptive_k_values)
    eta_ekf_adaptive = {
        bearing_ekf_adaptive_key(k): e1.copy() for k in ekf_adaptive_k_values
    }
    P_ekf_adaptive = {
        bearing_ekf_adaptive_key(k): initial_covariance * np.eye(3)
        for k in ekf_adaptive_k_values
    }

    hist = {
        "t": times,
        "eqf_err_deg": np.zeros_like(times),
        "eqf_star_err_deg": np.zeros_like(times),
        "eqf_secant_err_deg": np.zeros_like(times),
        "ekf_err_deg": np.zeros_like(times),
        "ekf_flipped_err_deg": np.zeros_like(times),
        "eqf_lyapunov": np.zeros_like(times),
        "eqf_star_lyapunov": np.zeros_like(times),
        "eqf_secant_lyapunov": np.zeros_like(times),
        "ekf_lyapunov": np.zeros_like(times),
        "ekf_flipped_lyapunov": np.zeros_like(times),
        "eqf_trace": np.zeros_like(times),
        "eqf_star_trace": np.zeros_like(times),
        "eqf_secant_trace": np.zeros_like(times),
        "ekf_trace": np.zeros_like(times),
        "ekf_flipped_trace": np.zeros_like(times),
        "ekf_norm": np.zeros_like(times),
        "ekf_flipped_norm": np.zeros_like(times),
    }
    for alpha in secant_blend_weights:
        key = bearing_secant_blend_key(alpha)
        hist[f"{key}_err_deg"] = np.zeros_like(times)
        hist[f"{key}_lyapunov"] = np.zeros_like(times)
        hist[f"{key}_trace"] = np.zeros_like(times)
    for k_value in secant_adaptive_k_values:
        key = bearing_secant_adaptive_key(k_value)
        hist[f"{key}_err_deg"] = np.zeros_like(times)
        hist[f"{key}_lyapunov"] = np.zeros_like(times)
        hist[f"{key}_trace"] = np.zeros_like(times)
    for k_value in ekf_adaptive_k_values:
        key = bearing_ekf_adaptive_key(k_value)
        hist[f"{key}_err_deg"] = np.zeros_like(times)
        hist[f"{key}_lyapunov"] = np.zeros_like(times)
        hist[f"{key}_trace"] = np.zeros_like(times)
        hist[f"{key}_norm"] = np.zeros_like(times)

    M_omega_filter = (filter_sigma_omega**2) * np.eye(3)
    N_y_filter = (filter_sigma_y**2 + n_epsilon) * np.eye(3)
    M_omega_ekf = (ekf_filter_sigma_omega**2) * np.eye(3)
    N_ekf = np.diag([ekf_filter_sigma_y**2 + ekf_n_epsilon] * 3 + [sigma_constraint**2 + ekf_n_epsilon])

    for k, t_now in enumerate(times):
        eta_eqf = R_eqf.T @ e1
        eta_eqf_star = R_eqf_star.T @ e1
        eta_eqf_secant = R_eqf_secant.T @ e1
        hist["eqf_err_deg"][k] = angle_between_directions_deg(eta_eqf, eta_true)
        hist["eqf_star_err_deg"][k] = angle_between_directions_deg(eta_eqf_star, eta_true)
        hist["eqf_secant_err_deg"][k] = angle_between_directions_deg(eta_eqf_secant, eta_true)
        hist["ekf_err_deg"][k] = angle_between_directions_deg(eta_ekf, eta_true)
        hist["ekf_flipped_err_deg"][k] = (
            angle_between_directions_deg(eta_ekf_flipped, eta_true)
            if ekf_flipped_alive
            else np.nan
        )
        hist["eqf_lyapunov"][k] = bearing_lyapunov(sym, R_eqf, eta_true, Sigma_eqf)
        hist["eqf_star_lyapunov"][k] = bearing_lyapunov(sym, R_eqf_star, eta_true, Sigma_eqf_star)
        hist["eqf_secant_lyapunov"][k] = bearing_lyapunov(sym, R_eqf_secant, eta_true, Sigma_eqf_secant)
        ekf_error = eta_true - eta_ekf
        hist["ekf_lyapunov"][k] = float(ekf_error.T @ np.linalg.solve(P_ekf, ekf_error))
        if ekf_flipped_alive:
            try:
                ekf_flipped_error = eta_true - eta_ekf_flipped
                hist["ekf_flipped_lyapunov"][k] = float(
                    ekf_flipped_error.T
                    @ np.linalg.solve(P_ekf_flipped, ekf_flipped_error)
                )
            except (np.linalg.LinAlgError, ValueError):
                ekf_flipped_alive = False
                hist["ekf_flipped_err_deg"][k] = np.nan
                hist["ekf_flipped_lyapunov"][k] = np.nan
        else:
            hist["ekf_flipped_lyapunov"][k] = np.nan
        hist["eqf_trace"][k] = np.trace(Sigma_eqf)
        hist["eqf_star_trace"][k] = np.trace(Sigma_eqf_star)
        hist["eqf_secant_trace"][k] = np.trace(Sigma_eqf_secant)
        hist["ekf_trace"][k] = np.trace(P_ekf)
        hist["ekf_flipped_trace"][k] = np.trace(P_ekf_flipped) if ekf_flipped_alive else np.nan
        hist["ekf_norm"][k] = np.linalg.norm(eta_ekf)
        hist["ekf_flipped_norm"][k] = np.linalg.norm(eta_ekf_flipped) if ekf_flipped_alive else np.nan
        for alpha in secant_blend_weights:
            key = bearing_secant_blend_key(alpha)
            hist[f"{key}_err_deg"][k] = angle_between_directions_deg(
                R_secant_blend[key].T @ e1, eta_true
            )
            hist[f"{key}_lyapunov"][k] = bearing_lyapunov(
                sym, R_secant_blend[key], eta_true, Sigma_secant_blend[key]
            )
            hist[f"{key}_trace"][k] = np.trace(Sigma_secant_blend[key])
        for k_value in secant_adaptive_k_values:
            key = bearing_secant_adaptive_key(k_value)
            hist[f"{key}_err_deg"][k] = angle_between_directions_deg(
                R_secant_adaptive[key].T @ e1, eta_true
            )
            hist[f"{key}_lyapunov"][k] = bearing_lyapunov(
                sym, R_secant_adaptive[key], eta_true, Sigma_secant_adaptive[key]
            )
            hist[f"{key}_trace"][k] = np.trace(Sigma_secant_adaptive[key])
        for k_value in ekf_adaptive_k_values:
            key = bearing_ekf_adaptive_key(k_value)
            hist[f"{key}_err_deg"][k] = angle_between_directions_deg(
                eta_ekf_adaptive[key], eta_true
            )
            ekf_adaptive_error = eta_true - eta_ekf_adaptive[key]
            hist[f"{key}_lyapunov"][k] = float(
                ekf_adaptive_error.T @ np.linalg.solve(P_ekf_adaptive[key], ekf_adaptive_error)
            )
            hist[f"{key}_trace"][k] = np.trace(P_ekf_adaptive[key])
            hist[f"{key}_norm"][k] = np.linalg.norm(eta_ekf_adaptive[key])

        if k == len(times) - 1:
            break

        omega = omega_profile(float(t_now))
        omega_m = omega + sigma_omega * rng.normal(size=3)
        y_m = c_m * eta_true + sigma_y * rng.normal(size=3)

        output_models = ["eqf", "eqf_star", "secant"]
        output_models.extend(bearing_secant_blend_key(alpha) for alpha in secant_blend_weights)
        output_models.extend(bearing_secant_adaptive_key(k) for k in secant_adaptive_k_values)
        for output_model in output_models:
            if output_model == "eqf":
                R, Sigma = R_eqf, Sigma_eqf
            elif output_model == "eqf_star":
                R, Sigma = R_eqf_star, Sigma_eqf_star
            elif output_model == "secant":
                R, Sigma = R_eqf_secant, Sigma_eqf_secant
            elif output_model.startswith("eqf_secant_adaptive"):
                R, Sigma = R_secant_adaptive[output_model], Sigma_secant_adaptive[output_model]
            else:
                R, Sigma = R_secant_blend[output_model], Sigma_secant_blend[output_model]

            B = np.array(sym.B(R, omega_m, c_m), dtype=float)
            yhat = np.array(sym.yhat(R, c_m), dtype=float).reshape(3)
            if output_model == "eqf":
                C = np.array(sym.C(R, c_m), dtype=float)
            elif output_model == "eqf_star":
                if sym.C_star is None:
                    raise RuntimeError("Bearing problem should have an EqF* matrix")
                C = np.array(sym.C_star(R, y_m, c_m), dtype=float)
            elif output_model == "secant":
                C = bearing_secant_output_matrix(sym, R, y_m, yhat, c_m)
            elif output_model.startswith("eqf_secant_adaptive"):
                k_value = int(output_model.rsplit("k", maxsplit=1)[-1])
                measurement_direction = normalize_np(np.asarray(y_m, dtype=float).reshape(3))
                eps_meas = np.array(
                    sym.normal_coord(R @ measurement_direction), dtype=float
                ).reshape(2)
                rho_sq = float(eps_meas @ eps_meas)
                noise_sq = float(filter_sigma_y**2)
                alpha = secant_adaptive_alpha_max * rho_sq / (
                    rho_sq + k_value * noise_sq
                )
                C_standard = np.array(sym.C(R, c_m), dtype=float)
                C_secant = bearing_secant_output_matrix(sym, R, y_m, yhat, c_m)
                C = (1.0 - alpha) * C_standard + alpha * C_secant
            else:
                alpha = int(output_model.rsplit("_", maxsplit=1)[-1]) / 100.0
                C_standard = np.array(sym.C(R, c_m), dtype=float)
                C_secant = bearing_secant_output_matrix(sym, R, y_m, yhat, c_m)
                C = (1.0 - alpha) * C_standard + alpha * C_secant

            process = B @ M_omega_filter @ B.T + m_epsilon * np.eye(2)
            N_inv = np.linalg.inv(N_y_filter)
            gain = Sigma @ C.T @ N_inv
            delta = gain @ (y_m - yhat)
            Sigma_dot = process - Sigma @ C.T @ N_inv @ C @ Sigma
            Sigma = symmetrize_psd(Sigma + dt * Sigma_dot)
            R = project_to_so3_np(
                exp_so3_np(dt * np.array([0.0, delta[0], delta[1]]))
                @ R
                @ exp_so3_np(dt * omega_m)
            )

            if output_model == "eqf":
                R_eqf, Sigma_eqf = R, Sigma
            elif output_model == "eqf_star":
                R_eqf_star, Sigma_eqf_star = R, Sigma
            elif output_model == "secant":
                R_eqf_secant, Sigma_eqf_secant = R, Sigma
            elif output_model.startswith("eqf_secant_adaptive"):
                R_secant_adaptive[output_model], Sigma_secant_adaptive[output_model] = R, Sigma
            else:
                R_secant_blend[output_model], Sigma_secant_blend[output_model] = R, Sigma

        def step_ekf(eta_est, P_est, constraint_jacobian_sign, adaptive_k=None):
            F = -hat_np(omega_m)
            G = hat_np(eta_est)
            yhat_ekf = np.array(sym.ekf_h(eta_est, c_m), dtype=float).reshape(3)
            H_y = np.array(sym.ekf_H(eta_est, c_m), dtype=float)
            if adaptive_k is not None:
                measurement_direction = normalize_np(np.asarray(y_m, dtype=float).reshape(3))
                state_displacement = measurement_direction - np.asarray(eta_est, dtype=float).reshape(3)
                displacement_norm_sq = float(state_displacement @ state_displacement)
                if displacement_norm_sq > 1e-12:
                    residual = y_m - yhat_ekf
                    H_secant = np.outer(residual, state_displacement) / displacement_norm_sq
                    alpha = ekf_adaptive_alpha_max * displacement_norm_sq / (
                        displacement_norm_sq + float(adaptive_k) * float(ekf_filter_sigma_y**2)
                    )
                    H_y = (1.0 - alpha) * H_y + alpha * H_secant
            zhat = float(sym.ekf_g(eta_est))
            H_g = constraint_jacobian_sign * np.array(
                sym.ekf_G(eta_est), dtype=float
            ).reshape(1, 3)
            innovation = np.concatenate((y_m - yhat_ekf, [1.0 - zhat]))
            H = np.vstack((H_y, H_g))
            Q = G @ M_omega_ekf @ G.T + m_epsilon * np.eye(3)
            if ekf_update == "continuous":
                N_ekf_inv = np.linalg.inv(N_ekf)
                K = P_est @ H.T @ N_ekf_inv
                eta_est = eta_est + dt * (F @ eta_est + K @ innovation)
                P_dot = F @ P_est + P_est @ F.T + Q - P_est @ H.T @ N_ekf_inv @ H @ P_est
                P_est = symmetrize_psd(P_est + dt * P_dot)
            else:
                process = F @ P_est + P_est @ F.T + Q
                P_est = symmetrize_psd(P_est + dt * process)
                S = H @ P_est @ H.T + N_ekf
                K = P_est @ H.T @ np.linalg.inv(S)
                eta_est = exp_so3_np(-dt * omega_m) @ eta_est + K @ innovation
                I_ekf = np.eye(3)
                P_est = symmetrize_psd((I_ekf - K @ H) @ P_est @ (I_ekf - K @ H).T + K @ N_ekf @ K.T)
            return eta_est, P_est

        eta_ekf, P_ekf = step_ekf(eta_ekf, P_ekf, ekf_constraint_jacobian_sign)
        if ekf_flipped_alive:
            try:
                eta_ekf_flipped, P_ekf_flipped = step_ekf(
                    eta_ekf_flipped,
                    P_ekf_flipped,
                    -ekf_constraint_jacobian_sign,
                )
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                ekf_flipped_alive = False
                eta_ekf_flipped = np.full(3, np.nan)
                P_ekf_flipped = np.full((3, 3), np.nan)
        for k_value in ekf_adaptive_k_values:
            key = bearing_ekf_adaptive_key(k_value)
            eta_ekf_adaptive[key], P_ekf_adaptive[key] = step_ekf(
                eta_ekf_adaptive[key],
                P_ekf_adaptive[key],
                ekf_constraint_jacobian_sign,
                adaptive_k=k_value,
            )

        eta_true = exp_so3_np(-dt * omega) @ eta_true

    return hist

def bearing_monte_carlo_summary(histories):
    keys = [
        "eqf_err_deg",
        "eqf_star_err_deg",
        "eqf_secant_err_deg",
        "ekf_err_deg",
        "ekf_flipped_err_deg",
        "eqf_lyapunov",
        "eqf_star_lyapunov",
        "eqf_secant_lyapunov",
        "ekf_lyapunov",
        "ekf_flipped_lyapunov",
        "eqf_trace",
        "eqf_star_trace",
        "eqf_secant_trace",
        "ekf_trace",
        "ekf_flipped_trace",
    ]
    keys.extend(
        key
        for key in histories[0]
        if (
            key.startswith("eqf_secant_blend_")
            or key.startswith("eqf_secant_adaptive_")
            or key.startswith("ekf_adaptive_")
        )
        and (
            key.endswith("_err_deg")
            or key.endswith("_lyapunov")
            or key.endswith("_trace")
        )
    )
    summary = {"t": histories[0]["t"], "n_trials": len(histories)}
    for key in keys:
        values = np.stack([hist[key] for hist in histories], axis=0)
        summary[f"{key}_median"] = np.nanpercentile(values, 50, axis=0)
        summary[f"{key}_q25"] = np.nanpercentile(values, 25, axis=0)
        summary[f"{key}_q75"] = np.nanpercentile(values, 75, axis=0)
    return summary

def _simulate_bearing_trial_worker(args):
    entropy, spawn_key, trial_kwargs = args
    child_seed = np.random.SeedSequence(entropy, spawn_key=tuple(spawn_key))
    sym = make_bearing_eqf_symbolics()
    return simulate_bearing_only_eqf(
        sym=sym,
        rng=np.random.default_rng(child_seed),
        seed=None,
        **trial_kwargs,
    )

def _resolve_parallel_jobs(n_jobs, n_trials):
    """Resolve joblib-style worker counts, capped by the trial count."""

    cpu_count = os.cpu_count() or 1
    if n_jobs is None:
        requested = cpu_count
    else:
        requested = int(n_jobs)
        if requested == 0:
            raise ValueError("n_jobs=0 is invalid; use None, a positive count, or a negative joblib-style count")
        if requested < 0:
            requested = max(1, cpu_count + 1 + requested)
    return max(1, min(requested, int(n_trials)))

def _run_bearing_trials_stdlib(tasks, n_jobs, backend, verbose):
    """Run trials without joblib, using only the Python standard library."""

    if backend == "threading":
        from concurrent.futures import ThreadPoolExecutor

        if verbose:
            print(f"joblib not found; using ThreadPoolExecutor with {n_jobs} workers.")
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            return list(executor.map(_simulate_bearing_trial_worker, tasks))

    import multiprocessing as mp

    if "fork" not in mp.get_all_start_methods():
        from concurrent.futures import ThreadPoolExecutor

        if verbose:
            print("joblib not found and fork is unavailable; using ThreadPoolExecutor instead.")
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            return list(executor.map(_simulate_bearing_trial_worker, tasks))

    mp_context = mp.get_context("fork")
    if verbose:
        print(f"joblib not found; using multiprocessing fork with {n_jobs} workers.")

    task_queue = mp_context.Queue()
    result_queue = mp_context.Queue()

    def worker_loop():
        while True:
            item = task_queue.get()
            if item is None:
                break
            index, task = item
            try:
                result_queue.put((index, None, _simulate_bearing_trial_worker(task)))
            except BaseException as exc:
                result_queue.put((index, repr(exc), None))

    workers = [mp_context.Process(target=worker_loop) for _ in range(n_jobs)]
    for worker in workers:
        worker.start()
    for index, task in enumerate(tasks):
        task_queue.put((index, task))
    for _ in workers:
        task_queue.put(None)

    histories = [None] * len(tasks)
    try:
        for _ in tasks:
            index, error, history = result_queue.get()
            if error is not None:
                raise RuntimeError(error)
            histories[index] = history
    finally:
        for worker in workers:
            worker.join()
    return histories

def simulate_bearing_monte_carlo(
    n_trials=500,
    seed=11,
    n_jobs=None,
    parallel=True,
    backend="loky",
    verbose=0,
    **trial_kwargs,
):
    """Run the paper-style noisy Monte Carlo experiment from Fig. 2.

    Set ``parallel=True`` and ``n_jobs`` to use multiple workers.  ``n_jobs``
    follows joblib's convention: ``None`` uses all available CPUs, ``-1`` uses
    all CPUs, and ``-2`` leaves one CPU free.  The default ``backend="loky"``
    uses joblib's process backend when joblib is installed, otherwise it falls
    back to Python's standard-library ``multiprocessing`` module.  Use
    ``backend="threading"`` if process startup overhead dominates for short
    smoke tests.  Increase ``verbose`` to print the worker count.
    """

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(n_trials)

    if not parallel or n_trials <= 1:
        sym = trial_kwargs.pop("sym", None)
        if sym is None:
            sym = make_bearing_eqf_symbolics()
        histories = [
            simulate_bearing_only_eqf(
                sym=sym,
                rng=np.random.default_rng(child_seed),
                seed=None,
                **trial_kwargs,
            )
            for child_seed in child_seeds
        ]
        return bearing_monte_carlo_summary(histories)

    if "sym" in trial_kwargs:
        trial_kwargs = dict(trial_kwargs)
        trial_kwargs.pop("sym")

    n_jobs = _resolve_parallel_jobs(n_jobs, n_trials)
    if n_jobs == 1:
        return simulate_bearing_monte_carlo(
            n_trials=n_trials,
            seed=seed,
            parallel=False,
            **trial_kwargs,
        )

    tasks = [(child.entropy, child.spawn_key, dict(trial_kwargs)) for child in child_seeds]
    try:
        try:
            from joblib import Parallel, delayed
        except ModuleNotFoundError:
            histories = _run_bearing_trials_stdlib(tasks, n_jobs, backend, verbose)
        else:

            histories = Parallel(n_jobs=n_jobs, backend=backend, batch_size=max(1, n_trials // (4 * n_jobs)), verbose=verbose)(
                delayed(_simulate_bearing_trial_worker)(task) for task in tasks
            )
    except Exception as exc:
        print(f"Parallel Monte Carlo failed ({exc}); falling back to serial execution.")
        histories = [_simulate_bearing_trial_worker(task) for task in tasks]
    return bearing_monte_carlo_summary(histories)

def measurement_linearization_error_grid(
    n_polar=121,
    n_azimuth=241,
    c_m=1.0,
    ekf_convention="text",
    include_secant=False,
):
    """Compute Fig. 3-style output-residual linearization errors.

    ``ekf_convention="text"`` uses the literal residual sign implied by
    ``vartheta(eta) := eta - e1`` in the Fig. 3 description.
    ``ekf_convention="paper"`` reproduces the opposite-sign convention visible
    in the paper's EKF heatmap.  Set ``include_secant=True`` to include an
    output-dependent rank-one secant matrix that exactly matches the measured
    residual for the current point.  That is useful as a "what would be
    possible with a measurement-dependent residual" diagnostic, not as the
    EqF linearization from the paper.
    """

    if ekf_convention not in {"paper", "text"}:
        raise ValueError('ekf_convention must be "paper" or "text"')

    sym = make_bearing_eqf_symbolics()
    e1 = np.array([1.0, 0.0, 0.0])
    I3 = np.eye(3)
    H_ekf = np.array(sym.ekf_H(e1, c_m), dtype=float)
    C_eqf = np.array(sym.C(I3, c_m), dtype=float)

    polar = np.linspace(0.0, np.pi, n_polar)
    azimuth = np.linspace(-np.pi, np.pi, n_azimuth)
    phi_grid, theta_grid = np.meshgrid(azimuth, polar)

    errors = {"EKF": np.zeros_like(theta_grid), "EqF": np.zeros_like(theta_grid), "EqF*": np.zeros_like(theta_grid)}
    if include_secant:
        errors["Secant"] = np.zeros_like(theta_grid)

    for i, theta in enumerate(polar):
        sin_theta = np.sin(theta)
        for j, phi in enumerate(azimuth):
            if np.isclose(theta, 0.5 * np.pi) and np.isclose(abs(phi), np.pi):
                phi = np.sign(phi) * (np.pi - 1e-9)
            eta = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)])
            true_residual = c_m * (eta - e1)
            if ekf_convention == "paper":
                ekf_residual = -H_ekf @ (eta - e1)
            else:
                ekf_residual = H_ekf @ (eta - e1)
            eps = np.array(sym.normal_coord(eta), dtype=float).reshape(2)
            eqf_residual = C_eqf @ eps
            if sym.C_star is None:
                raise RuntimeError("Bearing problem should have an EqF* matrix")
            C_eqf_star = np.array(sym.C_star(I3, c_m * eta, c_m), dtype=float)
            eqf_star_residual = C_eqf_star @ eps
            if include_secant:
                eps_norm_sq = float(eps @ eps)
                if eps_norm_sq < 1e-12:
                    secant_residual = true_residual
                else:
                    C_secant = np.outer(true_residual, eps) / eps_norm_sq
                    secant_residual = C_secant @ eps

            errors["EKF"][i, j] = np.linalg.norm(true_residual - ekf_residual)
            errors["EqF"][i, j] = np.linalg.norm(true_residual - eqf_residual)
            errors["EqF*"][i, j] = np.linalg.norm(true_residual - eqf_star_residual)
            if include_secant:
                errors["Secant"][i, j] = np.linalg.norm(true_residual - secant_residual)

    return {"polar": polar, "azimuth": azimuth, "theta_grid": theta_grid, "phi_grid": phi_grid, **errors}

def ekf_linearization_sign_error_grid(n_polar=121, n_azimuth=241, c_m=1.0):
    """Compare the EKF residual-sign convention in the text against the opposite sign."""

    sym = make_bearing_eqf_symbolics()
    e1 = np.array([1.0, 0.0, 0.0])
    H_ekf = np.array(sym.ekf_H(e1, c_m), dtype=float)

    polar = np.linspace(0.0, np.pi, n_polar)
    azimuth = np.linspace(-np.pi, np.pi, n_azimuth)
    phi_grid, theta_grid = np.meshgrid(azimuth, polar)
    errors = {
        "EKF Correct Sign": np.zeros_like(theta_grid),
        "EKF Paper/Fig. 3 Sign": np.zeros_like(theta_grid),
    }

    for i, theta in enumerate(polar):
        sin_theta = np.sin(theta)
        for j, phi in enumerate(azimuth):
            eta = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)])
            true_residual = c_m * (eta - e1)
            coordinate_residual = eta - e1
            errors["EKF Correct Sign"][i, j] = np.linalg.norm(
                true_residual - H_ekf @ coordinate_residual
            )
            errors["EKF Paper/Fig. 3 Sign"][i, j] = np.linalg.norm(
                true_residual + H_ekf @ coordinate_residual
            )

    return {
        "polar": polar,
        "azimuth": azimuth,
        "theta_grid": theta_grid,
        "phi_grid": phi_grid,
        **errors,
    }

def plot_ekf_linearization_sign_errors(grid=None, n_polar=121, n_azimuth=241, c_m=1.0, vmax=None):
    if grid is None:
        grid = ekf_linearization_sign_error_grid(
            n_polar=n_polar, n_azimuth=n_azimuth, c_m=c_m
        )

    names = ["EKF Correct Sign", "EKF Paper/Fig. 3 Sign"]
    if vmax is None:
        vmax = max(float(np.nanmax(grid[name])) for name in names)

    fig, axes = plt.subplots(len(names), 1, figsize=(5.8, 2.7 * len(names)), sharex=True)
    shared_norm = colors.Normalize(vmin=0.0, vmax=vmax)
    mesh = None
    x_ticks = [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
    x_ticklabels = [r"$-1.0\pi$", r"$-0.5\pi$", r"$0.0\pi$", r"$0.5\pi$", r"$1.0\pi$"]
    y_ticks = [0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi]
    y_ticklabels = [r"$0.0\pi$", r"$0.25\pi$", r"$0.5\pi$", r"$0.75\pi$", r"$1.0\pi$"]
    for ax, name in zip(axes, names):
        mesh = ax.pcolormesh(
            grid["azimuth"],
            grid["polar"],
            grid[name],
            shading="auto",
            norm=shared_norm,
            cmap="jet",
        )
        ax.text(0.07, 0.78, name, transform=ax.transAxes, fontsize=12, bbox={"facecolor": "white", "edgecolor": "black", "pad": 3.0})
        ax.set_ylabel(r"$\theta$ (rad)")
        ax.set_xlim(grid["azimuth"][0], grid["azimuth"][-1])
        ax.set_ylim(grid["polar"][0], grid["polar"][-1])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)

    axes[-1].set_xlabel(r"$\varphi$ (rad)")
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xticklabels(x_ticklabels)
    axes[0].set_title("EKF Output Linearisation Error: Corrected vs Paper Sign")
    fig.subplots_adjust(right=0.84, hspace=0.12)
    cax = fig.add_axes([0.87, 0.13, 0.035, 0.72])
    fig.colorbar(mesh, cax=cax, label=r"Linearisation Error $|\tilde{y} - H\vartheta|$")
    return fig, axes, grid

def plot_measurement_linearization_errors(grid=None, n_polar=121, n_azimuth=241, c_m=1.0, vmax=None):
    if grid is None:
        grid = measurement_linearization_error_grid(n_polar=n_polar, n_azimuth=n_azimuth, c_m=c_m)

    names = ["EKF", "EqF", "EqF*"]
    if "Secant" in grid:
        names.append("Secant")
    if vmax is None:
        vmax = max(float(np.nanmax(grid[name])) for name in names)

    fig, axes = plt.subplots(len(names), 1, figsize=(5.8, 2.7 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]
    shared_norm = colors.Normalize(vmin=0.0, vmax=vmax)
    mesh = None
    x_ticks = [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
    x_ticklabels = [r"$-1.0\pi$", r"$-0.5\pi$", r"$0.0\pi$", r"$0.5\pi$", r"$1.0\pi$"]
    y_ticks = [0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi]
    y_ticklabels = [r"$0.0\pi$", r"$0.25\pi$", r"$0.5\pi$", r"$0.75\pi$", r"$1.0\pi$"]
    for ax, name in zip(axes, names):
        mesh = ax.pcolormesh(grid["azimuth"], grid["polar"], grid[name], shading="auto", norm=shared_norm, cmap="jet")
        ax.text(0.07, 0.78, name, transform=ax.transAxes, fontsize=12, bbox={"facecolor": "white", "edgecolor": "black", "pad": 3.0})
        ax.set_ylabel(r"$\theta$ (rad)")
        ax.set_xlim(grid["azimuth"][0], grid["azimuth"][-1])
        ax.set_ylim(grid["polar"][0], grid["polar"][-1])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)

    axes[-1].set_xlabel(r"$\varphi$ (rad)")
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xticklabels(x_ticklabels)
    axes[0].set_title("Output Linearisation Error of Filters")

    fig.subplots_adjust(right=0.84, hspace=0.12)
    cax = fig.add_axes([0.87, 0.13, 0.035, 0.72])
    fig.colorbar(mesh, cax=cax, label=r"Linearisation Error $|\tilde{y} - C_t\epsilon|$")
    return fig, axes, grid

def log_plot_values(values, floor):
    values = np.asarray(values, dtype=float)
    return np.where(values > floor, values, np.nan)

def plot_bearing_history(hist, include_secant=True):
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True)
    axes[0].plot(hist["t"], log_plot_values(hist["eqf_star_err_deg"], 1e-8), "-", color="red", label="EqF*")
    if include_secant and "eqf_secant_err_deg" in hist:
        axes[0].plot(hist["t"], log_plot_values(hist["eqf_secant_err_deg"], 1e-8), ":", color="purple", label="EqF Secant")
    blend_styles = [
        ("eqf_secant_blend_25", "Secant Blend 25%", "tab:pink", (0, (3, 1, 1, 1))),
        ("eqf_secant_blend_50", "Secant Blend 50%", "tab:brown", (0, (1, 1))),
        ("eqf_secant_blend_75", "Secant Blend 75%", "tab:gray", (0, (5, 1))),
        ("eqf_secant_adaptive_k04", "Adaptive Secant k=4", "tab:cyan", (0, (4, 1))),
        ("eqf_secant_adaptive_k08", "Adaptive Secant k=8", "tab:olive", (0, (2, 1))),
        ("eqf_secant_adaptive_k16", "Adaptive Secant k=16", "black", (0, (6, 1))),
        ("ekf_adaptive_k04", "EKF Adaptive k=4", "gold", (0, (4, 1))),
        ("ekf_adaptive_k08", "EKF Adaptive k=8", "darkorange", (0, (2, 1))),
        ("ekf_adaptive_k16", "EKF Adaptive k=16", "sienna", (0, (6, 1))),
    ]
    for key, label, color, linestyle in blend_styles:
        if include_secant and f"{key}_err_deg" in hist:
            axes[0].plot(
                hist["t"],
                log_plot_values(hist[f"{key}_err_deg"], 1e-8),
                linestyle=linestyle,
                color=color,
                label=label,
            )
    axes[0].plot(hist["t"], log_plot_values(hist["eqf_err_deg"], 1e-8), "-.", color="green", label="EqF")
    axes[0].plot(hist["t"], log_plot_values(hist["ekf_err_deg"], 1e-8), "--", color="blue", label="EKF")
    if "ekf_flipped_err_deg" in hist:
        axes[0].plot(
            hist["t"],
            log_plot_values(hist["ekf_flipped_err_deg"], 1e-8),
            "--",
            color="tab:orange",
            label="EKF Flipped Constraint",
        )
    axes[0].set_ylabel("Bearing Error (deg)")
    axes[0].set_yscale("log")
    axes[0].set_title("Single Bearing Estimation (Noiseless)")
    axes[0].legend()

    axes[1].plot(hist["t"], log_plot_values(hist["eqf_star_lyapunov"], 1e-14), "-", color="red", label="EqF*")
    if include_secant and "eqf_secant_lyapunov" in hist:
        axes[1].plot(hist["t"], log_plot_values(hist["eqf_secant_lyapunov"], 1e-14), ":", color="purple", label="EqF Secant")
    for key, label, color, linestyle in blend_styles:
        if include_secant and f"{key}_lyapunov" in hist:
            axes[1].plot(
                hist["t"],
                log_plot_values(hist[f"{key}_lyapunov"], 1e-14),
                linestyle=linestyle,
                color=color,
                label=label,
            )
    axes[1].plot(hist["t"], log_plot_values(hist["eqf_lyapunov"], 1e-14), "-.", color="green", label="EqF")
    axes[1].plot(hist["t"], log_plot_values(hist["ekf_lyapunov"], 1e-14), "--", color="blue", label="EKF")
    if "ekf_flipped_lyapunov" in hist:
        axes[1].plot(
            hist["t"],
            log_plot_values(hist["ekf_flipped_lyapunov"], 1e-14),
            "--",
            color="tab:orange",
            label="EKF Flipped Constraint",
        )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Lyapunov Value")
    axes[1].set_yscale("log")
    axes[0].grid(False)
    axes[1].grid(False)
    fig.tight_layout()
    return fig, axes

def plot_bearing_monte_carlo(summary, include_secant=True):
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True)
    styles = {
        "eqf_star": ("EqF*", "red", "-"),
        "eqf_secant": ("EqF Secant", "purple", ":"),
        "eqf_secant_blend_25": ("Secant Blend 25%", "tab:pink", (0, (3, 1, 1, 1))),
        "eqf_secant_blend_50": ("Secant Blend 50%", "tab:brown", (0, (1, 1))),
        "eqf_secant_blend_75": ("Secant Blend 75%", "tab:gray", (0, (5, 1))),
        "eqf_secant_adaptive_k04": ("Adaptive Secant k=4", "tab:cyan", (0, (4, 1))),
        "eqf_secant_adaptive_k08": ("Adaptive Secant k=8", "tab:olive", (0, (2, 1))),
        "eqf_secant_adaptive_k16": ("Adaptive Secant k=16", "black", (0, (6, 1))),
        "eqf": ("EqF", "green", "-."),
        "ekf": ("EKF", "blue", "--"),
        "ekf_adaptive_k04": ("EKF Adaptive k=4", "gold", (0, (4, 1))),
        "ekf_adaptive_k08": ("EKF Adaptive k=8", "darkorange", (0, (2, 1))),
        "ekf_adaptive_k16": ("EKF Adaptive k=16", "sienna", (0, (6, 1))),
        "ekf_flipped": ("EKF Flipped Constraint", "tab:orange", "--"),
    }

    for prefix, (label, color, linestyle) in styles.items():
        if prefix.startswith("eqf_secant") and not include_secant:
            continue
        key = f"{prefix}_err_deg"
        if f"{key}_median" not in summary:
            continue
        axes[0].fill_between(
            summary["t"],
            log_plot_values(summary[f"{key}_q25"], 1e-6),
            log_plot_values(summary[f"{key}_q75"], 1e-6),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        axes[0].plot(
            summary["t"],
            log_plot_values(summary[f"{key}_median"], 1e-6),
            linestyle=linestyle,
            color=color,
            label=label,
        )

        key = f"{prefix}_lyapunov"
        if f"{key}_median" not in summary:
            continue
        axes[1].fill_between(
            summary["t"],
            log_plot_values(summary[f"{key}_q25"], 1e-12),
            log_plot_values(summary[f"{key}_q75"], 1e-12),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        axes[1].plot(
            summary["t"],
            log_plot_values(summary[f"{key}_median"], 1e-12),
            linestyle=linestyle,
            color=color,
            label=label,
        )

    axes[0].set_ylabel("Bearing Error (deg)")
    axes[0].set_yscale("log")
    axes[0].set_title("Single Bearing Estimation (Noisy)")
    axes[0].legend()
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Lyapunov Value")
    axes[1].set_yscale("log")
    axes[0].grid(False)
    axes[1].grid(False)
    fig.tight_layout()
    return fig, axes

def plot_official_auto_eqf_history(hist):
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.2), sharex=True)
    axes[0].plot(hist["t"], log_plot_values(hist["err_deg"], 1e-8), "r-")
    axes[0].set_ylabel("Angle Error (deg)")
    axes[0].set_title("Automatic EqF for single-bearing estimation")
    axes[0].set_yscale("log")

    axes[1].plot(hist["t"], log_plot_values(hist["lyapunov"], 1e-12), "r-")
    axes[1].set_ylabel("Lyapunov Value")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.grid(False)
        ax.set_xlim(hist["t"][0], hist["t"][-1])

    fig.tight_layout()
    return fig, axes
