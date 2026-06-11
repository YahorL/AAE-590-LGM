"""Attitude IEKF problem expressed through the generic EqF derivation API."""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import cyecca.lie as lie
import matplotlib.pyplot as plt
import numpy as np

from equivariant_filter_casadi import EqFProblem, derive_equivariant_filter
from so3_example_utils import (
    attitude_error_deg,
    exp_so3_np,
    normalize_np,
    omega_profile,
    project_to_so3_np,
    symmetrize_psd,
)

@dataclass(frozen=True)
class AttitudeIEKFSymbolics:
    derived: object
    A: ca.Function
    C: ca.Function
    yhat: ca.Function
    error_log: ca.Function

def make_attitude_iekf_symbolics():
    Rhat = ca.SX.sym("Rhat", 3, 3)
    R_state = ca.SX.sym("R_state", 3, 3)
    Omega = ca.SX.sym("Omega_iekf", 3)
    r1 = ca.SX.sym("r1", 3)
    r2 = ca.SX.sym("r2", 3)
    eps = ca.SX.sym("eps", 3)
    y_meas = ca.SX.sym("y_iekf", 6)
    I3 = ca.SX.eye(3)

    def phi(X, R):
        return R @ X

    def psi(X, omega):
        return X.T @ omega

    def h(R):
        return ca.vertcat(R.T @ r1, R.T @ r2)

    def chart_inv(error):
        return lie.so3.elem(error).exp(lie.SO3Dcm).to_Matrix()

    iekf_problem = EqFProblem(
        name="iekf",
        state=R_state,
        group=Rhat,
        input=Omega,
        error=eps,
        output_measurement=y_meas,
        lie_group=lie.SO3Dcm,
        origin=I3,
        parameters=(r1, r2),
        action=phi,
        input_action=psi,
        output=h,
        output_action=None,
        lift=lambda _R, omega: omega,
        chart_inv=chart_inv,
        algebra_from_error=lambda error: error,
    )
    derived = derive_equivariant_filter(iekf_problem)

    R_true_sym = ca.SX.sym("R_true", 3, 3)
    error_log = lie.SO3Dcm.from_Matrix(R_true_sym @ Rhat.T).log().param

    return AttitudeIEKFSymbolics(
        derived=derived,
        A=derived.A,
        C=derived.C,
        yhat=derived.yhat,
        error_log=ca.Function("iekf_error_log", [R_true_sym, Rhat], [error_log]),
    )

def simulate_attitude_iekf(seed=4, T=5.0, dt=0.01, sigma_omega=0.01, sigma_y=0.03):
    sym = make_attitude_iekf_symbolics()
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, T + dt, dt)

    r1 = normalize_np(np.array([0.0, 0.0, 1.0]))
    r2 = normalize_np(np.array([0.3, 0.9, 0.1]))
    R_true = exp_so3_np(np.array([0.8, -0.6, 0.4]))
    R_hat = np.eye(3)
    P = 2.0 * np.eye(3)

    hist = {"t": times, "err_deg": np.zeros_like(times), "trace": np.zeros_like(times)}

    for k, t_now in enumerate(times):
        omega = omega_profile(float(t_now))
        omega_m = omega + sigma_omega * rng.normal(size=3)
        y_m = np.concatenate((R_true.T @ r1, R_true.T @ r2))
        y_m = y_m + sigma_y * rng.normal(size=6)

        R_hat = project_to_so3_np(R_hat @ exp_so3_np(dt * omega_m))
        P = symmetrize_psd(P + dt * ((sigma_omega**2 + 1e-6) * np.eye(3)))

        C = np.array(sym.C(R_hat, r1, r2), dtype=float)
        yhat = np.array(sym.yhat(R_hat, r1, r2), dtype=float).reshape(6)
        N = (sigma_y**2 + 1e-6) * np.eye(6)
        S = C @ P @ C.T + N
        K = P @ C.T @ np.linalg.inv(S)
        delta = K @ (y_m - yhat)
        R_hat = project_to_so3_np(exp_so3_np(delta) @ R_hat)
        I = np.eye(3)
        P = symmetrize_psd((I - K @ C) @ P @ (I - K @ C).T + K @ N @ K.T)

        hist["err_deg"][k] = attitude_error_deg(R_true, R_hat)
        hist["trace"][k] = np.trace(P)
        R_true = project_to_so3_np(R_true @ exp_so3_np(dt * omega))

    return hist

def plot_iekf_history(hist):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["t"], hist["err_deg"], color="tab:blue")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("SO(3) error [deg]")
    axes[0].grid(True)

    axes[1].plot(hist["t"], hist["trace"], color="tab:orange")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("covariance trace")
    axes[1].grid(True)
    fig.tight_layout()
    return fig, axes
