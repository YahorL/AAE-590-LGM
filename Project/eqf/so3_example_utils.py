"""Shared SO(3) numerical helpers for the EqF examples."""

from __future__ import annotations

import casadi as ca
import cyecca.lie as lie
import numpy as np

_phi_sym = ca.SX.sym("phi_np", 3)

SO3_EXP = ca.Function(
    "SO3_EXP",
    [_phi_sym],
    [lie.so3.elem(_phi_sym).exp(lie.SO3Dcm).to_Matrix()],
)

_R_sym = ca.SX.sym("R_np", 3, 3)

SO3_LOG = ca.Function(
    "SO3_LOG",
    [_R_sym],
    [lie.SO3Dcm.from_Matrix(_R_sym).log().param],
)

def exp_so3_np(phi):
    return np.array(SO3_EXP(np.asarray(phi, dtype=float).reshape(3))).reshape(3, 3)

def log_so3_np(R):
    return np.array(SO3_LOG(np.asarray(R, dtype=float).reshape(3, 3))).reshape(3)

def hat_np(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )

def normalize_np(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n

def project_to_so3_np(R):
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float).reshape(3, 3))
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0.0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt
    return R_proj

def symmetrize_psd(M, floor=1e-10):
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    return vecs @ np.diag(np.maximum(vals, floor)) @ vecs.T

def riccati_information_step(P, process, H, R_inv, dt):
    P_pred = symmetrize_psd(P + dt * process)
    information = np.linalg.inv(P_pred) + dt * (H.T @ R_inv @ H)
    return symmetrize_psd(np.linalg.inv(information))

def angle_between_directions_deg(a, b):
    c = np.clip(abs(float(normalize_np(a) @ normalize_np(b))), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def attitude_error_deg(R_true, R_hat):
    err = log_so3_np(R_true @ R_hat.T)
    return float(np.degrees(np.linalg.norm(err)))

def omega_profile(t):
    return np.array([0.1 * np.cos(2.0 * t), 0.2 * np.sin(t), -0.1 * np.cos(1.5 * t)])
