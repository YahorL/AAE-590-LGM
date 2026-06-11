import numpy as np


def so3_wedge(omega):
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0],
    ])

def so3_vee(Omega):
    return np.array([Omega[2][1], Omega[0][2], Omega[1][0]])

def so3_exp(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3) + so3_wedge(omega)
    K = so3_wedge(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def so3_log(R):
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return so3_vee(R - np.eye(3))
    if np.pi - theta < 1e-6:
        # near pi: use diagonal of R = I + 2*K^2 to recover axis
        diag = np.clip((np.diag(R) + 1) / 2, 0.0, 1.0)
        axis = np.sqrt(diag)
        # fix signs using off-diagonals
        if axis[0] > 1e-6:
            axis[1] = np.sign(R[0, 1] + R[1, 0]) * axis[1]
            axis[2] = np.sign(R[0, 2] + R[2, 0]) * axis[2]
        elif axis[1] > 1e-6:
            axis[2] = np.sign(R[1, 2] + R[2, 1]) * axis[2]
        return theta * axis
    return theta / (2 * np.sin(theta)) * so3_vee(R - R.T)
