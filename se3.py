import numpy as np
from so3 import so3_wedge, so3_exp, so3_log


def se3_compose(X1, X2):
    return (X1[0] + X1[1] @ X2[0], X1[1] @ X2[1])

def se3_inverse(X):
    return (-X[1].T @ X[0], X[1].T)

# xi = [x, y, z, wx, wy, wz]
def se3_wedge(xi):
    rho = np.array([xi[0], xi[1], xi[2]])
    phi = np.array([xi[3], xi[4], xi[5]])
    return np.block([
        [so3_wedge(phi), rho.reshape(3, 1)],
        [np.zeros((1, 3)), np.array([[0.0]])],
    ])

def se3_vee(Xi):
    return np.array([Xi[0][3], Xi[1][3], Xi[2][3], Xi[2][1], Xi[0][2], Xi[1][0]])

def se3_exp(xi):
    rho = np.array([xi[0], xi[1], xi[2]])
    phi = np.array([xi[3], xi[4], xi[5]])
    theta = np.linalg.norm(phi)
    K = so3_wedge(phi)
    if theta < 1e-6:
        a = 0.5 - (theta**2) / 24 + (theta**4) / 720
        b = 1 / 6 - (theta**2) / 120 + (theta**4) / 5040
        V = np.eye(3) + a * K + b * (K @ K)
    else:
        a = (1 - np.cos(theta)) / (theta**2)
        b = (theta - np.sin(theta)) / (theta**3)
        V = np.eye(3) + a * K + b * (K @ K)
    R = so3_exp(phi)
    t = V @ rho
    return np.block([
        [R, t.reshape(3, 1)],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])

def se3_log(X):
    R = X[0:3, 0:3]
    t = X[0:3, 3]
    phi = so3_log(R)
    theta = np.linalg.norm(phi)
    K = so3_wedge(phi)
    if theta < 1e-6:
        c = 1 / 12 + (theta**2) / 720 + (theta**4) / 30240
    else:
        c = (1 - 0.5 * theta * (np.cos(theta / 2) / np.sin(theta / 2))) / (theta**2)
    V_inv = np.eye(3) - 0.5 * K + c * (K @ K)
    rho = V_inv @ t
    return np.array([rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]])

def se3_Ad(X):
    # X is xi = [x, y, z, wx, wy, wz]: translation in group, rotation in algebra
    t = np.array([X[0], X[1], X[2]])
    phi = np.array([X[3], X[4], X[5]])
    R = so3_exp(phi)
    T = so3_wedge(t)
    return np.block([
        [R, T @ R],
        [np.zeros((3, 3)), R],
    ])
