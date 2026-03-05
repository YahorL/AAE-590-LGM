import numpy as np
from so2 import so2_wedge, so2_exp, so2_log

def se2_compose(X1, X2):
    return (X1[0] + X1[1]@X2[0], X1[1]@X2[1])

def se2_inverse(X1):
    return (-X1[1].T@X1[0], X1[1].T)

def se2_wedge(xi):
    return np.block([
    [so2_wedge(xi[2]), np.array([[xi[0]], [xi[1]]])],
    [np.zeros((1, 2)), np.array([[0.0]])],
    ])

def se2_vee(Xi):
    return np.array([Xi[0][2], Xi[1][2], Xi[1][0]])

def se2_exp(xi):
    if np.abs(xi[2]) < 1e-6:
        a = 1 - (xi[2]**2)/6 + (xi[2]**4)/120   
        b = xi[2]/2 - (xi[2]**3)/24 + (xi[2]**5)/720
        V = a * np.eye(2) + b * np.array([[0, -1], [1, 0]])
    else:
        sin = np.sin(xi[2])
        cos = np.cos(xi[2])
        V = (sin * np.eye(2) + (1 - cos) * np.array([[0, -1], [1, 0]])) / xi[2]

    return np.block([
        [so2_exp(xi[2]), V@np.array([[xi[0]], [xi[1]]])],
        [np.zeros((1, 2)), np.array([[1]])],
    ])

def se2_log(X):
    w = so2_log(X[0:2, 0:2])
    if np.abs(w) < 1e-6:
        wcot_2 = 1 - (w**2)/12 - (w**4)/720 - (w**6)/30240 - (w**8)/1209600
    else:
        wcot_2 = 0.5 * w * (np.cos(w/2) / np.sin(w/2))
    
    V_inv = np.eye(2) * wcot_2 + 0.5 * w * np.array([[0, 1], [-1, 0]])
    v = V_inv @ np.array([X[0][2], X[1][2]])
    return np.array([v[0], v[1], w])

def se2_Ad(X):
    return np.block([
        [so2_exp(X[2]), np.array([[X[1]], [-X[0]]])],
        [np.zeros((1, 2)), np.array([[1]])],
    ])
