import numpy as np
from so2 import so2_wedge, so2_exp, so2_log

def se22_compose(X1, X2):
    return (X1[0]@X2[0], X1[0]@X2[1] + X1[1], X1[0]@X2[2] + X1[2])

def se22_inverse(X):
    return (X[0].inv(), -X[0].inv()@X[1], -X[0].inv()@X[2])

# xi = [a1, a2, b1, b2, theta]
def se22_wedge(xi):
    return np.block([
        [so2_wedge(xi[4]), np.array([[xi[0]], [xi[1]]]), np.array([[xi[2]], [xi[3]]])],
        [np.zeros((1, 2)), np.array([[1.0]]), np.array([[0.0]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[1.0]])],
    ])

def se22_vee(Xi):
    return np.array([Xi[0][2], Xi[1][2], Xi[0][3], Xi[1][3], Xi[1][0]])

def se22_exp(xi):
    if np.abs(xi[4]) < 1e-6:
        a = 1 - (xi[4]**2)/6 + (xi[4]**4)/120   
        b = xi[4]/2 - (xi[4]**3)/24 + (xi[4]**5)/720
        V = a * np.eye(2) + b * np.array([[0, -1], [1, 0]])
    else:
        sin = np.sin(xi[4])
        cos = np.cos(xi[4])
        V = (sin * np.eye(2) + (1 - cos) * np.array([[0, -1], [1, 0]])) / xi[4]
    return np.block([
        [so2_exp(xi[4]), V@np.array([[xi[0]], [xi[1]]]), V@np.array([[xi[2]], [xi[3]]])],
        [np.zeros((1, 2)), np.array([[1.0]]), np.array([[0.0]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[1.0]])],
    ])

def se22_log(X):
    w = so2_log(X[0:2, 0:2])
    if np.abs(w) < 1e-6:
        wcot_2 = 1 - (w**2)/12 - (w**4)/720 - (w**6)/30240 - (w**8)/1209600
    else:
        wcot_2 = 0.5 * w * (np.cos(w/2) / np.sin(w/2))
    
    V_inv = np.eye(2) * wcot_2 + 0.5 * w * np.array([[0, 1], [-1, 0]])
    a = V_inv @ np.array([X[0][2], X[1][2]])
    b = V_inv @ np.array([X[0][3], X[1][3]])
    return np.array([a[0], a[1], b[0], b[1], w])

def se22_Ad(X):
    return np.block([
        [X[0], np.zeros((2, 2)), np.array([[X[1][1]], [-X[1][0]]])],
        [np.zeros((2, 2)), X[0], np.array([[X[2][1]], [-X[2][0]]])],
        [np.zeros((1, 2)), np.zeros((1, 2)), np.array([[1.0]])],
    ])

def se22_ad(xi):
    return np.block([
        [so2_wedge(xi[4]), np.zeros((2, 2)), np.array([[xi[1]], [-xi[0]]])],
        [np.zeros((2, 2)), so2_wedge(xi[4]), np.array([[xi[3]], [-xi[2]]])],
        [np.zeros((1, 2)), np.zeros((1, 2)), np.array([[0.0]])],
    ])