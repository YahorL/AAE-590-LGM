import numpy as np


def so2_wedge(theta):
    return np.array([[0, -theta], [theta, 0]])

def so2_vee(omega):
    return omega[1][0] 

def so2_exp(theta):
    return np.eye(2)*np.cos(theta) + np.array([[0, -1], [1, 0]])*np.sin(theta)

def so2_log(R):
    return np.arctan2(R[1][0], R[0][0])
