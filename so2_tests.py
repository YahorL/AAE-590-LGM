import pytest

import numpy as np
from so2 import so2_wedge, so2_vee, so2_exp, so2_log

@pytest.mark.parametrize("theta,expected_wedge", [
    (0, np.array([[0, 0], [0, 0]])),
    (np.pi/4, np.array([[0, -np.pi/4], [np.pi/4, 0]])),
    (np.pi/2, np.array([[0, -np.pi/2], [np.pi/2, 0]])),
])
def test_so2_wedge(theta, expected_wedge):
    result = so2_wedge(theta)
    np.testing.assert_array_almost_equal(result, expected_wedge)


@pytest.mark.parametrize("omega,expected_vee", [
    (np.array([[0, -1.5], [1.5, 0]]), 1.5),
    (np.array([[0, -2.0], [2.0, 0]]), 2.0),
    (np.array([[0, 0], [0, 0]]), 0.0),
])
def test_so2_vee(omega, expected_vee):
    result = so2_vee(omega)
    assert abs(result - expected_vee) < 1e-10


@pytest.mark.parametrize("theta,expected_R", [
    (0, np.eye(2)),
    (np.pi/2, np.array([[0, -1], [1, 0]])),
    (np.pi, -np.eye(2)),
],)
def test_so2_exp(theta, expected_R):
    result = so2_exp(theta)
    np.testing.assert_array_almost_equal(result, expected_R)


@pytest.mark.parametrize("R,expected_theta", [
    (np.eye(2), 0.0),
    (np.array([[0, -1], [1, 0]]), np.pi/2),
    (np.array([[-1, 0], [0, -1]]), np.pi),
])
def test_so2_log(R, expected_theta):
    result = so2_log(R)
    assert abs(result - expected_theta) < 1e-10


@pytest.mark.parametrize("R", [
    (np.eye(2)),
    (np.array([[0, -1], [1, 0]])),
    (np.array([[-1, 0], [0, -1]])),
])
def test_so2_exp_log(R):
    result = so2_exp(so2_log(R))
    np.testing.assert_array_almost_equal(result, R)


angles = np.linspace(-np.pi, np.pi, num=1000)
@pytest.mark.parametrize("theta", angles)
def test_so2_log_exp(theta):
    result = so2_log(so2_exp(theta))
    np.testing.assert_array_almost_equal(result, theta)


theta1_range = np.linspace(-np.pi, np.pi, num=20)
theta2_range = np.linspace(-np.pi, np.pi, num=20)
@pytest.mark.parametrize("theta1", theta1_range)
@pytest.mark.parametrize("theta2", theta2_range)
def test_so2_exp_composition(theta1, theta2):
    R1 = so2_exp(theta1)
    R2 = so2_exp(theta2)
    R_composed = R1 @ R2
    R_expected = so2_exp(theta1 + theta2)
    np.testing.assert_array_almost_equal(R_composed, R_expected)


