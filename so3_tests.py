import pytest

import numpy as np
from so3 import so3_wedge, so3_vee, so3_exp, so3_log


@pytest.mark.parametrize("omega,expected_wedge", [
    (np.array([0, 0, 0]), np.zeros((3, 3))),
    (np.array([1, 0, 0]), np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])),
    (np.array([0, 1, 0]), np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])),
    (np.array([0, 0, 1]), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])),
    (np.array([1, 2, 3]), np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])),
])
def test_so3_wedge(omega, expected_wedge):
    result = so3_wedge(omega)
    np.testing.assert_array_almost_equal(result, expected_wedge)


@pytest.mark.parametrize("Omega,expected_vee", [
    (np.zeros((3, 3)), np.array([0, 0, 0])),
    (np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]]), np.array([1, 2, 3])),
    (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]), np.array([1, 0, 0])),
])
def test_so3_vee(Omega, expected_vee):
    result = so3_vee(Omega)
    np.testing.assert_array_almost_equal(result, expected_vee)


@pytest.mark.parametrize("omega,expected_R", [
    (np.array([0, 0, 0]), np.eye(3)),
    (np.array([np.pi/2, 0, 0]), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
    (np.array([0, np.pi/2, 0]), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])),
    (np.array([0, 0, np.pi/2]), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
    (np.array([np.pi, 0, 0]), np.diag([1, -1, -1])),
])
def test_so3_exp(omega, expected_R):
    result = so3_exp(omega)
    np.testing.assert_array_almost_equal(result, expected_R)


def test_so3_exp_is_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        omega = rng.uniform(-np.pi, np.pi, size=3)
        R = so3_exp(omega)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


@pytest.mark.parametrize("R", [
    np.eye(3),
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    np.diag([1.0, -1.0, -1.0]),
])
def test_so3_exp_log(R):
    result = so3_exp(so3_log(R))
    np.testing.assert_array_almost_equal(result, R)


def _random_omega(rng, max_norm=np.pi - 1e-3):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, max_norm)
    return axis * angle


def test_so3_log_exp_random():
    rng = np.random.default_rng(1)
    for _ in range(200):
        omega = _random_omega(rng)
        result = so3_log(so3_exp(omega))
        np.testing.assert_array_almost_equal(result, omega)


def test_so3_exp_log_random():
    rng = np.random.default_rng(2)
    for _ in range(200):
        omega = _random_omega(rng)
        R = so3_exp(omega)
        np.testing.assert_array_almost_equal(so3_exp(so3_log(R)), R)


def test_so3_exp_composition_same_axis():
    rng = np.random.default_rng(3)
    for _ in range(50):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        a1 = rng.uniform(-1, 1)
        a2 = rng.uniform(-1, 1)
        R1 = so3_exp(axis * a1)
        R2 = so3_exp(axis * a2)
        np.testing.assert_array_almost_equal(R1 @ R2, so3_exp(axis * (a1 + a2)))


def test_so3_wedge_vee_inverse():
    rng = np.random.default_rng(4)
    for _ in range(20):
        omega = rng.normal(size=3)
        np.testing.assert_array_almost_equal(so3_vee(so3_wedge(omega)), omega)
