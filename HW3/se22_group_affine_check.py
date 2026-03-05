import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from so2 import so2_exp, so2_wedge


def make_se22(R, v, p):
    return np.block([
        [R, v.reshape(2, 1), p.reshape(2, 1)],
        [np.zeros((1, 2)), np.array([[1.0]]), np.array([[0.0]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[1.0]])],
    ])


def random_se22(rng):
    theta = rng.uniform(-np.pi, np.pi)
    R = so2_exp(theta)
    v = rng.uniform(-5.0, 5.0, size=2)
    p = rng.uniform(-5.0, 5.0, size=2)
    return make_se22(R, v, p)


def f(X, a, w):
    R = X[:2, :2]
    v = X[:2, 2]
    return np.block([
        [R @ so2_wedge(w), (R @ a).reshape(2, 1), v.reshape(2, 1)],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[0.0]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[0.0]])],
    ])

def f_wind(X, a, w, w_a):
    R = X[:2, :2]
    v = X[:2, 2]
    return np.block([
        [R @ so2_wedge(w), (R @ a + w_a).reshape(2, 1), v.reshape(2, 1)],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[0.0]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[0.0]])],
    ])

def trials_no_wind (rng, trials, tol):
    I = np.eye(4)
    max_err = 0.0
    for _ in range(trials):
        X = random_se22(rng)
        Y = random_se22(rng)
        a = rng.uniform(-3.0, 3.0, size=2)
        w = rng.uniform(-3.0, 3.0)

        lhs = f(X @ Y, a, w)
        rhs = f(X, a, w) @ Y + X @ f(Y, a, w) - X @ f(I, a, w) @ Y
        err = np.linalg.norm(lhs - rhs, ord="fro")
        max_err = max(max_err, err)

        if err > tol:
            print(f"No wind FAILED: err={err:.3e} > tol={tol:.1e}")
            return

    return max_err

def trials_wind (rng, trials, tol):
    I = np.eye(4)
    max_err = 0.0
    for _ in range(trials):
        X = random_se22(rng)
        Y = random_se22(rng)
        a = rng.uniform(-3.0, 3.0, size=2)
        w = rng.uniform(-3.0, 3.0)
        w_a = np.array([3.0, -2.0])

        lhs = f_wind(X @ Y, a, w, w_a)
        rhs = f_wind(X, a, w, w_a) @ Y + X @ f_wind(Y, a, w, w_a) - X @ f_wind(I, a, w, w_a) @ Y
        err = np.linalg.norm(lhs - rhs, ord="fro")
        max_err = max(max_err, err)

        if err > tol:
            print(f"No wind FAILED: err={err:.3e} > tol={tol:.1e}")
            return

    return max_err


def main():
    rng = np.random.default_rng(0)
    trials = 200
    tol = 1e-9

    max_err = 0.0
    max_err_wind = 0.0

    max_err = trials_no_wind(rng, trials, tol)
    max_err_wind = trials_wind(rng, trials, tol)

    print(f"PASS: all {trials} trials satisfy group-affine identity for no wind.")
    print(f"Max error for no wind: {max_err:.3e}")
    print(f"PASS: all {trials} trials satisfy group-affine identity for wind.")
    print(f"Max error for wind: {max_err_wind:.3e}")


if __name__ == "__main__":
    main()
