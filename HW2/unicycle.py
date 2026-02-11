import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from se2 import se2_wedge, se2_exp
from so2 import so2_exp

v = 1;
w = 0.5;
dt = 0.1;
tf = 20;
X0 = np.eye(3);

def euler_sim(X0, tf, dt, v, w):
    t = 0
    X = X0
    xi = np.array([v, 0, w])
    t_list = [t]
    X_list = [X.copy()]
    error_list = [0]

    while t < tf:
        X = X@(np.eye(3) + se2_wedge(xi * dt))
        R = X[0:2, 0:2]
        error = np.linalg.norm(R.T @ R - np.eye(2), ord='fro')
        t = t + dt
        t_list.append(t)
        X_list.append(X.copy())
        error_list.append(error)

    return t_list, X_list, error_list

def lie_sim(X0, tf, dt, v, w):
    t = 0
    X = X0
    xi = np.array([v, 0, w])
    t_list = [t]
    X_list = [X.copy()]
    error_list = [0]
    while t < tf:
        X = X@se2_exp(xi * dt)
        R = X[0:2, 0:2]
        error = np.linalg.norm(R.T @ R - np.eye(2), ord='fro')
        t = t + dt
        t_list.append(t)
        X_list.append(X.copy())
        error_list.append(error)
    return t_list, X_list, error_list


def plot_trajectories(X_euler_list, X_lie_list):
    x_euler = [X[0, 2] for X in X_euler_list]
    y_euler = [X[1, 2] for X in X_euler_list]
    x_lie = [X[0, 2] for X in X_lie_list]
    y_lie = [X[1, 2] for X in X_lie_list]

    plt.figure()
    plt.plot(x_euler, y_euler, label="Euler")
    plt.plot(x_lie, y_lie, label="Lie")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.title("Trajectories")
    output_path = os.path.join(os.path.dirname(__file__), "trajectories.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_error(t_euler, t_lie, error_euler, error_lie):
    plt.figure()
    plt.plot(t_euler, error_euler, label="Euler")
    plt.plot(t_lie, error_lie, label="Lie")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Error")
    output_path = os.path.join(os.path.dirname(__file__), "error.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    t_euler, X_euler, error_euler = euler_sim(X0, tf, dt, v, w)
    t_lie, X_lie, error_lie = lie_sim(X0, tf, dt, v, w)
    plot_trajectories(X_euler, X_lie)
    plot_error(t_euler, t_lie, error_euler, error_lie)
    xi = np.array([v, 0, w])
    X_exact = se2_exp(xi * tf)
    X_lie_final = X_lie[-1]
    exact_error = np.linalg.norm(X_lie_final - X_exact, ord="fro")
    print(f"Exact solution error at t={tf}: {exact_error:.3e}")
