import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from se22 import se22_exp


TEST_SEGMENTS = [
    {"a_x": 2.0, "a_y": 0.0, "w": 0.0, "t": 5.0},
    {"a_x": 0.0, "a_y": 0.0, "w": 0.0, "t": 5.0},
    {"a_x": 2.0, "a_y": 0.0, "w": 0.3, "t": 10.5},
    {"a_x": -1.0, "a_y": 0.0, "w": 0.0, "t": 5.0},
]

SPIRAL_SEGMENTS = [
    {"a_x": 1.0, "a_y": 0.0, "w": 0.5, "t": 10.0},
]

def propagate_ins(
    segments: list[dict],
    dt: float = 0.01,
) -> tuple[np.ndarray, list[int]]:
    """Propagate INS state using constant acceleration and angular velocity."""
    X = np.eye(4)
    X_list = [X.copy()]
    key_indices = [0]
    G = np.block([
        [np.eye(2), np.zeros((2, 1)), np.zeros((2, 1))],
        [np.zeros((1, 2)), np.array([[1.0]]), np.array([[dt]])],
        [np.zeros((1, 2)), np.array([[0.0]]), np.array([[1.0]])],
    ])
    for segment in segments:
        steps = int(np.round(segment["t"] / dt))
        for _ in range(steps):
            a_x = segment["a_x"]
            v_body = X[:2, :2].T @ X[:2, 2]  # rotate world-frame v into body frame
            a_y = segment["a_y"] + v_body[0] * segment["w"]
            xi = np.array([a_x, a_y, 0.0, 0.0, segment["w"]])
            X = X @ se22_exp(xi * dt) @ G
            X[2, :] = [0, 0, 1, 0]
            X[3, :] = [0, 0, 0, 1]
            X_list.append(X.copy())
        key_indices.append(len(X_list) - 1)

    return np.array(X_list), key_indices


def plot_ins_trajectory(
    ax: plt.Axes,
    X_arr: np.ndarray,
    key_indices: list[int],
    arrow_scale: float = 15.0,
) -> None:

    x = X_arr[:, 0, 3]
    y = X_arr[:, 1, 3]

    ax.plot(x, y, "b-", linewidth=1.5, label="Trajectory")

    X_key = X_arr[key_indices]
    theta = np.arctan2(X_key[:, 1, 0], X_key[:, 0, 0])
    u = arrow_scale * np.cos(theta)
    v = arrow_scale * np.sin(theta)
    ax.quiver(
        X_key[:, 0, 3], X_key[:, 1, 3], u, v,
        color="red", scale=1, scale_units="xy", angles="xy",
        width=0.005, label="Heading",
    )

    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("(x, y) trajectory with heading")
    ax.legend()


def plot_speed(ax: plt.Axes, X_arr: np.ndarray, dt: float = 0.01) -> None:
    """Plot speed |v(t)| vs time."""
    vx = X_arr[:, 0, 2]
    vy = X_arr[:, 1, 2]
    speed = np.sqrt(vx**2 + vy**2)
    t = np.arange(len(speed)) * dt
    ax.plot(t, speed, "k-")
    ax.grid(True)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|v(t)| (m/s)")
    ax.set_title("Speed vs time")


def plot_heading(ax: plt.Axes, X_arr: np.ndarray, dt: float = 0.01) -> None:
    """Plot heading theta(t) vs time."""
    theta = np.degrees(np.arctan2(X_arr[:, 1, 0], X_arr[:, 0, 0]))
    t = np.arange(len(theta)) * dt
    ax.plot(t, theta, "g-")
    ax.grid(True)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("θ(t) (deg)")
    ax.set_title("Heading vs time")


def main() -> None:
    dt = 0.01
    X_arr, key_indices = propagate_ins(TEST_SEGMENTS, dt=dt)

    X_arr_spiral, key_indices_spiral = propagate_ins(SPIRAL_SEGMENTS, dt=dt)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_ins_trajectory(axes[0, 0], X_arr, key_indices)
    plot_speed(axes[0, 1], X_arr, dt=dt)
    plot_heading(axes[1, 0], X_arr, dt=dt)
    plot_ins_trajectory(axes[1, 1], X_arr_spiral, key_indices_spiral, 5)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "ins_plots.png")
    plt.savefig(out, dpi=200)
    plt.show()
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
