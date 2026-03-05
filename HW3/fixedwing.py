import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from se2 import se2_exp


# Race-track trajectory: each segment has twist ξ_i = (V_i, v_y,i, ω_i)^T and duration T_i
RACETRACK_SEGMENTS = [
    {"V": 20.0, "v_y": 0.0, "omega": 0.0, "T": 10.0},      # Segment 0
    {"V": 15.0, "v_y": 0.0, "omega": 0.30, "T": 5.24},     # Segment 1
    {"V": 20.0, "v_y": 0.0, "omega": 0.0, "T": 5.0},       # Segment 2
    {"V": 15.0, "v_y": 0.0, "omega": 0.30, "T": 5.24},     # Segment 3
    {"V": 20.0, "v_y": 0.0, "omega": 0.0, "T": 10.0},      # Segment 4
    {"V": 15.0, "v_y": 0.0, "omega": 0.30, "T": 5.24},     # Segment 5
    {"V": 20.0, "v_y": 0.0, "omega": 0.0, "T": 5.0},       # Segment 6
    {"V": 15.0, "v_y": 0.0, "omega": 0.30, "T": 5.24},     # Segment 7
]

G = 9.81


def plot_turning_radius(
    ax: plt.Axes,
    g: float = G,
    phi_deg: float = 30.0,
    v_min: float = 10.0,
    v_max: float = 40.0,
    n_pts: int = 1000,
) -> None:
    """Plot turning radius r(V) = V^2 / (g * tan(φ)) for V in [v_min, v_max]."""
    V = np.linspace(v_min, v_max, n_pts)
    r = V**2 / (g * np.tan(np.radians(phi_deg)))
    ax.plot(V, r)
    ax.grid(True)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Radius (m)")
    ax.set_title("Radius vs Velocity")


def propagate_racetrack_trajectory(
    segments: list[dict],
    n_pts_per_seg: int = 100,
) -> np.ndarray:
    """Propagate waypoints and sample within each segment for smooth curve."""
    Xi = np.eye(3)
    X_list = [Xi.copy()]
    for segment in segments:
        t_prev = 0.0
        xi = np.array([segment["V"], segment["v_y"], segment["omega"]])
        for t in np.linspace(0, segment["T"], n_pts_per_seg):
            Xi = Xi @ se2_exp(xi * (t - t_prev))
            X_list.append(Xi.copy())
            t_prev = t
    return np.array(X_list)


def feasibility_check(
    segments: list[dict],
    g: float = G,
    max_bank_deg: float = 45.0,
) -> bool:
    """Compute required bank angle for turning segments and verify |φ| <= 45°."""
    print("Feasibility check (bank angle):")
    all_feasible = True
    for i, seg in enumerate(segments):
        if seg["omega"] != 0:
            phi_rad = np.arctan(seg["omega"] * seg["V"] / g)
            phi_deg = np.degrees(phi_rad)
            feasible = abs(phi_deg) <= max_bank_deg
            all_feasible = all_feasible and feasible
            status = "OK" if feasible else "FAIL"
            print(f"  Segment {i}: phi = {phi_deg:.2f} deg, |phi| <= 45 deg: {status}")
    print(f"  Overall: {'Feasible' if all_feasible else 'Not feasible'}")
    return all_feasible


def plot_reference_trajectory(
    ax: plt.Axes,
    X_arr: np.ndarray,
    segments: list[dict],
    n_pts_per_seg: int = 100,
    arrow_scale: float = 15.0,
) -> None:
    """Plot reference trajectory in (x, y) plane with heading arrows and segment numbers."""
    x = X_arr[:, 0, 2]
    y = X_arr[:, 1, 2]
    ax.plot(x, y, "b-", label="Trajectory")

    waypoint_indices = [0] + [n_pts_per_seg * (i + 1) for i in range(len(segments))]
    X_waypoints = X_arr[waypoint_indices]

    theta = np.arctan2(X_waypoints[:, 1, 0], X_waypoints[:, 0, 0])
    u = arrow_scale * np.cos(theta)
    v = arrow_scale * np.sin(theta)
    ax.quiver(
        X_waypoints[:, 0, 2], X_waypoints[:, 1, 2], u, v,
        color="red", scale=1, scale_units="xy", angles="xy", label="Heading"
    )

    for i, (xi, yi) in enumerate(zip(X_waypoints[:, 0, 2], X_waypoints[:, 1, 2])):
        ax.annotate(str(i), (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Race-track trajectory")
    ax.legend()


def simulate_constant_twist(
    xi: np.ndarray,
    dt: float = 0.01,
    t_end: float = 10.0,
) -> np.ndarray:
    """Propagate state X_{k+1} = X_k @ Exp(Δt * ξ) with constant twist."""
    n_steps = int(t_end / dt)
    X = np.eye(3)
    X_list = [X.copy()]
    for _ in range(n_steps):
        X = X @ se2_exp(xi * dt)
        X_list.append(X.copy())
    return np.array(X_list)


def plot_straight_line(ax: plt.Axes, X_arr: np.ndarray) -> None:
    """Plot straight-line trajectory from simulation."""
    ax.plot(X_arr[:, 0, 2], X_arr[:, 1, 2], "b-")
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Straight line (ξ = [20, 0, 0]ᵀ)")


def plot_circle(ax: plt.Axes, X_arr: np.ndarray) -> None:
    """Plot circular trajectory from simulation."""
    ax.plot(X_arr[:, 0, 2], X_arr[:, 1, 2], "r-")
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Circle (ξ = [15, 0, 0.3]ᵀ)")


def main() -> None:
    """Run all parts and display plots in a single figure."""
    X_arr = propagate_racetrack_trajectory(RACETRACK_SEGMENTS)
    feasibility_check(RACETRACK_SEGMENTS)

    X_straight = simulate_constant_twist(np.array([20.0, 0.0, 0.0]))
    X_circle = simulate_constant_twist(np.array([15.0, 0.0, 0.3]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_turning_radius(axes[0, 0])
    plot_reference_trajectory(axes[0, 1], X_arr, RACETRACK_SEGMENTS)
    plot_straight_line(axes[1, 0], X_straight)
    plot_circle(axes[1, 1], X_circle)

    plt.tight_layout()
    plt.savefig("HW3/all_plots.png")
    plt.show()

    # Save individual plots
    fig1, ax1 = plt.subplots()
    plot_turning_radius(ax1)
    #plt.savefig("HW3/radius_vs_velocity.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    plot_reference_trajectory(ax2, X_arr, RACETRACK_SEGMENTS)
    #plt.savefig("HW3/race_track_trajectory.png")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    plot_straight_line(ax3, X_straight)
    #plt.savefig("HW3/straight_line.png")
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    plot_circle(ax4, X_circle)
    #plt.savefig("HW3/circle.png")
    plt.close(fig4)


if __name__ == "__main__":
    main()
