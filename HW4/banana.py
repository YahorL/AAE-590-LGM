import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from se2 import se2_exp, se2_log

pi = np.pi

v_nom = 2
w_turn = 0.4
r_fig = 5
sigma_v = 0.05
sigma_w = 0.5

T_lobe = 2 * pi / w_turn
T_fig8 = 10 * pi
T = 35

dt = 0.1

# Nominal trajectory
X0 = np.eye(3)
n_steps = int(T / dt)
X_ref = np.zeros((n_steps + 1, 3, 3))
X_ref[0] = X0
for i in range(n_steps):
    t = i * dt
    if t % T_fig8 < T_lobe:
        w_ref = w_turn
    else:
        w_ref = -w_turn
    xi = np.array([v_nom, 0, w_ref]) * dt
    X_ref[i + 1] = X_ref[i] @ se2_exp(xi)

## Monte Carlo simulation
N = 1000
snap_times = [5, 15, 30]
snap_steps = [int(ts / dt) for ts in snap_times]
X_snaps = {ts: np.zeros((N, 3, 3)) for ts in snap_times}
X_finals = np.zeros((N, 3, 3))

for j in range(N):
    w_v = np.random.normal(0, sigma_v, n_steps)
    w_w = np.random.normal(0, sigma_w, n_steps)
    X = X0.copy()
    for i in range(n_steps):
        t = i * dt
        if t % T_fig8 < T_lobe:
            w_ref = w_turn
        else:
            w_ref = -w_turn
        xi = np.array([v_nom + w_v[i], 0, w_ref + w_w[i]]) * dt
        X = X @ se2_exp(xi)
        for ts, ns in zip(snap_times, snap_steps):
            if i + 1 == ns:
                X_snaps[ts][j] = X
    X_finals[j] = X
    if j % 100 == 0:
        print(f"Iteration {j}/{N} completed")

# Extract final positions and headings
x_final = X_finals[:, 0, 2]
y_final = X_finals[:, 1, 2]
theta_final = np.arctan2(X_finals[:, 1, 0], X_finals[:, 0, 0])

# Statistics
positions = np.vstack([x_final, y_final])
p_bar = np.mean(positions, axis=1)
S = np.cov(positions)
print(f"Sample mean p_bar = [{p_bar[0]:.4f}, {p_bar[1]:.4f}]")
print(f"Sample covariance S =\n{S}")

# Covariance ellipse
t_ell = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(t_ell), np.sin(t_ell)])
eigvals, eigvecs = np.linalg.eigh(S)
L = eigvecs @ np.diag(3 * np.sqrt(eigvals))
ellipse_pts = L @ circle + p_bar[:, None]

# Log-map errors: eps_i = Log(X_nom^{-1} @ X_i)^vee  ∈ R^3
eps = np.array([se2_log(np.linalg.inv(X_ref[-1]) @ X_finals[i]) for i in range(N)])
eps_bar = np.mean(eps, axis=0)
Sigma = np.cov(eps.T)
print(f"Lie-algebra sample mean eps_bar = {eps_bar}")
print(f"Lie-algebra sample covariance Sigma =\n{Sigma}")

# 3σ ellipsoid boundary in Lie algebra, projected to (x, y) via Exp map
phi_grid = np.linspace(0, np.pi, 50)
psi_grid = np.linspace(0, 2 * np.pi, 50)
PHI, PSI = np.meshgrid(phi_grid, psi_grid)
sphere = np.array([np.sin(PHI) * np.cos(PSI),
                    np.sin(PHI) * np.sin(PSI),
                    np.cos(PHI)])  # (3, 100, 50)

eigvals_e, eigvecs_e = np.linalg.eigh(Sigma)
VD_half = eigvecs_e @ np.diag(3 * np.sqrt(eigvals_e))  # V @ (3*D^{1/2})

X_nom = X_ref[-1]
ell_x = np.zeros_like(PHI)
ell_y = np.zeros_like(PHI)
for ii in range(PHI.shape[0]):
    for jj in range(PHI.shape[1]):
        eps_pt = eps_bar + VD_half @ sphere[:, ii, jj]
        X_pt = X_nom @ se2_exp(eps_pt)
        ell_x[ii, jj] = X_pt[0, 2]
        ell_y[ii, jj] = X_pt[1, 2]

# --- Figure 1: final distribution at t = T ---
fig1, ax1 = plt.subplots()
ax1.plot(X_ref[:, 0, 2], X_ref[:, 1, 2], "b-", label="Nominal trajectory")
ax1.scatter(x_final, y_final, s=10, c="r", zorder=3, alpha=0.5, label="Final poses")
ax1.quiver(x_final, y_final, np.cos(theta_final), np.sin(theta_final),
           scale=40, width=0.002, color="salmon", zorder=4, alpha=0.5)
ax1.plot(ellipse_pts[0], ellipse_pts[1], "g-", lw=2, label=r"Euclidean $3\sigma$ ellipse")
ax1.scatter(ell_x, ell_y, s=5, c="m", alpha=0.3, zorder=5,
            label=r"Exp($3\sigma$ ellipsoid) projection")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
ax1.axis("equal")
ax1.legend()
ax1.set_title("Monte Carlo Distribution at t = T")
fig1.tight_layout()
fig1.savefig(os.path.join(os.path.dirname(__file__), "banana_final.png"), dpi=150)

# --- Figure 2: intermediate snapshots at t = 5, 15, 30 ---
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[1, 1].set_visible(False)
plot_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
for ax, ts in zip(plot_axes, snap_times):
    Xs = X_snaps[ts]
    xs = Xs[:, 0, 2]
    ys = Xs[:, 1, 2]
    ths = np.arctan2(Xs[:, 1, 0], Xs[:, 0, 0])
    ref_idx = int(ts / dt)
    ax.plot(X_ref[:ref_idx + 1, 0, 2], X_ref[:ref_idx + 1, 1, 2], "b-",
            label="Nominal trajectory")
    ax.scatter(xs, ys, s=10, c="r", zorder=3, alpha=0.5, label="Poses")
    ax.quiver(xs, ys, np.cos(ths), np.sin(ths),
              scale=40, width=0.002, color="salmon", zorder=4, alpha=0.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"t = {ts} s")
fig2.suptitle("Intermediate Snapshots", fontsize=14)
fig2.tight_layout()
fig2.savefig(os.path.join(os.path.dirname(__file__), "banana_snapshots.png"), dpi=150)

# --- Figure 3: Lie-algebra error projections (eps1-eps2 and eps1-eps3) ---
t_ell = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(t_ell), np.sin(t_ell)])
labels = [r"$\varepsilon_1$", r"$\varepsilon_2$", r"$\varepsilon_3$"]
proj_pairs = [(0, 1), (0, 2)]

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
for ax, (i, j) in zip(axes3, proj_pairs):
    ax.scatter(eps[:, i], eps[:, j], s=5, c="r", alpha=0.5)
    sub_mean = eps_bar[[i, j]]
    sub_cov = Sigma[np.ix_([i, j], [i, j])]
    eigvals_s, eigvecs_s = np.linalg.eigh(sub_cov)
    L_s = eigvecs_s @ np.diag(3 * np.sqrt(eigvals_s))
    ell = L_s @ circle + sub_mean[:, None]
    ax.plot(ell[0], ell[1], "g-", lw=2, label=r"$3\sigma$ ellipse")
    ax.plot(sub_mean[0], sub_mean[1], "g+", ms=12, mew=2)
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])
    ax.axis("equal")
    ax.legend()
    ax.set_title(f"{labels[i]} vs {labels[j]}")
fig3.suptitle("Lie-Algebra Error Distribution", fontsize=14)
fig3.tight_layout()
fig3.savefig(os.path.join(os.path.dirname(__file__), "banana_lie_errors.png"), dpi=150)

plt.show()