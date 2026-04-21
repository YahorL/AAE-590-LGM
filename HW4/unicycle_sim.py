import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import sympy as sp
import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from se2 import se2_exp

# ============================================================
# Part (b): Build ODE from rumoca-generated symbolic model
# ============================================================
sys.path.insert(0, HERE)
from rumoca import Model, der

model = Model()
print("rumoca model summary:", model.summary())

targets = model.explicit_targets
residuals = list(model.f_x)
solution = {}
for target, residual in zip(targets, residuals):
    eq = residual.subs(solution)
    sol_list = sp.solve(eq, target)
    solution[target] = sol_list[0]

omega_turn_val = float(model.p_start["omega_turn"])
pi_val         = float(model.p_start["pi"])
v_nom_val      = float(model.p_start["v_nom"])
T_lobe_val     = 2 * pi_val / omega_turn_val
T_fig8_val     = 2 * T_lobe_val

param_subs = {sym: val for sym, val in zip(
    model.p, [T_fig8_val, T_lobe_val, omega_turn_val, pi_val, v_nom_val]
)}

theta_sym, x_sym, y_sym = model.x
dtheta_expr = solution[der(theta_sym)].subs(param_subs)
dx_expr     = solution[der(x_sym)].subs(param_subs)
dy_expr     = solution[der(y_sym)].subs(param_subs)

_f = sp.lambdify(
    (model.time, theta_sym),
    [dtheta_expr, dx_expr, dy_expr],
    modules="numpy",
)

def ode_rhs(t, state):
    theta = state[0]
    d = _f(t, theta)
    return [float(d[0]), float(d[1]), float(d[2])]

T = 35.0
x0 = [float(model.x_start[s]) for s in ["theta", "x", "y"]]
t_eval = np.linspace(0, T, 3501)

print(f"\nIntegrating ODE (RK45, rtol=1e-10) for T = {T} s ...")
sol = solve_ivp(
    ode_rhs, [0, T], x0,
    method="RK45",
    t_eval=t_eval,
    rtol=1e-10, atol=1e-12,
    max_step=0.01,
)
assert sol.success, f"solve_ivp failed: {sol.message}"
print(f"  nfev = {sol.nfev}")

t_mod     = sol.t
theta_mod = sol.y[0]
x_mod     = sol.y[1]
y_mod     = sol.y[2]

dt_py   = 0.1
n_steps = int(T / dt_py)
X_ref   = np.zeros((n_steps + 1, 3, 3))
X_ref[0] = np.eye(3)

for i in range(n_steps):
    t = i * dt_py
    tau = t % T_fig8_val
    w_ref = omega_turn_val if tau < T_lobe_val else -omega_turn_val
    xi = np.array([v_nom_val, 0, w_ref]) * dt_py
    X_ref[i + 1] = X_ref[i] @ se2_exp(xi)

t_py     = np.arange(n_steps + 1) * dt_py
x_py     = X_ref[:, 0, 2]
y_py     = X_ref[:, 1, 2]
theta_py = np.arctan2(X_ref[:, 1, 0], X_ref[:, 0, 0])

x_mod_at_py = interp1d(t_mod, x_mod, kind="cubic")(t_py)
y_mod_at_py = interp1d(t_mod, y_mod, kind="cubic")(t_py)

pos_err     = np.sqrt((x_mod_at_py - x_py)**2 + (y_mod_at_py - y_py)**2)
max_pos_err = np.max(pos_err)
idx_max     = np.argmax(pos_err)

print(f"  max_k ||p_Modelica - p_Python|| = {max_pos_err:.6e} m")
print(f"  at t = {t_py[idx_max]:.1f} s")


fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(x_mod, y_mod, "b-", lw=2, label="Modelica")
ax.plot(x_py, y_py, "r--", lw=1.5, alpha=0.8,
        label=f"Python (Lie-group dt={dt_py} s)")

arrow_times = [0, 5, 10, 15, 20, 25, 30, 34]
for ta in arrow_times:
    idx = np.argmin(np.abs(t_mod - ta))
    cx, cy = x_mod[idx], y_mod[idx]
    dx = np.cos(theta_mod[idx])
    dy = np.sin(theta_mod[idx])
    ax.annotate(
        "", xy=(cx + 0.7 * dx, cy + 0.7 * dy), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="->", color="darkblue", lw=2),
    )
    ax.plot(cx, cy, "bo", ms=5, zorder=5)
    ax.text(cx + 0.9 * dx, cy + 0.9 * dy, f"t={ta} s",
            fontsize=8, ha="center", va="bottom")

ax.set_xlabel("x [m]", fontsize=12)
ax.set_ylabel("y [m]", fontsize=12)
ax.set_title("Modelica vs Lie Integration",
             fontsize=14)
ax.legend(fontsize=11, loc="best")
ax.axis("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()
out_path = os.path.join(HERE, "unicycle_trajectory.png")
fig.savefig(out_path, dpi=150)
print(f"\nFigure saved to {out_path}")
plt.show()
