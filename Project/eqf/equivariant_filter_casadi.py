"""Generic CasADi derivation of equivariant-filter matrices.

Users provide a cyecca Lie group plus CasADi maps for the state action,
input action, output, lift, chart, and local error embedding.  The derivation
uses CasADi automatic differentiation and the supplied cyecca group API for
exponential, inverse, and adjoint operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import casadi as ca
import numpy as np


def _as_col(x: ca.SX | ca.MX | ca.DM) -> ca.SX | ca.MX | ca.DM:
    return ca.reshape(x, x.numel(), 1)


def _restore_shape(x: ca.SX, shape: tuple[int, int]) -> ca.SX:
    return ca.reshape(x, shape)


def _substitute(expr: ca.SX, var: ca.SX, val: ca.SX | ca.DM) -> ca.SX:
    expr_shape = expr.shape
    substituted = ca.substitute(_as_col(expr), _as_col(var), _as_col(val))
    return _restore_shape(substituted, expr_shape)


@dataclass(frozen=True)
class EqFProblem:
    """Definition of an EqF problem in embedded CasADi coordinates.

    The supplied maps define the geometry.  This class is independent of any
    particular state space, symmetry group, or output model.  The input space
    is assumed to be a vector space because ``B`` is computed by
    differentiating with respect to an additive input perturbation.

    ``lie_group`` is a cyecca Lie group.  ``group`` may use either the group's
    native parameter coordinates or its matrix representation; matrix
    coordinates are detected from ``lie_group.matrix_shape``.
    """

    name: str
    state: ca.SX
    group: ca.SX
    input: ca.SX
    error: ca.SX
    output_measurement: ca.SX
    lie_group: object
    origin: ca.SX | ca.DM
    parameters: tuple[ca.SX, ...]
    action: Callable[[ca.SX, ca.SX], ca.SX]
    input_action: Callable[[ca.SX, ca.SX], ca.SX]
    output: Callable[[ca.SX], ca.SX]
    output_action: Callable[[ca.SX, ca.SX], ca.SX] | None
    lift: Callable[[ca.SX, ca.SX], ca.SX]
    chart_inv: Callable[[ca.SX], ca.SX]
    algebra_from_error: Callable[[ca.SX], ca.SX]
    chart_projector: ca.SX | ca.DM | None = None


@dataclass(frozen=True)
class EqFDerivation:
    """CasADi-derived EqF matrices and intermediate expressions."""

    problem: EqFProblem
    A: ca.Function
    B: ca.Function
    C: ca.Function
    C_star: ca.Function | None
    yhat: ca.Function
    chart_inv: ca.Function
    chart_projector: ca.DM
    A_expr: ca.SX
    B_expr: ca.SX
    C_expr: ca.SX
    C_star_expr: ca.SX | None
    yhat_expr: ca.SX


def _uses_group_matrix_coordinates(lie_group: object, coordinates: ca.SX) -> bool:
    matrix_shape = getattr(lie_group, "matrix_shape", None)
    return matrix_shape is not None and tuple(coordinates.shape) == tuple(matrix_shape)


def _group_element(problem: EqFProblem, coordinates: ca.SX):
    if _uses_group_matrix_coordinates(problem.lie_group, coordinates):
        return problem.lie_group.from_Matrix(coordinates)
    return problem.lie_group.elem(_as_col(coordinates))


def _group_coordinates(problem: EqFProblem, group_element) -> ca.SX:
    if _uses_group_matrix_coordinates(problem.lie_group, problem.group):
        return group_element.to_Matrix()
    return ca.reshape(group_element.param, problem.group.shape)


def _algebra_element(problem: EqFProblem, coordinates: ca.SX):
    return problem.lie_group.algebra.elem(_as_col(coordinates))


def _exp_coordinates(problem: EqFProblem, algebra_coordinates: ca.SX) -> ca.SX:
    return _group_coordinates(
        problem,
        _algebra_element(problem, algebra_coordinates).exp(problem.lie_group),
    )


def _inverse_coordinates(problem: EqFProblem, group_coordinates: ca.SX) -> ca.SX:
    return _group_coordinates(problem, _group_element(problem, group_coordinates).inverse())


def _adjoint_matrix(problem: EqFProblem, group_coordinates: ca.SX, algebra: ca.SX) -> ca.SX:
    group_element = _group_element(problem, group_coordinates)
    try:
        adjoint = group_element.Ad()
    except TypeError as exc:
        # Some cyecca matrix-coordinate groups have an Ad() implementation
        # that fails even though their matrix representation is the adjoint.
        algebra_dim = _as_col(algebra).numel()
        if tuple(group_coordinates.shape) == (algebra_dim, algebra_dim):
            adjoint = group_element.to_Matrix()
        else:
            raise exc

    algebra_dim = _as_col(algebra).numel()
    if adjoint.shape == (algebra_dim, algebra_dim):
        return adjoint

    if type(problem.lie_group).__module__.endswith(".group_rn") and adjoint.shape == (
        algebra_dim + 1,
        algebra_dim + 1,
    ):
        return ca.SX.eye(algebra_dim)

    raise ValueError(
        f"{problem.name}: cyecca adjoint has shape {adjoint.shape}, "
        f"but algebra coordinate dimension is {algebra_dim}"
    )


def _adjoint_coordinates(
    problem: EqFProblem, group_coordinates: ca.SX, algebra_coordinates: ca.SX
) -> ca.SX:
    result = _adjoint_matrix(problem, group_coordinates, algebra_coordinates) @ _as_col(
        algebra_coordinates
    )
    return _restore_shape(result, algebra_coordinates.shape)


def _infinitesimal_action(problem: EqFProblem, state: ca.SX, algebra: ca.SX) -> ca.SX:
    tau = ca.SX.sym(f"{problem.name}_tau")
    curve = problem.action(_exp_coordinates(problem, tau * algebra), state)
    return _substitute(ca.jacobian(_as_col(curve), tau), tau, ca.SX(0))


def _chart_projector(problem: EqFProblem) -> ca.DM:
    if problem.chart_projector is not None:
        return ca.DM(problem.chart_projector)

    chart_inv_expr = problem.chart_inv(problem.error)
    jacobian_expr = ca.jacobian(_as_col(chart_inv_expr), _as_col(problem.error))
    jacobian_at_origin = ca.Function(
        f"{problem.name}_chart_inv_jacobian_at_origin",
        [problem.error, *problem.parameters],
        [jacobian_expr],
    )(
        ca.DM.zeros(problem.error.numel(), 1),
        *[ca.DM.zeros(parameter.shape) for parameter in problem.parameters],
    )
    return ca.DM(np.linalg.pinv(np.array(jacobian_at_origin, dtype=float)))


def derive_equivariant_filter(problem: EqFProblem) -> EqFDerivation:
    """Derive EqF matrices from an :class:`EqFProblem`.

    The returned functions are:

    - ``A(X, u, *params)``: deterministic error-state Jacobian.
    - ``B(X, u, *params)``: input-noise Jacobian.
    - ``C(X, *params)``: standard output-residual Jacobian.
    - ``C_star(X, y, *params)``: equivariant output-residual Jacobian, when
      ``problem.output_action`` is supplied.
    - ``yhat(X, *params)``: predicted output at the transformed origin.
    """

    X = problem.group
    u = problem.input
    eps = problem.error
    y = problem.output_measurement
    params = problem.parameters
    origin = problem.origin
    zero_eps = ca.SX.zeros(eps.shape)
    du = ca.SX.sym(f"{problem.name}_du", u.numel(), 1)

    chart_projector = _chart_projector(problem)
    X_inv = _inverse_coordinates(problem, X)
    u_origin = problem.input_action(X_inv, u)
    e_eps = problem.chart_inv(eps)

    lift_error = problem.lift(e_eps, u_origin) - problem.lift(origin, u_origin)
    e_dot = _infinitesimal_action(problem, e_eps, lift_error)
    eps_dot = chart_projector @ _as_col(e_dot)
    A_expr = _substitute(ca.jacobian(eps_dot, eps), eps, zero_eps)

    u_origin_perturbed = problem.input_action(X_inv, u + du)
    lift_input_error = problem.lift(origin, u_origin_perturbed) - problem.lift(
        origin, u_origin
    )
    e_dot_input = _infinitesimal_action(problem, origin, lift_input_error)
    B_expr = _substitute(
        ca.jacobian(chart_projector @ _as_col(e_dot_input), du),
        du,
        ca.SX.zeros(du.shape),
    )

    xi_hat = problem.action(X, origin)
    yhat_expr = _as_col(problem.output(xi_hat))
    residual = _as_col(problem.output(problem.action(X, e_eps))) - yhat_expr
    C_expr = _substitute(ca.jacobian(residual, eps), eps, zero_eps)

    C_star_expr = None
    C_star = None
    if problem.output_action is not None:
        tau = ca.SX.sym(f"{problem.name}_rho_tau")
        algebra_eps = _adjoint_coordinates(
            problem,
            X_inv,
            problem.algebra_from_error(eps),
        )
        output_group_curve = _exp_coordinates(problem, tau * algebra_eps)
        dy_at_y = _substitute(
            ca.jacobian(_as_col(problem.output_action(output_group_curve, y)), tau),
            tau,
            ca.SX(0),
        )
        dy_at_yhat = _substitute(
            ca.jacobian(
                _as_col(problem.output_action(output_group_curve, yhat_expr)),
                tau,
            ),
            tau,
            ca.SX(0),
        )
        C_star_expr = ca.jacobian(0.5 * (dy_at_y + dy_at_yhat), eps)
        C_star = ca.Function(f"{problem.name}_C_star", [X, y, *params], [C_star_expr])

    return EqFDerivation(
        problem=problem,
        A=ca.Function(f"{problem.name}_A", [X, u, *params], [A_expr]),
        B=ca.Function(f"{problem.name}_B", [X, u, *params], [B_expr]),
        C=ca.Function(f"{problem.name}_C", [X, *params], [C_expr]),
        C_star=C_star,
        yhat=ca.Function(f"{problem.name}_yhat", [X, *params], [yhat_expr]),
        chart_inv=ca.Function(f"{problem.name}_chart_inv", [eps, *params], [e_eps]),
        chart_projector=chart_projector,
        A_expr=A_expr,
        B_expr=B_expr,
        C_expr=C_expr,
        C_star_expr=C_star_expr,
        yhat_expr=yhat_expr,
    )
