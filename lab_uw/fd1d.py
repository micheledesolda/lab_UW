"""Finite-difference operators for the one-dimensional ultrasonic model.

This module contains only the numerical propagation kernel and receiver
sampling operators.  It deliberately has no dependency on the laboratory data
format, the PZT geometry builders, or the inversion code.  Keeping this layer
small makes it possible to verify the finite-difference scheme independently
from the effective source/receiver embedding used in the reference block and
in the DDS experiments.

Units follow the rest of :mod:`lab_uw`: distance in centimetres and time in
microseconds.  Consequently velocities are expressed in cm/us, density may
be supplied in any consistent unit, and the Kelvin--Voigt coefficient
``nu`` has units of ``length**2 / time``.  The physical constitutive
coefficients are assembled internally as ``M = rho*c**2`` and
``eta = rho*nu``.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np


BoundaryCondition = Literal[
    "legacy_fixed",
    "fixed",
    "free",
    "impedance",
    "absorbing",
]
SourceInjection = Literal[
    "legacy_increment",
    "displacement_increment",
    "body_force",
]
ReceiverQuantity = Literal["displacement", "velocity", "strain", "stress"]


def _extend_constant(model: np.ndarray, n_pad: int) -> np.ndarray:
    """Extend a nodal model with constant end values."""
    if n_pad <= 0:
        return np.asarray(model, dtype=float).copy()
    return np.pad(np.asarray(model, dtype=float), n_pad, mode="edge")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _parse_impedance(
    boundary_impedance: Optional[float | Sequence[float]],
) -> tuple[float, float]:
    """Return the left/right boundary impedances."""
    if boundary_impedance is None:
        raise ValueError(
            "boundary_impedance is required when boundary_condition='impedance'"
        )
    if np.isscalar(boundary_impedance):
        value = float(boundary_impedance)
        if value < 0:
            raise ValueError("boundary impedance must be non-negative")
        return value, value

    values = tuple(float(v) for v in boundary_impedance)
    if len(values) != 2:
        raise ValueError("boundary_impedance must be a scalar or a pair")
    if values[0] < 0 or values[1] < 0:
        raise ValueError("boundary impedance must be non-negative")
    return values


def _finite_difference_1d_kelvin_voigt_core(
    *,
    num_x: int,
    delta_x: float,
    num_t: int,
    delta_t: float,
    source_spatial_function: np.ndarray,
    source_time_function: np.ndarray,
    velocity_model: np.ndarray,
    damping_model: Optional[np.ndarray] = None,
    density_model: Optional[np.ndarray] = None,
    boundary_condition: BoundaryCondition = "free",
    boundary_impedance: Optional[float | Sequence[float]] = None,
    source_injection: SourceInjection = "body_force",
    n_pad: int = 200,
    receiver_spatial_function: Optional[np.ndarray] = None,
    receiver_quantity: ReceiverQuantity = "displacement",
    store_wavefield: bool = True,
) -> np.ndarray:
    r"""Propagate the variable-density 1D Kelvin--Voigt equation.

    The continuous equation represented by ``source_injection='body_force'``
    is

    .. math::

        \rho u_{tt}=\partial_x\left(\rho c^2u_x+\rho\nu u_{xt}\right)
        +\rho f(x)s(t).

    ``body_force`` therefore multiplies the source by ``dt**2`` in the
    leapfrog recurrence.  ``displacement_increment`` adds ``f*s`` directly to
    ``u[n+1]`` and is a discrete kinematic source, not a body force.
    ``legacy_increment`` is its backward-compatible alias.

    Boundary conditions
    -------------------
    ``free``
        Zero traction/flux at both external faces.
    ``fixed`` or ``legacy_fixed``
        Zero displacement at both external nodes.  ``legacy_fixed`` is an
        explicit name for the former implicit behaviour.
    ``impedance``
        Robin/radiation condition expressed with physical impedance.  A value
        of zero is a free surface; ``rho*c`` is approximately matched.
    ``absorbing``
        Compatibility implementation of the historical padded taper.
    """
    nx = int(num_x)
    nt = int(num_t)
    dx = float(delta_x)
    dt = float(delta_t)

    if nx < 3 or nt < 2:
        raise ValueError("the solver requires at least 3 space and 2 time samples")
    if dx <= 0 or dt <= 0:
        raise ValueError("delta_x and delta_t must be positive")

    velocity = np.asarray(velocity_model, dtype=float).reshape(-1)
    source_x = np.asarray(source_spatial_function, dtype=float).reshape(-1)
    source_t = np.asarray(source_time_function, dtype=float).reshape(-1)
    if velocity.size != nx or source_x.size != nx:
        raise ValueError("velocity_model and source_spatial_function must have num_x samples")
    if source_t.size < nt:
        raise ValueError("source_time_function must contain at least num_t samples")
    if np.any(~np.isfinite(velocity)) or np.any(velocity <= 0):
        raise ValueError("velocity_model must be finite and strictly positive")

    if damping_model is None:
        damping = np.zeros_like(velocity)
    else:
        damping = np.asarray(damping_model, dtype=float).reshape(-1)
        if damping.size != nx:
            raise ValueError("damping_model must have num_x samples")
    if np.any(~np.isfinite(damping)) or np.any(damping < 0):
        raise ValueError("damping_model must be finite and non-negative")

    if density_model is None:
        density = np.ones_like(velocity)
    else:
        density = np.asarray(density_model, dtype=float).reshape(-1)
        if density.size != nx:
            raise ValueError("density_model must have num_x samples")
    if np.any(~np.isfinite(density)) or np.any(density <= 0):
        raise ValueError("density_model must be finite and strictly positive")

    modulus = density * velocity**2
    viscosity = density * damping
    modulus_half = 0.5 * (modulus[1:] + modulus[:-1])
    viscosity_half = 0.5 * (viscosity[1:] + viscosity[:-1])
    minimum_face_density = np.minimum(density[1:], density[:-1])
    # Conservative at density jumps; exactly the historical bound when rho
    # is uniform.
    effective_c2 = max(
        float(np.max(velocity**2)),
        float(np.max(modulus_half / minimum_face_density)),
    )
    effective_nu = max(
        float(np.max(damping)),
        float(np.max(viscosity_half / minimum_face_density)),
    )
    stability_number = (
        effective_c2 * dt * dt + 2.0 * effective_nu * dt
    ) / (dx * dx)
    if stability_number > 1.0 + 1e-12:
        raise ValueError(
            "unstable explicit time step for the variable-density "
            "Kelvin--Voigt operator: conservative stability number "
            f"{stability_number:.6g} > 1"
        )

    valid_boundaries = {"legacy_fixed", "fixed", "free", "impedance", "absorbing"}
    if boundary_condition not in valid_boundaries:
        raise ValueError(f"unknown boundary_condition={boundary_condition!r}")
    if source_injection not in {
        "legacy_increment",
        "displacement_increment",
        "body_force",
    }:
        raise ValueError(f"unknown source_injection={source_injection!r}")
    if store_wavefield == (receiver_spatial_function is not None):
        raise ValueError(
            "request exactly one output: a stored wavefield or a receiver trace"
        )
    if receiver_quantity not in {"displacement", "velocity", "strain", "stress"}:
        raise ValueError(f"unknown receiver quantity={receiver_quantity!r}")
    if receiver_spatial_function is not None:
        receiver_weights = np.asarray(
            receiver_spatial_function, dtype=float
        ).reshape(-1)
        if receiver_weights.size != nx:
            raise ValueError(
                "receiver_spatial_function must contain num_x samples"
            )
    else:
        receiver_weights = None

    if boundary_condition == "absorbing":
        if n_pad < 1:
            raise ValueError("n_pad must be positive for absorbing boundaries")
        velocity_ext = _extend_constant(velocity, n_pad)
        damping_ext = _extend_constant(damping, n_pad)
        density_ext = _extend_constant(density, n_pad)
        n_ext = nx + 2 * n_pad

        taper_side = _sigmoid(np.linspace(-1.0, 6.0, n_pad))
        taper = np.ones(n_ext, dtype=float)
        taper[:n_pad] = taper_side
        taper[-n_pad:] = taper_side[::-1]
    else:
        n_pad = 0
        n_ext = nx
        velocity_ext = velocity
        damping_ext = damping
        density_ext = density
        taper = None

    modulus_ext = density_ext * velocity_ext**2
    viscosity_ext = density_ext * damping_ext
    modulus_half = 0.5 * (modulus_ext[1:] + modulus_ext[:-1])
    viscosity_half = 0.5 * (viscosity_ext[1:] + viscosity_ext[:-1])

    source_ext = np.zeros(n_ext, dtype=float)
    source_ext[n_pad : n_pad + nx] = source_x
    source_factor = dt * dt if source_injection == "body_force" else 1.0

    u_past = np.zeros(n_ext, dtype=float)
    u_curr = np.zeros(n_ext, dtype=float)
    u_fut = np.zeros(n_ext, dtype=float)
    q_half = np.zeros(n_ext - 1, dtype=float)
    dqdx = np.zeros(n_ext, dtype=float)
    wavefield = np.empty((nt, nx), dtype=float) if store_wavefield else None
    receiver_trace = np.empty(nt, dtype=float) if not store_wavefield else None
    sampled_now = sampled_previous = sampled_two_back = 0.0
    elastic_now = elastic_previous = 0.0

    inv_dx = 1.0 / dx
    dt2 = dt * dt
    impedance = _parse_impedance(boundary_impedance) if boundary_condition == "impedance" else None

    for it in range(nt):
        du_curr = u_curr[1:] - u_curr[:-1]
        du_past = u_past[1:] - u_past[:-1]
        ux_half = du_curr * inv_dx
        uxt_half = (du_curr - du_past) * (inv_dx / dt)
        q_half[:] = modulus_half * ux_half + viscosity_half * uxt_half

        if boundary_condition == "impedance":
            velocity_nodes = (u_curr - u_past) / dt
            # Outward radiation has opposite signs at the two ends.
            q_left_face = impedance[0] * velocity_nodes[0]
            q_right_face = -impedance[1] * velocity_nodes[-1]
        else:
            q_left_face = 0.0
            q_right_face = 0.0

        # The physical endpoints own half control volumes on a node-centred
        # grid.  Free/impedance boundary fluxes therefore carry a factor two.
        # Legacy fixed and padded-taper modes retain the historical stencil.
        boundary_factor = 2.0 if boundary_condition in {"free", "impedance"} else 1.0
        dqdx[0] = boundary_factor * (q_half[0] - q_left_face) * inv_dx
        dqdx[1:-1] = (q_half[1:] - q_half[:-1]) * inv_dx
        dqdx[-1] = boundary_factor * (q_right_face - q_half[-1]) * inv_dx

        u_fut[:] = 2.0 * u_curr - u_past + dt2 * dqdx / density_ext
        u_fut += source_factor * source_ext * source_t[it]

        if taper is not None:
            u_fut *= taper
        elif boundary_condition in {"fixed", "legacy_fixed"}:
            u_fut[0] = 0.0
            u_fut[-1] = 0.0
            u_curr[0] = 0.0
            u_curr[-1] = 0.0
            u_past[0] = 0.0
            u_past[-1] = 0.0

        u_past, u_curr, u_fut = u_curr, u_fut, u_past
        current = u_curr[n_pad : n_pad + nx]
        if wavefield is not None:
            wavefield[it] = current
        else:
            if receiver_quantity == "displacement":
                receiver_trace[it] = np.dot(current, receiver_weights)
            elif receiver_quantity == "velocity":
                sampled_now = float(np.dot(current, receiver_weights))
                if it == 0:
                    sampled_previous = sampled_now
                elif it == 1:
                    receiver_trace[0] = (sampled_now - sampled_previous) / dt
                    sampled_two_back = sampled_previous
                    sampled_previous = sampled_now
                else:
                    receiver_trace[it - 1] = (
                        sampled_now - sampled_two_back
                    ) / (2.0 * dt)
                    sampled_two_back = sampled_previous
                    sampled_previous = sampled_now
            else:
                strain = np.gradient(current, dx, edge_order=1)
                if receiver_quantity == "strain":
                    receiver_trace[it] = np.dot(strain, receiver_weights)
                else:
                    elastic_now = float(
                        np.dot(modulus * strain, receiver_weights)
                    )
                    sampled_now = float(np.dot(viscosity * strain, receiver_weights))
                    if it == 0:
                        elastic_previous = elastic_now
                        sampled_previous = sampled_now
                    elif it == 1:
                        receiver_trace[0] = elastic_previous + (
                            sampled_now - sampled_previous
                        ) / dt
                        sampled_two_back = sampled_previous
                        sampled_previous = sampled_now
                        elastic_previous = elastic_now
                    else:
                        receiver_trace[it - 1] = elastic_previous + (
                            sampled_now - sampled_two_back
                        ) / (2.0 * dt)
                        sampled_two_back = sampled_previous
                        sampled_previous = sampled_now
                        elastic_previous = elastic_now

    if receiver_trace is not None and receiver_quantity == "velocity":
        receiver_trace[-1] = (sampled_now - sampled_two_back) / dt
    elif receiver_trace is not None and receiver_quantity == "stress":
        receiver_trace[-1] = elastic_now + (sampled_now - sampled_two_back) / dt

    return wavefield if wavefield is not None else receiver_trace


def finite_difference_1d_kelvin_voigt(
    *,
    num_x: int,
    delta_x: float,
    num_t: int,
    delta_t: float,
    source_spatial_function: np.ndarray,
    source_time_function: np.ndarray,
    velocity_model: np.ndarray,
    damping_model: Optional[np.ndarray] = None,
    density_model: Optional[np.ndarray] = None,
    boundary_condition: BoundaryCondition = "free",
    boundary_impedance: Optional[float | Sequence[float]] = None,
    source_injection: SourceInjection = "body_force",
    n_pad: int = 200,
) -> np.ndarray:
    """Propagate and return the complete displacement wavefield."""
    return _finite_difference_1d_kelvin_voigt_core(
        num_x=num_x,
        delta_x=delta_x,
        num_t=num_t,
        delta_t=delta_t,
        source_spatial_function=source_spatial_function,
        source_time_function=source_time_function,
        velocity_model=velocity_model,
        damping_model=damping_model,
        density_model=density_model,
        boundary_condition=boundary_condition,
        boundary_impedance=boundary_impedance,
        source_injection=source_injection,
        n_pad=n_pad,
        store_wavefield=True,
    )


def finite_difference_1d_receiver_trace(
    *,
    num_x: int,
    delta_x: float,
    num_t: int,
    delta_t: float,
    source_spatial_function: np.ndarray,
    source_time_function: np.ndarray,
    receiver_spatial_function: np.ndarray,
    receiver_quantity: ReceiverQuantity,
    velocity_model: np.ndarray,
    damping_model: Optional[np.ndarray] = None,
    density_model: Optional[np.ndarray] = None,
    boundary_condition: BoundaryCondition = "free",
    boundary_impedance: Optional[float | Sequence[float]] = None,
    source_injection: SourceInjection = "body_force",
    n_pad: int = 200,
) -> np.ndarray:
    """Propagate while retaining only one receiver trace.

    This is numerically identical to propagating the full wavefield and then
    sampling it, including the one-sided endpoint and centred interior time
    derivatives.  Its memory use is ``O(num_x + num_t)`` instead of
    ``O(num_x * num_t)``, which is essential for convergence tests and
    deterministic calibration grids.
    """
    return _finite_difference_1d_kelvin_voigt_core(
        num_x=num_x,
        delta_x=delta_x,
        num_t=num_t,
        delta_t=delta_t,
        source_spatial_function=source_spatial_function,
        source_time_function=source_time_function,
        receiver_spatial_function=receiver_spatial_function,
        receiver_quantity=receiver_quantity,
        velocity_model=velocity_model,
        damping_model=damping_model,
        density_model=density_model,
        boundary_condition=boundary_condition,
        boundary_impedance=boundary_impedance,
        source_injection=source_injection,
        n_pad=n_pad,
        store_wavefield=False,
    )


def receiver_field(
    wavefield: np.ndarray,
    *,
    quantity: ReceiverQuantity,
    delta_t: float,
    delta_x: float,
    velocity_model: Optional[np.ndarray] = None,
    damping_model: Optional[np.ndarray] = None,
    density_model: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert displacement wavefield to the requested receiver observable."""
    u = np.asarray(wavefield, dtype=float)
    if u.ndim != 2:
        raise ValueError("wavefield must have shape (num_t, num_x)")

    if quantity == "displacement":
        return u
    if quantity == "velocity":
        return np.gradient(u, float(delta_t), axis=0, edge_order=1)

    strain = np.gradient(u, float(delta_x), axis=1, edge_order=1)
    if quantity == "strain":
        return strain
    if quantity != "stress":
        raise ValueError(f"unknown receiver quantity={quantity!r}")

    if velocity_model is None:
        raise ValueError("velocity_model is required for stress sampling")
    velocity = np.asarray(velocity_model, dtype=float).reshape(-1)
    if velocity.size != u.shape[1]:
        raise ValueError("velocity_model length does not match wavefield")
    if damping_model is None:
        damping = np.zeros_like(velocity)
    else:
        damping = np.asarray(damping_model, dtype=float).reshape(-1)
        if damping.size != velocity.size:
            raise ValueError("damping_model length does not match wavefield")
    if density_model is None:
        density = np.ones_like(velocity)
    else:
        density = np.asarray(density_model, dtype=float).reshape(-1)
        if density.size != velocity.size:
            raise ValueError("density_model length does not match wavefield")

    strain_rate = np.gradient(strain, float(delta_t), axis=0, edge_order=1)
    return (
        density[None, :] * velocity[None, :] ** 2 * strain
        + density[None, :] * damping[None, :] * strain_rate
    )


def sample_receiver(
    wavefield: np.ndarray,
    receiver_spatial_function: np.ndarray,
    *,
    quantity: ReceiverQuantity = "displacement",
    delta_t: float,
    delta_x: float,
    velocity_model: Optional[np.ndarray] = None,
    damping_model: Optional[np.ndarray] = None,
    density_model: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample the requested observable at the receiver.

    For displacement and velocity, first project the displacement
    wavefield onto the receiver and only then differentiate the resulting
    one-dimensional trace. This avoids allocating a second full
    (num_t, num_x) array.
    """

    u = np.asarray(wavefield)
    if u.ndim != 2:
        raise ValueError(
            "wavefield must have shape (num_t, num_x)"
        )

    weights = np.asarray(
        receiver_spatial_function,
        dtype=u.dtype,
    ).reshape(-1)

    if weights.size != u.shape[1]:
        raise ValueError(
            "receiver_spatial_function length does not match wavefield"
        )

    # Spatial projection: (num_t, num_x) @ (num_x,) -> (num_t,)
    displacement_trace = u @ weights

    if quantity == "displacement":
        return displacement_trace

    if quantity == "velocity":
        # d/dt sum_x[r(x)u(x,t)] = sum_x[r(x) du(x,t)/dt]
        return np.gradient(
            displacement_trace,
            float(delta_t),
            edge_order=1,
        )

    # Strain and stress still require spatial derivatives of the field.
    field = receiver_field(
        u,
        quantity=quantity,
        delta_t=delta_t,
        delta_x=delta_x,
        velocity_model=velocity_model,
        damping_model=damping_model,
        density_model=density_model,
    )
    return field @ weights

def interpolation_adjoint(
    source_time: np.ndarray,
    target_time: np.ndarray,
    target_gradient: np.ndarray,
) -> np.ndarray:
    """Transpose of ``np.interp(target_time, source_time, source_values)``.

    The forward workflow interpolates a simulated receiver trace onto the
    acquisition clock.  An adjoint calculation must scatter the data gradient
    with the same linear weights; interpolating the residual in the opposite
    direction is not the transpose of that operation.
    """
    xp = np.asarray(source_time, dtype=float).reshape(-1)
    x = np.asarray(target_time, dtype=float).reshape(-1)
    g = np.asarray(target_gradient, dtype=float).reshape(-1)
    if x.size != g.size:
        raise ValueError("target_time and target_gradient must have equal length")
    if xp.size < 2 or np.any(np.diff(xp) <= 0):
        raise ValueError("source_time must be strictly increasing")

    out = np.zeros(xp.size, dtype=float)
    left = x <= xp[0]
    right = x >= xp[-1]
    np.add.at(out, 0, np.sum(g[left]))
    np.add.at(out, xp.size - 1, np.sum(g[right & ~left]))

    inside = ~(left | right)
    if np.any(inside):
        xi = x[inside]
        gi = g[inside]
        j = np.searchsorted(xp, xi, side="right") - 1
        fraction = (xi - xp[j]) / (xp[j + 1] - xp[j])
        np.add.at(out, j, gi * (1.0 - fraction))
        np.add.at(out, j + 1, gi * fraction)
    return out


def receiver_adjoint(
    trace_gradient: np.ndarray,
    receiver_spatial_function: np.ndarray,
    *,
    quantity: ReceiverQuantity,
    delta_t: float,
) -> np.ndarray:
    """Transpose of the displacement or velocity receiver sampler."""
    g = np.asarray(trace_gradient, dtype=float).reshape(-1)
    r = np.asarray(receiver_spatial_function, dtype=float).reshape(-1)
    if quantity == "displacement":
        return g[:, None] * r[None, :]
    if quantity != "velocity":
        raise NotImplementedError(
            "the verified inversion adjoint currently supports displacement "
            "and velocity receivers"
        )
    if g.size < 2:
        raise ValueError("velocity receiver adjoint requires at least two time samples")

    dt = float(delta_t)
    temporal = np.zeros_like(g)
    temporal[0] -= g[0] / dt
    temporal[1] += g[0] / dt
    if g.size > 2:
        temporal[:-2] -= g[1:-1] / (2.0 * dt)
        temporal[2:] += g[1:-1] / (2.0 * dt)
    temporal[-2] -= g[-1] / dt
    temporal[-1] += g[-1] / dt
    return temporal[:, None] * r[None, :]


def _divergence_transpose(
    nodal: np.ndarray,
    dx: float,
    *,
    boundary_condition: BoundaryCondition,
) -> np.ndarray:
    """Apply the exact transpose of the flux divergence used by the solver."""
    z = np.asarray(nodal, dtype=float)
    out = np.empty(z.size - 1, dtype=float)
    boundary_factor = 2.0 if boundary_condition in {"free", "impedance"} else 1.0
    if out.size == 1:
        out[0] = (boundary_factor * z[0] - boundary_factor * z[1]) / dx
        return out
    out[0] = (boundary_factor * z[0] - z[1]) / dx
    out[1:-1] = (z[1:-2] - z[2:-1]) / dx
    out[-1] = (z[-2] - boundary_factor * z[-1]) / dx
    return out


def _gradient_transpose(halfgrid: np.ndarray, dx: float) -> np.ndarray:
    """Transpose of ``(u[1:] - u[:-1]) / dx``."""
    h = np.asarray(halfgrid, dtype=float)
    out = np.zeros(h.size + 1, dtype=float)
    out[:-1] -= h / dx
    out[1:] += h / dx
    return out


def discrete_model_gradient_1d_kelvin_voigt(
    *,
    wavefield: np.ndarray,
    wavefield_gradient: np.ndarray,
    velocity_model: np.ndarray,
    damping_model: np.ndarray,
    density_model: Optional[np.ndarray] = None,
    delta_x: float,
    delta_t: float,
    boundary_condition: BoundaryCondition,
    consume_wavefield_gradient: bool = False,
    return_density_gradient: bool = False,
) -> tuple[np.ndarray, ...]:
    """Reverse-mode derivative of the implemented leapfrog recurrence.

    Returns gradients with respect to nodal velocity and Kelvin--Voigt
    damping, plus the accumulated adjoint wavefield, while density is held
    fixed by default. With return_density_gradient=True, return (g_c, g_nu, g_rho, adjoint).
    The differentiated constitutive coefficients are
    ``M=rho*c**2`` and ``eta=rho*nu`` and the nodal acceleration contains the
    same ``1/rho`` factor as the forward recurrence. ``wavefield_gradient``
    must be the derivative of the scalar objective with respect to the stored
    displacement wavefield.  When ``consume_wavefield_gradient=True`` that
    input is used as the adjoint work array and is overwritten in place.  This
    removes one complete ``(nt, nx)`` allocation; callers must not reuse the
    supplied array afterwards.  The default preserves the former copying
    behaviour.  Free and fixed boundaries are differentiated exactly;
    absorbing padding and impedance boundaries are intentionally not enabled
    for inversion yet.
    """
    if boundary_condition not in {"free", "fixed", "legacy_fixed"}:
        raise NotImplementedError(
            "the verified discrete model gradient supports free and fixed boundaries"
        )
    u = np.asarray(wavefield, dtype=float)
    input_bar_u = np.asarray(wavefield_gradient, dtype=float)
    if consume_wavefield_gradient:
        if not input_bar_u.flags.writeable:
            raise ValueError(
                "consume_wavefield_gradient=True requires a writable array"
            )
        bar_u = input_bar_u
    else:
        bar_u = input_bar_u.copy()
    velocity = np.asarray(velocity_model, dtype=float).reshape(-1)
    damping = np.asarray(damping_model, dtype=float).reshape(-1)
    if u.shape != bar_u.shape or u.ndim != 2:
        raise ValueError("wavefield and wavefield_gradient must be matched 2D arrays")
    if u.shape[1] != velocity.size or damping.size != velocity.size:
        raise ValueError("model sizes do not match the wavefield")
    if density_model is None:
        density = np.ones_like(velocity)
    else:
        density = np.asarray(density_model, dtype=float).reshape(-1)
        if density.size != velocity.size:
            raise ValueError("density_model size does not match the wavefield")
    if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        raise ValueError("density_model must be finite and strictly positive")

    dt = float(delta_t)
    dx = float(delta_x)
    modulus = density * velocity**2
    viscosity = density * damping
    modulus_half = 0.5 * (modulus[1:] + modulus[:-1])
    viscosity_half = 0.5 * (viscosity[1:] + viscosity[:-1])
    grad_modulus_half = np.zeros_like(modulus_half)
    grad_viscosity_half = np.zeros_like(viscosity_half)
    zero = np.zeros(velocity.size, dtype=float)
    # LOCAL_DENSITY_INVERSION_V1
    grad_density_inertia = np.zeros_like(density) if return_density_gradient else None

    for k in range(u.shape[0] - 1, -1, -1):
        # The stored row k is u_next for recurrence k.  Fixed boundaries are
        # a projection applied after the raw recurrence.
        bar_next = bar_u[k].copy()
        if boundary_condition in {"fixed", "legacy_fixed"}:
            bar_next[0] = 0.0
            bar_next[-1] = 0.0

        u_curr = u[k - 1] if k >= 1 else zero
        u_prev = u[k - 2] if k >= 2 else zero
        grad_curr = (u_curr[1:] - u_curr[:-1]) / dx
        grad_delta = (
            (u_curr[1:] - u_curr[:-1])
            - (u_prev[1:] - u_prev[:-1])
        ) / dx

        if return_density_gradient:
            # Differentiate the nodal inverse mass at fixed M and eta.
            # The existing source is acceleration f*s, independent of rho.
            traction = modulus_half * grad_curr + viscosity_half * grad_delta / dt
            divergence = np.empty_like(density)
            factor = 2.0 if boundary_condition == "free" else 1.0
            divergence[0] = factor * traction[0] / dx
            divergence[1:-1] = (traction[1:] - traction[:-1]) / dx
            divergence[-1] = -factor * traction[-1] / dx
            grad_density_inertia -= dt*dt * bar_next * divergence / (density*density)

        # u_next contains dt**2 * D(q) / rho.  Density is fixed, so its
        # reciprocal weights the recurrence adjoint before D.T is applied.
        div_t = _divergence_transpose(
            bar_next / density, dx, boundary_condition=boundary_condition
        )
        grad_modulus_half += (dt * dt) * div_t * grad_curr
        grad_viscosity_half += dt * div_t * grad_delta

        elastic_t = _gradient_transpose(modulus_half * div_t, dx)
        viscous_t = _gradient_transpose(viscosity_half * div_t, dx)
        bar_curr = 2.0 * bar_next + (dt * dt) * elastic_t + dt * viscous_t
        bar_prev = -bar_next - dt * viscous_t
        if k >= 1:
            bar_u[k - 1] += bar_curr
        if k >= 2:
            bar_u[k - 2] += bar_prev

    grad_modulus = np.zeros_like(velocity)
    grad_modulus[:-1] += 0.5 * grad_modulus_half
    grad_modulus[1:] += 0.5 * grad_modulus_half
    grad_viscosity = np.zeros_like(damping)
    grad_viscosity[:-1] += 0.5 * grad_viscosity_half
    grad_viscosity[1:] += 0.5 * grad_viscosity_half

    # M=rho*c**2 and eta=rho*nu, with rho fixed.
    grad_velocity = 2.0 * density * velocity * grad_modulus
    grad_damping = density * grad_viscosity
    if return_density_gradient:
        # Chain rule back to (c, nu, rho) at fixed c and nu.
        grad_density = grad_density_inertia + velocity**2*grad_modulus + damping*grad_viscosity
        return grad_velocity, grad_damping, grad_density, bar_u
    return grad_velocity, grad_damping, bar_u
