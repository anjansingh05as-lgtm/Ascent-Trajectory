"""Point-mass 2-D ascent physics.

Net force is resolved in the local vertical / downrange plane:

    Tx = T * cos(theta)
    Ty = T * sin(theta)
    D  = 0.5 * rho * v^2 * Cd * A   (opposite the velocity vector)
    g  = g0 * (R / (R + h))^2       (optional)

    ax = (Tx - D * vx / v) / m
    ay = (Ty - D * vy / v) / m - g

    mdot = T / (Isp * g0)
"""

from __future__ import annotations

import numpy as np

from atmosphere import AtmosphereModel, calculate_atmospheric_density
from config import RocketConfig

_EPS_V = 1e-9
_EPS_M = 1e-9


def calculate_gravity(altitude_m: float, config: RocketConfig) -> float:
    h = max(float(altitude_m), 0.0)
    g0 = config.sea_level_gravity
    if not config.vary_gravity_with_altitude:
        return g0
    r = config.earth_radius
    return g0 * (r / (r + h)) ** 2


def calculate_mass_flow(thrust_n: float, config: RocketConfig) -> float:
    isp_g0 = config.specific_impulse * config.g0
    if isp_g0 <= 0.0 or thrust_n <= 0.0:
        return 0.0
    return float(thrust_n / isp_g0)


def calculate_thrust(t: float, mass_kg: float, config: RocketConfig) -> float:
    """Instantaneous thrust. Zero after commanded burnout or dry-mass limit."""
    propellant = mass_kg - config.dry_mass
    if t < 0.0 or t >= config.burn_time:
        return 0.0
    if propellant <= _EPS_M:
        return 0.0
    return float(max(config.thrust, 0.0))


def calculate_drag(
    density: float,
    velocity: float,
    config: RocketConfig,
) -> float:
    return 0.5 * density * velocity * velocity * config.drag_coefficient * config.reference_area


def drag_components(density: float, vx: float, vy: float, config: RocketConfig) -> tuple[float, float, float]:
    """Return (D, Dx, Dy). Dx, Dy are aerodynamic force components ON the vehicle."""
    speed = float(np.hypot(vx, vy))
    drag_mag = calculate_drag(density, speed, config)
    if speed < _EPS_V:
        return drag_mag, 0.0, 0.0
    return drag_mag, -drag_mag * vx / speed, -drag_mag * vy / speed


def dynamic_pressure(density: float, velocity: float) -> float:
    return 0.5 * density * velocity * velocity


def acceleration_components(
    t: float,
    x: float,
    h: float,
    vx: float,
    vy: float,
    mass: float,
    config: RocketConfig,
    atmosphere: AtmosphereModel | None = None,
) -> dict[str, float]:
    """Evaluate instantaneous derived quantities and Cartesian acceleration."""
    del x  # planar model does not currently depend on downrange position
    atm = atmosphere if atmosphere is not None else AtmosphereModel(config)
    m = max(float(mass), config.dry_mass)
    density = float(atm.density(h))
    thrust = calculate_thrust(t, m, config)
    mdot = calculate_mass_flow(thrust, config)
    remaining = max(m - config.dry_mass, 0.0)
    # Prevent the ODE from requesting more propellant than remains.
    if remaining <= _EPS_M:
        mdot = 0.0
        thrust = 0.0

    speed = float(np.hypot(vx, vy))
    drag_mag, dx, dy = drag_components(density, vx, vy, config)
    g = calculate_gravity(h, config)
    theta = np.deg2rad(config.launch_angle_deg)
    tx = thrust * np.cos(theta)
    ty = thrust * np.sin(theta)
    ax = (tx + dx) / m
    ay = (ty + dy) / m - g
    acc_mag = float(np.hypot(ax, ay))
    q = dynamic_pressure(density, speed)
    return {
        "mass": m,
        "propellant": remaining,
        "thrust": thrust,
        "thrust_x": float(tx),
        "thrust_y": float(ty),
        "drag": drag_mag,
        "drag_x": dx,
        "drag_y": dy,
        "gravity": g,
        "density": density,
        "velocity": speed,
        "ax": float(ax),
        "ay": float(ay),
        "acceleration": acc_mag,
        "dynamic_pressure": q,
        "mdot": mdot,
    }


def state_derivatives(
    t: float,
    state: np.ndarray,
    config: RocketConfig,
    atmosphere: AtmosphereModel,
) -> np.ndarray:
    """ODE right-hand side: [x, h, vx, vy, m]."""
    x, h, vx, vy, mass = (float(v) for v in state)
    deriv_fields = acceleration_components(t, x, h, vx, vy, mass, config, atmosphere)
    return np.array(
        [
            vx,
            vy,
            deriv_fields["ax"],
            deriv_fields["ay"],
            -deriv_fields["mdot"],
        ],
        dtype=float,
    )


__all__ = [
    "acceleration_components",
    "calculate_atmospheric_density",
    "calculate_drag",
    "calculate_gravity",
    "calculate_mass_flow",
    "calculate_thrust",
    "drag_components",
    "dynamic_pressure",
    "state_derivatives",
]
