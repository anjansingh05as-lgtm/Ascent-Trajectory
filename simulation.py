"""Numerical integration of the 2-D ascent trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from atmosphere import AtmosphereModel
from config import RocketConfig
from physics import acceleration_components, state_derivatives

_EPS_M = 1e-9


@dataclass
class Trajectory:
    time: np.ndarray
    downrange: np.ndarray
    altitude: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    velocity: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    acceleration: np.ndarray
    mass: np.ndarray
    propellant: np.ndarray
    thrust: np.ndarray
    drag: np.ndarray
    drag_x: np.ndarray
    drag_y: np.ndarray
    gravity: np.ndarray
    density: np.ndarray
    dynamic_pressure: np.ndarray
    config: RocketConfig
    events: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "time_s": self.time,
                "downrange_m": self.downrange,
                "altitude_m": self.altitude,
                "vx_mps": self.vx,
                "vy_mps": self.vy,
                "velocity_mps": self.velocity,
                "ax_mps2": self.ax,
                "ay_mps2": self.ay,
                "acceleration_mps2": self.acceleration,
                "mass_kg": self.mass,
                "propellant_kg": self.propellant,
                "thrust_n": self.thrust,
                "drag_n": self.drag,
                "gravity_mps2": self.gravity,
                "density_kgpm3": self.density,
                "dynamic_pressure_pa": self.dynamic_pressure,
            }
        )


def _initial_state(config: RocketConfig) -> np.ndarray:
    theta = np.deg2rad(config.launch_angle_deg)
    v0 = config.initial_velocity
    return np.array(
        [
            config.initial_downrange,
            config.initial_altitude,
            v0 * np.cos(theta),
            v0 * np.sin(theta),
            config.initial_mass,
        ],
        dtype=float,
    )


def _hit_ground(t: float, state: np.ndarray, config: RocketConfig) -> float:
    # Avoid triggering at the pad (h = 0 at t = 0).
    if t < max(3.0 * config.timestep, 0.2):
        return 1.0
    return float(state[1])


def _hit_ceiling(t: float, state: np.ndarray, config: RocketConfig) -> float:
    del t
    return float(config.user_maximum_altitude - state[1])


def _dry_mass_reached(t: float, state: np.ndarray, config: RocketConfig) -> float:
    del t
    return float(state[4] - config.dry_mass)


def _enrich_history(
    time: np.ndarray,
    states: np.ndarray,
    config: RocketConfig,
) -> Trajectory:
    atm = AtmosphereModel(config)
    n = time.size
    downrange = states[:, 0]
    altitude = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]
    mass_state = states[:, 4]

    ax = np.zeros(n)
    ay = np.zeros(n)
    acceleration = np.zeros(n)
    mass = np.zeros(n)
    propellant = np.zeros(n)
    thrust = np.zeros(n)
    drag = np.zeros(n)
    drag_x = np.zeros(n)
    drag_y = np.zeros(n)
    gravity = np.zeros(n)
    density = np.zeros(n)
    velocity = np.zeros(n)
    q = np.zeros(n)

    for i in range(n):
        fields = acceleration_components(
            float(time[i]),
            float(downrange[i]),
            float(altitude[i]),
            float(vx[i]),
            float(vy[i]),
            float(mass_state[i]),
            config,
            atm,
        )
        ax[i] = fields["ax"]
        ay[i] = fields["ay"]
        acceleration[i] = fields["acceleration"]
        mass[i] = fields["mass"]
        propellant[i] = fields["propellant"]
        thrust[i] = fields["thrust"]
        drag[i] = fields["drag"]
        drag_x[i] = fields["drag_x"]
        drag_y[i] = fields["drag_y"]
        gravity[i] = fields["gravity"]
        density[i] = fields["density"]
        velocity[i] = fields["velocity"]
        q[i] = fields["dynamic_pressure"]

    return Trajectory(
        time=time,
        downrange=downrange,
        altitude=altitude,
        vx=vx,
        vy=vy,
        velocity=velocity,
        ax=ax,
        ay=ay,
        acceleration=acceleration,
        mass=mass,
        propellant=propellant,
        thrust=thrust,
        drag=drag,
        drag_x=drag_x,
        drag_y=drag_y,
        gravity=gravity,
        density=density,
        dynamic_pressure=q,
        config=config,
    )


def _integrate_rk45(config: RocketConfig, rtol: float, atol: float) -> tuple[np.ndarray, np.ndarray]:
    atm = AtmosphereModel(config)
    y0 = _initial_state(config)
    t_eval = np.arange(0.0, config.max_simulation_time + config.timestep, config.timestep)

    ground = lambda t, y: _hit_ground(t, y, config)
    ground.terminal = True
    ground.direction = -1

    ceiling = lambda t, y: _hit_ceiling(t, y, config)
    ceiling.terminal = True
    ceiling.direction = -1

    def rhs(t, y):
        return state_derivatives(t, y, config, atm)

    solution = solve_ivp(
        rhs,
        (0.0, config.max_simulation_time),
        y0,
        method="RK45",
        t_eval=t_eval,
        events=(ground, ceiling),
        rtol=rtol,
        atol=atol,
        dense_output=False,
        max_step=max(config.timestep * 4.0, 0.2),
    )
    if solution.t.size == 0:
        return np.array([0.0]), y0.reshape(1, -1)
    return solution.t, solution.y.T


def _integrate_euler(config: RocketConfig) -> tuple[np.ndarray, np.ndarray]:
    atm = AtmosphereModel(config)
    dt = config.timestep
    t = 0.0
    state = _initial_state(config)
    times = [t]
    states = [state.copy()]
    while t < config.max_simulation_time - 1e-12:
        deriv = state_derivatives(t, state, config, atm)
        state = state + dt * deriv
        t = t + dt
        if state[4] < config.dry_mass:
            state[4] = config.dry_mass
        if t > 0.2 and state[1] <= 0.0 and state[3] <= 0.0:
            state[1] = 0.0
            times.append(t)
            states.append(state.copy())
            break
        if state[1] >= config.user_maximum_altitude:
            times.append(t)
            states.append(state.copy())
            break
        times.append(t)
        states.append(state.copy())
    return np.asarray(times), np.vstack(states)


def simulate_peaks(config: RocketConfig) -> tuple[float, float]:
    """Return (max_altitude, max_downrange) using the same ODE as the full sim.

    Intended for optimizer inner loops: same physics, no per-step enrichment.
    """
    atm = AtmosphereModel(config)
    y0 = _initial_state(config)

    ground = lambda t, y: _hit_ground(t, y, config)
    ground.terminal = True
    ground.direction = -1

    ceiling = lambda t, y: _hit_ceiling(t, y, config)
    ceiling.terminal = True
    ceiling.direction = -1

    def rhs(t, y):
        return state_derivatives(t, y, config, atm)

    solution = solve_ivp(
        rhs,
        (0.0, config.max_simulation_time),
        y0,
        method="RK45",
        events=(ground, ceiling),
        rtol=config.optimizer_rtol,
        atol=config.optimizer_atol,
        dense_output=False,
        max_step=1.0,
    )
    if solution.y.size == 0:
        return 0.0, 0.0
    return float(np.max(solution.y[1])), float(np.max(solution.y[0]))


def simulate_trajectory(
    config: RocketConfig,
    *,
    fast: bool = False,
) -> Trajectory:
    """Propagate the vehicle until impact, time-out, or altitude ceiling.

    `fast=True` uses optimizer tolerances / sampling and is intended only
    for inner-loop objective evaluations. Final reported cases should use
    `fast=False` so baseline and optimized results share the same integrator
    settings.
    """
    cfg = config
    if fast:
        cfg = config.copy(
            timestep=config.optimizer_timestep,
            rk45_rtol=config.optimizer_rtol,
            rk45_atol=config.optimizer_atol,
        )

    if cfg.integrator == "euler":
        time, states = _integrate_euler(cfg)
    else:
        time, states = _integrate_rk45(cfg, cfg.rk45_rtol, cfg.rk45_atol)

    trajectory = _enrich_history(time, states, config)
    # Keep event bookkeeping on the original (non-fast) config values.
    trajectory.config = config
    return trajectory
