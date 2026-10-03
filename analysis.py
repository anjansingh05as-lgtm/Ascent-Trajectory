"""Performance metrics, event detection, validation, and comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config import RocketConfig
from simulation import Trajectory


@dataclass
class PerformanceMetrics:
    maximum_altitude_m: float
    maximum_velocity_mps: float
    maximum_acceleration_mps2: float
    average_acceleration_mps2: float
    time_to_apogee_s: float
    apogee_altitude_m: float
    total_flight_time_s: float
    maximum_downrange_m: float
    downrange_at_apogee_m: float
    initial_mass_kg: float
    final_mass_kg: float
    propellant_consumed_kg: float
    propellant_remaining_kg: float
    commanded_burn_time_s: float
    actual_burn_time_s: float
    launch_angle_deg: float
    thrust_n: float
    drag_coefficient: float
    dry_mass_kg: float
    maximum_dynamic_pressure_pa: float
    time_of_max_q_s: float
    time_of_meco_s: float
    time_of_impact_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def find_apogee(trajectory: Trajectory) -> tuple[float, float, float]:
    """Return (time, altitude, downrange) at apogee.

    Apogee is the sample of maximum altitude, refined by linear interpolation
    of the vertical-velocity zero crossing when that crossing exists.
    """
    if trajectory.time.size == 0:
        return 0.0, 0.0, 0.0

    i_max = int(np.argmax(trajectory.altitude))
    t_peak = float(trajectory.time[i_max])
    h_peak = float(trajectory.altitude[i_max])
    x_peak = float(trajectory.downrange[i_max])

    vy = trajectory.vy
    sign_change = np.where((vy[:-1] > 0.0) & (vy[1:] <= 0.0))[0]
    if sign_change.size:
        i = int(sign_change[0])
        y0, y1 = float(vy[i]), float(vy[i + 1])
        if abs(y1 - y0) > 1e-12:
            frac = y0 / (y0 - y1)
            frac = min(max(frac, 0.0), 1.0)
            t_peak = float(trajectory.time[i] + frac * (trajectory.time[i + 1] - trajectory.time[i]))
            h_peak = float(trajectory.altitude[i] + frac * (trajectory.altitude[i + 1] - trajectory.altitude[i]))
            x_peak = float(trajectory.downrange[i] + frac * (trajectory.downrange[i + 1] - trajectory.downrange[i]))
    return t_peak, h_peak, x_peak


def _last_powered_time(trajectory: Trajectory) -> float:
    burning = np.where(trajectory.thrust > 1e-6)[0]
    if burning.size == 0:
        return 0.0
    return float(trajectory.time[burning[-1]])


def calculate_performance_metrics(trajectory: Trajectory) -> PerformanceMetrics:
    cfg = trajectory.config
    t_apogee, h_apogee, x_apogee = find_apogee(trajectory)
    i_q = int(np.argmax(trajectory.dynamic_pressure)) if trajectory.time.size else 0
    consumed = max(cfg.propellant_mass - float(trajectory.propellant[-1]), 0.0)
    trajectory.events = {
        "liftoff_s": float(trajectory.time[0]) if trajectory.time.size else 0.0,
        "max_q_s": float(trajectory.time[i_q]) if trajectory.time.size else 0.0,
        "meco_s": _last_powered_time(trajectory),
        "apogee_s": t_apogee,
        "impact_s": float(trajectory.time[-1]) if trajectory.time.size else 0.0,
        "max_q_pa": float(trajectory.dynamic_pressure[i_q]) if trajectory.time.size else 0.0,
        "apogee_altitude_m": h_apogee,
    }
    return PerformanceMetrics(
        maximum_altitude_m=float(np.max(trajectory.altitude)) if trajectory.time.size else 0.0,
        maximum_velocity_mps=float(np.max(trajectory.velocity)) if trajectory.time.size else 0.0,
        maximum_acceleration_mps2=float(np.max(trajectory.acceleration)) if trajectory.time.size else 0.0,
        average_acceleration_mps2=float(np.mean(trajectory.acceleration)) if trajectory.time.size else 0.0,
        time_to_apogee_s=t_apogee,
        apogee_altitude_m=h_apogee,
        total_flight_time_s=float(trajectory.time[-1]) if trajectory.time.size else 0.0,
        maximum_downrange_m=float(np.max(trajectory.downrange)) if trajectory.time.size else 0.0,
        downrange_at_apogee_m=x_apogee,
        initial_mass_kg=float(trajectory.mass[0]) if trajectory.time.size else cfg.initial_mass,
        final_mass_kg=float(trajectory.mass[-1]) if trajectory.time.size else cfg.dry_mass,
        propellant_consumed_kg=consumed,
        propellant_remaining_kg=float(trajectory.propellant[-1]) if trajectory.time.size else 0.0,
        commanded_burn_time_s=cfg.burn_time,
        actual_burn_time_s=_last_powered_time(trajectory),
        launch_angle_deg=cfg.launch_angle_deg,
        thrust_n=cfg.thrust,
        drag_coefficient=cfg.drag_coefficient,
        dry_mass_kg=cfg.dry_mass,
        maximum_dynamic_pressure_pa=float(np.max(trajectory.dynamic_pressure)) if trajectory.time.size else 0.0,
        time_of_max_q_s=float(trajectory.time[i_q]) if trajectory.time.size else 0.0,
        time_of_meco_s=_last_powered_time(trajectory),
        time_of_impact_s=float(trajectory.time[-1]) if trajectory.time.size else 0.0,
    )


def validate_trajectory(trajectory: Trajectory) -> list[str]:
    """Return human-readable warnings. Empty list means all checks passed."""
    warnings: list[str] = []
    cfg = trajectory.config
    if trajectory.time.size == 0:
        warnings.append("Trajectory contains no samples.")
        trajectory.warnings = warnings
        return warnings

    if np.any(trajectory.mass < cfg.dry_mass - 1e-6):
        warnings.append("Mass fell below dry mass.")
    if np.any(trajectory.propellant < -1e-6):
        warnings.append("Propellant became negative.")
    if np.any(trajectory.altitude < -1.0):
        warnings.append("Altitude became physically invalid (below ground).")

    speed = trajectory.velocity
    mask = speed > 1e-6
    if np.any(mask):
        # Drag force on the vehicle should have a non-positive projection on velocity.
        power = trajectory.drag_x[mask] * trajectory.vx[mask] + trajectory.drag_y[mask] * trajectory.vy[mask]
        limit = 1e-3 * np.maximum(trajectory.drag[mask] * speed[mask], 1e-9)
        if np.any(power > limit):
            warnings.append("Drag does not consistently oppose the velocity vector.")

    after_burn = trajectory.time > cfg.burn_time + 2.0 * cfg.timestep
    if np.any(after_burn) and np.max(trajectory.thrust[after_burn]) > 1e-3:
        warnings.append("Thrust is non-zero after commanded burnout.")

    dry = trajectory.propellant <= 1e-6
    if np.any(dry) and np.max(trajectory.thrust[dry]) > 1e-3:
        warnings.append("Thrust is non-zero after propellant is exhausted.")

    if not np.all(np.diff(trajectory.mass) <= 1e-6):
        # Mass may be flat after burnout; it must not increase.
        if np.any(np.diff(trajectory.mass) > 1e-4):
            warnings.append("Vehicle mass increased during flight.")

    trajectory.warnings = warnings
    return warnings


def validate_config(config: RocketConfig) -> list[str]:
    warnings: list[str] = []
    if config.dry_mass <= 0.0:
        warnings.append("Dry mass must be positive.")
    if config.propellant_mass < 0.0:
        warnings.append("Propellant mass must be non-negative.")
    if config.thrust < 0.0:
        warnings.append("Thrust must be non-negative.")
    if config.specific_impulse <= 0.0:
        warnings.append("Specific impulse must be positive.")
    if config.burn_time < 0.0:
        warnings.append("Burn time must be non-negative.")
    if config.reference_area <= 0.0:
        warnings.append("Reference area must be positive.")
    if config.initial_mass > config.maximum_initial_mass + 1e-6:
        warnings.append("Initial mass exceeds MAXIMUM_INITIAL_MASS.")
    if config.propellant_mass > config.available_propellant + 1e-6:
        warnings.append("Propellant exceeds AVAILABLE_PROPELLANT.")
    if config.thrust > config.maximum_thrust + 1e-6:
        warnings.append("Thrust exceeds MAXIMUM_THRUST.")
    if config.burn_time > config.maximum_burn_time + 1e-6:
        warnings.append("Burn time exceeds MAXIMUM_BURN_TIME.")
    twr = config.thrust / max(config.initial_mass * config.g0, 1e-9)
    if twr < 1.0:
        warnings.append(
            f"Initial thrust-to-weight ratio is {twr:.2f} (< 1). The vehicle may not lift off."
        )
    return warnings


def constraint_report(config: RocketConfig) -> dict[str, str]:
    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    return {
        "propellant": status(config.propellant_mass <= config.available_propellant + 1e-6),
        "mass": status(config.initial_mass <= config.maximum_initial_mass + 1e-6),
        "thrust": status(
            config.min_thrust - 1e-6 <= config.thrust <= config.effective_thrust_cap() + 1e-6
        ),
        "burn_time": status(
            config.min_burn_time - 1e-6 <= config.burn_time <= config.effective_burn_time_cap() + 1e-6
        ),
        "launch_angle": status(
            config.min_launch_angle_deg - 1e-6
            <= config.launch_angle_deg
            <= config.max_launch_angle_deg + 1e-6
        ),
        "dry_mass": status(
            config.min_dry_mass - 1e-6 <= config.dry_mass <= config.max_dry_mass + 1e-6
        ),
        "drag_coefficient": status(
            config.min_drag_coefficient - 1e-6
            <= config.drag_coefficient
            <= config.max_drag_coefficient + 1e-6
        ),
    }


def _pct_change(baseline: float, optimized: float) -> float:
    if abs(baseline) < 1e-12:
        return float("nan")
    return 100.0 * (optimized - baseline) / baseline


def compare_configurations(
    baseline_cfg: RocketConfig,
    optimized_cfg: RocketConfig,
    baseline_metrics: PerformanceMetrics,
    optimized_metrics: PerformanceMetrics,
) -> pd.DataFrame:
    rows = [
        ("Maximum altitude (m)", baseline_metrics.maximum_altitude_m, optimized_metrics.maximum_altitude_m),
        ("Maximum velocity (m/s)", baseline_metrics.maximum_velocity_mps, optimized_metrics.maximum_velocity_mps),
        ("Maximum acceleration (m/s²)", baseline_metrics.maximum_acceleration_mps2, optimized_metrics.maximum_acceleration_mps2),
        ("Time to apogee (s)", baseline_metrics.time_to_apogee_s, optimized_metrics.time_to_apogee_s),
        ("Flight time (s)", baseline_metrics.total_flight_time_s, optimized_metrics.total_flight_time_s),
        ("Downrange distance (m)", baseline_metrics.maximum_downrange_m, optimized_metrics.maximum_downrange_m),
        ("Initial mass (kg)", baseline_metrics.initial_mass_kg, optimized_metrics.initial_mass_kg),
        ("Propellant consumed (kg)", baseline_metrics.propellant_consumed_kg, optimized_metrics.propellant_consumed_kg),
        ("Burn time (s)", baseline_metrics.actual_burn_time_s, optimized_metrics.actual_burn_time_s),
        ("Launch angle (deg)", baseline_metrics.launch_angle_deg, optimized_metrics.launch_angle_deg),
        ("Thrust (N)", baseline_metrics.thrust_n, optimized_metrics.thrust_n),
        ("Drag coefficient (-)", baseline_metrics.drag_coefficient, optimized_metrics.drag_coefficient),
        ("Dry mass (kg)", baseline_metrics.dry_mass_kg, optimized_metrics.dry_mass_kg),
        ("Maximum dynamic pressure (Pa)", baseline_metrics.maximum_dynamic_pressure_pa, optimized_metrics.maximum_dynamic_pressure_pa),
    ]
    records = []
    for name, base, opt in rows:
        records.append(
            {
                "Parameter": name,
                "Baseline": base,
                "Optimized": opt,
                "Improvement": opt - base,
                "Improvement (%)": _pct_change(base, opt),
            }
        )
    return pd.DataFrame.from_records(records)
