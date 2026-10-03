"""Bounded global optimization of the ascent design."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution, minimize

from analysis import calculate_performance_metrics
from config import RocketConfig
from simulation import simulate_peaks, simulate_trajectory

# Decision-variable order used internally.
_VAR_NAMES = (
    "thrust",
    "propellant_mass",
    "burn_time",
    "launch_angle_deg",
    "drag_coefficient",
    "dry_mass",
)


@dataclass
class OptimizationResult:
    config: RocketConfig
    objective_value: float
    nfev: int
    success: bool
    message: str
    history_best_objective: list[float] = field(default_factory=list)
    history_best_altitude: list[float] = field(default_factory=list)
    heatmap_angles: np.ndarray | None = None
    heatmap_thrusts: np.ndarray | None = None
    heatmap_altitude: np.ndarray | None = None


def _active_variables(config: RocketConfig) -> list[str]:
    flags = {
        "thrust": config.optimize_thrust,
        "propellant_mass": config.optimize_propellant,
        "burn_time": config.optimize_burn_time,
        "launch_angle_deg": config.optimize_launch_angle,
        "drag_coefficient": config.optimize_drag_coefficient,
        "dry_mass": config.optimize_dry_mass,
    }
    return [name for name in _VAR_NAMES if flags[name]]


def _bounds(config: RocketConfig) -> dict[str, tuple[float, float]]:
    return {
        "thrust": (config.min_thrust, config.effective_thrust_cap()),
        "propellant_mass": (config.min_propellant_mass, config.effective_propellant_cap()),
        "burn_time": (config.min_burn_time, config.effective_burn_time_cap()),
        "launch_angle_deg": (config.min_launch_angle_deg, config.max_launch_angle_deg),
        "drag_coefficient": (config.min_drag_coefficient, config.max_drag_coefficient),
        "dry_mass": (config.min_dry_mass, config.max_dry_mass),
    }


def _vector_to_config(vector: np.ndarray, template: RocketConfig, names: list[str]) -> RocketConfig:
    updates = {name: float(value) for name, value in zip(names, vector)}
    return template.copy(**updates)


def _penalty(config: RocketConfig) -> float:
    penalty = 0.0
    if config.propellant_mass > config.available_propellant:
        penalty += 1e5 * (config.propellant_mass - config.available_propellant)
    if config.initial_mass > config.maximum_initial_mass:
        penalty += 1e5 * (config.initial_mass - config.maximum_initial_mass)
    if config.thrust > config.effective_thrust_cap():
        penalty += 1e5 * (config.thrust - config.effective_thrust_cap())
    if config.burn_time > config.effective_burn_time_cap():
        penalty += 1e5 * (config.burn_time - config.effective_burn_time_cap())
    if config.initial_mass > config.maximum_allowable_vehicle_mass:
        penalty += 1e5 * (config.initial_mass - config.maximum_allowable_vehicle_mass)
    if config.dry_mass + 1e-9 < 0.0:
        penalty += 1e6
    return penalty


def evaluate_objective(config: RocketConfig, *, fast: bool = True) -> tuple[float, float, float]:
    """Return (objective, max_altitude, max_downrange)."""
    extra = _penalty(config)
    try:
        if fast:
            altitude, downrange = simulate_peaks(config)
        else:
            traj = simulate_trajectory(config, fast=False)
            metrics = calculate_performance_metrics(traj)
            altitude = metrics.maximum_altitude_m
            downrange = metrics.maximum_downrange_m
    except Exception:
        return 1e9 + extra, 0.0, 0.0

    if config.objective == "max_altitude":
        obj = -altitude
    elif config.objective == "max_downrange":
        obj = -downrange
    elif config.objective == "min_propellant":
        miss = max(config.target_altitude - altitude, 0.0)
        obj = config.propellant_mass + 50.0 * miss + 1e-3 * miss * miss
    else:
        obj = -altitude
    return float(obj + extra), float(altitude), float(downrange)


def optimize_rocket(baseline: RocketConfig, *, verbose: bool = True) -> OptimizationResult:
    names = _active_variables(baseline)
    if not names:
        raise ValueError("No optimization variables are enabled in RocketConfig.")

    bounds_map = _bounds(baseline)
    bounds = [bounds_map[name] for name in names]
    history_obj: list[float] = []
    history_alt: list[float] = []
    best_obj = [np.inf]

    def objective(vector: np.ndarray) -> float:
        cfg = _vector_to_config(vector, baseline, names)
        obj, alt, _ = evaluate_objective(cfg, fast=True)
        if obj < best_obj[0]:
            best_obj[0] = obj
        return obj

    def callback(*args, **kwargs):
        first = args[0] if args else kwargs.get("xk")
        xk = first.x if hasattr(first, "x") else first
        cfg = _vector_to_config(np.asarray(xk, dtype=float), baseline, names)
        obj, alt, _ = evaluate_objective(cfg, fast=True)
        history_obj.append(obj)
        history_alt.append(alt)
        if verbose:
            print(
                f"  DE iterate: altitude = {alt/1000.0:.3f} km, "
                f"objective = {obj:.4f}"
            )

    if verbose:
        print("Running differential evolution (this may take a minute)...")

    de_result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=baseline.de_maxiter,
        popsize=baseline.de_popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=baseline.de_seed,
        polish=False,
        updating="deferred",
        workers=1,
        callback=callback,
        atol=0.0,
        tol=1e-3,
    )

    x_best = np.asarray(de_result.x, dtype=float)
    nfev = int(de_result.nfev)

    if baseline.refine_locally:
        if verbose:
            print("Refining with a local SLSQP search...")

        def local_objective(vector):
            return objective(vector)

        cons = []

        def mass_constraint(vector):
            cfg = _vector_to_config(vector, baseline, names)
            return baseline.maximum_initial_mass - cfg.initial_mass

        cons.append({"type": "ineq", "fun": mass_constraint})
        local = minimize(
            local_objective,
            x_best,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 40, "ftol": 1e-6, "disp": False},
        )
        nfev += int(getattr(local, "nfev", 0) or 0)
        if local.success or (local.fun < de_result.fun):
            x_best = np.asarray(local.x, dtype=float)
            message = f"DE: {de_result.message}; local: {local.message}"
            success = bool(local.success or de_result.success)
        else:
            message = str(de_result.message)
            success = bool(de_result.success)
    else:
        message = str(de_result.message)
        success = bool(de_result.success)

    optimized = _vector_to_config(x_best, baseline, names)
    # Clip to resource caps in case of numerical sliver overshoot.
    optimized = optimized.copy(
        propellant_mass=min(optimized.propellant_mass, optimized.effective_propellant_cap()),
        thrust=min(optimized.thrust, optimized.effective_thrust_cap()),
        burn_time=min(optimized.burn_time, optimized.effective_burn_time_cap()),
    )
    if optimized.initial_mass > optimized.maximum_initial_mass:
        overflow = optimized.initial_mass - optimized.maximum_initial_mass
        optimized = optimized.copy(
            propellant_mass=max(optimized.min_propellant_mass, optimized.propellant_mass - overflow)
        )

    final_obj, final_alt, _ = evaluate_objective(optimized, fast=False)
    history_obj.append(final_obj)
    history_alt.append(final_alt)

    if verbose:
        print(f"Optimizer finished after {nfev} evaluations. Best altitude = {final_alt/1000.0:.3f} km")

    heat_angles, heat_thrusts, heat_alt = _design_space_slice(optimized, baseline)
    return OptimizationResult(
        config=optimized,
        objective_value=final_obj,
        nfev=nfev,
        success=success,
        message=message,
        history_best_objective=history_obj,
        history_best_altitude=history_alt,
        heatmap_angles=heat_angles,
        heatmap_thrusts=heat_thrusts,
        heatmap_altitude=heat_alt,
    )


def _design_space_slice(
    optimized: RocketConfig,
    baseline: RocketConfig,
    n_angle: int = 7,
    n_thrust: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coarse altitude map vs launch angle and thrust, other params held at optimum."""
    angles = np.linspace(baseline.min_launch_angle_deg, baseline.max_launch_angle_deg, n_angle)
    thrusts = np.linspace(baseline.min_thrust, baseline.effective_thrust_cap(), n_thrust)
    altitude = np.zeros((n_thrust, n_angle))
    for i, thrust in enumerate(thrusts):
        for j, angle in enumerate(angles):
            cfg = optimized.copy(thrust=float(thrust), launch_angle_deg=float(angle))
            _, alt, _ = evaluate_objective(cfg, fast=True)
            altitude[i, j] = alt
    return angles, thrusts, altitude


def monte_carlo_sensitivity(
    config: RocketConfig,
    samples: int | None = None,
    relative_std: float | None = None,
) -> dict[str, np.ndarray]:
    n = samples if samples is not None else config.monte_carlo_samples
    std = relative_std if relative_std is not None else config.monte_carlo_relative_std
    rng = np.random.default_rng(config.de_seed + 7)
    altitudes = np.zeros(n)
    keys = ("thrust", "propellant_mass", "burn_time", "launch_angle_deg", "drag_coefficient", "dry_mass")
    draws = {k: np.zeros(n) for k in keys}
    for i in range(n):
        updates = {}
        for key in keys:
            nominal = getattr(config, key)
            value = float(rng.normal(nominal, abs(nominal) * std))
            draws[key][i] = value
            updates[key] = value
        perturbed = config.copy(**updates)
        _, alt, _ = evaluate_objective(perturbed, fast=True)
        altitudes[i] = alt
    draws["maximum_altitude_m"] = altitudes
    return draws
