"""CHALLENGE 01 – Ascent trajectory optimisation.

Educational 2-D point-mass launch-vehicle simulation. This program is not a
model of any operational rocket.

Usage:
    python main.py
    python main.py --objective max_downrange
    python main.py --objective min_propellant
    python main.py --skip-optimize
    python main.py --show-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

from analysis import (
    calculate_performance_metrics,
    compare_configurations,
    constraint_report,
    validate_config,
    validate_trajectory,
)
from config import BASELINE_CONFIG, RocketConfig
from optimizer import monte_carlo_sensitivity, optimize_rocket
from simulation import simulate_trajectory
from visualization import plot_results


def _print_warnings(title: str, warnings: list[str]) -> None:
    if not warnings:
        print(f"{title}: none")
        return
    print(title)
    for item in warnings:
        print(f"  WARNING: {item}")


def _print_metrics(title: str, metrics) -> None:
    print(f"\n{title}")
    print(f"  Maximum altitude:           {metrics.maximum_altitude_m/1000.0:10.3f} km")
    print(f"  Maximum velocity:           {metrics.maximum_velocity_mps:10.2f} m/s")
    print(f"  Maximum acceleration:       {metrics.maximum_acceleration_mps2:10.2f} m/s²")
    print(f"  Average acceleration:       {metrics.average_acceleration_mps2:10.2f} m/s²")
    print(f"  Time to apogee:             {metrics.time_to_apogee_s:10.2f} s")
    print(f"  Total flight time:          {metrics.total_flight_time_s:10.2f} s")
    print(f"  Maximum downrange:          {metrics.maximum_downrange_m/1000.0:10.3f} km")
    print(f"  Initial mass:               {metrics.initial_mass_kg:10.2f} kg")
    print(f"  Final mass:                 {metrics.final_mass_kg:10.2f} kg")
    print(f"  Propellant consumed:        {metrics.propellant_consumed_kg:10.2f} kg")
    print(f"  Propellant remaining:       {metrics.propellant_remaining_kg:10.2f} kg")
    print(f"  Commanded burn time:        {metrics.commanded_burn_time_s:10.2f} s")
    print(f"  Actual engine cutoff:       {metrics.actual_burn_time_s:10.2f} s")
    print(f"  Launch angle:               {metrics.launch_angle_deg:10.2f} deg")
    print(f"  Thrust:                     {metrics.thrust_n:10.1f} N")
    print(f"  Maximum dynamic pressure:   {metrics.maximum_dynamic_pressure_pa/1000.0:10.2f} kPa")


def _pct(a: float, b: float) -> float:
    if abs(a) < 1e-12:
        return float("nan")
    return 100.0 * (b - a) / a


def print_final_banner(baseline_m, optimized_m, optimized_cfg, checks: dict[str, str]) -> None:
    alt_imp = _pct(baseline_m.maximum_altitude_m, optimized_m.maximum_altitude_m)
    vel_imp = _pct(baseline_m.maximum_velocity_mps, optimized_m.maximum_velocity_mps)
    prop_save = _pct(baseline_m.propellant_consumed_kg, optimized_m.propellant_consumed_kg)
    print("\n========================================")
    print("ASCENT TRAJECTORY OPTIMIZATION")
    print("========================================")
    print("\nBASELINE CONFIGURATION")
    print(f"Maximum Altitude: {baseline_m.maximum_altitude_m:.2f} m")
    print(f"Maximum Velocity: {baseline_m.maximum_velocity_mps:.2f} m/s")
    print(f"Maximum Acceleration: {baseline_m.maximum_acceleration_mps2:.2f} m/s^2")
    print(f"Time to Apogee: {baseline_m.time_to_apogee_s:.2f} s")
    print(f"Downrange Distance: {baseline_m.maximum_downrange_m:.2f} m")
    print(f"Propellant Consumed: {baseline_m.propellant_consumed_kg:.2f} kg")
    print("\nOPTIMIZED CONFIGURATION")
    print(f"Maximum Altitude: {optimized_m.maximum_altitude_m:.2f} m")
    print(f"Maximum Velocity: {optimized_m.maximum_velocity_mps:.2f} m/s")
    print(f"Maximum Acceleration: {optimized_m.maximum_acceleration_mps2:.2f} m/s^2")
    print(f"Time to Apogee: {optimized_m.time_to_apogee_s:.2f} s")
    print(f"Downrange Distance: {optimized_m.maximum_downrange_m:.2f} m")
    print(f"Propellant Consumed: {optimized_m.propellant_consumed_kg:.2f} kg")
    print("\nIMPROVEMENT")
    print(f"Altitude Improvement: {alt_imp:.2f} %")
    print(f"Velocity Change: {vel_imp:.2f} %")
    print(f"Propellant Saving: {-prop_save:.2f} %")
    print("\nOPTIMIZED DESIGN PARAMETERS")
    print(f"Thrust: {optimized_cfg.thrust:.1f} N")
    print(f"Propellant Mass: {optimized_cfg.propellant_mass:.2f} kg")
    print(f"Burn Time: {optimized_cfg.burn_time:.2f} s")
    print(f"Launch Angle: {optimized_cfg.launch_angle_deg:.2f} deg")
    print(f"Vehicle Mass: {optimized_cfg.initial_mass:.2f} kg")
    print(f"Drag Coefficient: {optimized_cfg.drag_coefficient:.4f}")
    print("\nCONSTRAINT CHECK")
    print(f"Propellant Constraint: {checks['propellant']}")
    print(f"Mass Constraint: {checks['mass']}")
    print(f"Thrust Constraint: {checks['thrust']}")
    print(f"Burn-Time Constraint: {checks['burn_time']}")
    print(f"Launch-Angle Constraint: {checks['launch_angle']}")
    print("========================================")
    print(
        "\nModel disclaimer: simplified educational 2-D point-mass dynamics. "
        "Not an operational launch-vehicle simulation."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ascent trajectory optimisation challenge")
    parser.add_argument(
        "--objective",
        choices=["max_altitude", "max_downrange", "min_propellant"],
        default=None,
        help="Override RocketConfig.objective",
    )
    parser.add_argument("--skip-optimize", action="store_true", help="Run the baseline only")
    parser.add_argument("--skip-monte-carlo", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--euler", action="store_true", help="Use Euler integration instead of RK45")
    return parser.parse_args(argv)


def run(config: RocketConfig | None = None, argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.show_plots:
        matplotlib.use("Agg")

    # Users should edit BASELINE_CONFIG in config.py. Optional CLI overrides
    # are applied on a copy so the module-level object stays the template.
    baseline_cfg = (config or BASELINE_CONFIG).copy()
    if args.objective:
        baseline_cfg = baseline_cfg.copy(objective=args.objective)
    if args.euler:
        baseline_cfg = baseline_cfg.copy(integrator="euler")
    if args.show_plots:
        baseline_cfg = baseline_cfg.copy(show_plots=True)

    results_dir = Path(baseline_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("CHALLENGE 01 – ASCENT TRAJECTORY OPTIMISATION")
    print("========================================")
    print(baseline_cfg.notes)
    print(f"Objective: {baseline_cfg.objective}")
    print(f"Integrator: {baseline_cfg.integrator}")
    print(
        f"Baseline: m0={baseline_cfg.initial_mass:.1f} kg, T={baseline_cfg.thrust:.0f} N, "
        f"mp={baseline_cfg.propellant_mass:.1f} kg, tb={baseline_cfg.burn_time:.1f} s, "
        f"theta={baseline_cfg.launch_angle_deg:.1f} deg"
    )

    _print_warnings("Baseline configuration checks", validate_config(baseline_cfg))

    print("\n1) Simulating baseline configuration...")
    baseline_traj = simulate_trajectory(baseline_cfg, fast=False)
    baseline_metrics = calculate_performance_metrics(baseline_traj)
    _print_warnings("Baseline trajectory checks", validate_trajectory(baseline_traj))
    _print_metrics("BASELINE RESULTS", baseline_metrics)
    baseline_traj.to_dataframe().to_csv(results_dir / "baseline_trajectory.csv", index=False)

    if args.skip_optimize:
        print("\nOptimization skipped.")
        return 0

    print("\n2) Running optimizer (differential evolution)...")
    opt_result = optimize_rocket(baseline_cfg, verbose=True)
    optimized_cfg = opt_result.config

    print("\nOptimized parameters:")
    print(f"  Thrust:            {optimized_cfg.thrust:.2f} N")
    print(f"  Propellant mass:   {optimized_cfg.propellant_mass:.2f} kg")
    print(f"  Burn time:         {optimized_cfg.burn_time:.2f} s")
    print(f"  Launch angle:      {optimized_cfg.launch_angle_deg:.2f} deg")
    print(f"  Drag coefficient:  {optimized_cfg.drag_coefficient:.4f}")
    print(f"  Dry mass:          {optimized_cfg.dry_mass:.2f} kg")
    print(f"  Initial mass:      {optimized_cfg.initial_mass:.2f} kg")
    print(f"  Mass flow rate:    {optimized_cfg.mass_flow_rate:.4f} kg/s")
    print(f"  Optimizer message: {opt_result.message}")

    print("\n3) Simulating optimized configuration with the same physics model...")
    optimized_traj = simulate_trajectory(optimized_cfg, fast=False)
    optimized_metrics = calculate_performance_metrics(optimized_traj)
    _print_warnings("Optimized configuration checks", validate_config(optimized_cfg))
    _print_warnings("Optimized trajectory checks", validate_trajectory(optimized_traj))
    _print_metrics("OPTIMIZED RESULTS", optimized_metrics)
    optimized_traj.to_dataframe().to_csv(results_dir / "optimized_trajectory.csv", index=False)

    comparison = compare_configurations(
        baseline_cfg, optimized_cfg, baseline_metrics, optimized_metrics
    )
    comparison.to_csv(results_dir / "performance_comparison.csv", index=False)
    print("\n4) Baseline vs optimized comparison")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    opt_rows = {
        "parameter": [
            "thrust_n",
            "propellant_mass_kg",
            "burn_time_s",
            "launch_angle_deg",
            "drag_coefficient",
            "dry_mass_kg",
            "initial_mass_kg",
            "objective_value",
            "nfev",
            "success",
        ],
        "value": [
            optimized_cfg.thrust,
            optimized_cfg.propellant_mass,
            optimized_cfg.burn_time,
            optimized_cfg.launch_angle_deg,
            optimized_cfg.drag_coefficient,
            optimized_cfg.dry_mass,
            optimized_cfg.initial_mass,
            opt_result.objective_value,
            opt_result.nfev,
            opt_result.success,
        ],
    }
    import pandas as pd

    pd.DataFrame(opt_rows).to_csv(results_dir / "optimization_results.csv", index=False)

    sensitivity = None
    if baseline_cfg.enable_monte_carlo and not args.skip_monte_carlo:
        print("\n5) Monte Carlo sensitivity around the optimized design...")
        sensitivity = monte_carlo_sensitivity(optimized_cfg)
        pd.DataFrame(sensitivity).to_csv(results_dir / "monte_carlo_samples.csv", index=False)

    print("\n6) Generating plots...")
    saved = plot_results(
        baseline_traj,
        optimized_traj,
        baseline_metrics,
        optimized_metrics,
        opt_result,
        results_dir,
        sensitivity=sensitivity,
        show=baseline_cfg.show_plots,
    )
    for path in saved:
        print(f"  saved {path}")

    payload = {
        "disclaimer": baseline_cfg.notes,
        "objective": baseline_cfg.objective,
        "baseline": {
            "config": {
                "dry_mass": baseline_cfg.dry_mass,
                "propellant_mass": baseline_cfg.propellant_mass,
                "thrust": baseline_cfg.thrust,
                "specific_impulse": baseline_cfg.specific_impulse,
                "burn_time": baseline_cfg.burn_time,
                "drag_coefficient": baseline_cfg.drag_coefficient,
                "reference_area": baseline_cfg.reference_area,
                "launch_angle_deg": baseline_cfg.launch_angle_deg,
            },
            "metrics": baseline_metrics.as_dict(),
        },
        "optimized": {
            "config": {
                "dry_mass": optimized_cfg.dry_mass,
                "propellant_mass": optimized_cfg.propellant_mass,
                "thrust": optimized_cfg.thrust,
                "specific_impulse": optimized_cfg.specific_impulse,
                "burn_time": optimized_cfg.burn_time,
                "drag_coefficient": optimized_cfg.drag_coefficient,
                "reference_area": optimized_cfg.reference_area,
                "launch_angle_deg": optimized_cfg.launch_angle_deg,
            },
            "metrics": optimized_metrics.as_dict(),
            "nfev": opt_result.nfev,
            "message": opt_result.message,
        },
        "constraints": constraint_report(optimized_cfg),
    }
    if baseline_cfg.save_json:
        (results_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_final_banner(
        baseline_metrics,
        optimized_metrics,
        optimized_cfg,
        constraint_report(optimized_cfg),
    )
    print(f"\nNumerical results written to: {results_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
