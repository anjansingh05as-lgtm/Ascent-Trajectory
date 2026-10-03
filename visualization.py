"""Engineering plots for baseline, optimized, and design-space results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from analysis import PerformanceMetrics, find_apogee
from optimizer import OptimizationResult
from simulation import Trajectory

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
    }
)


def _event_lines(ax, trajectory: Trajectory, x_is_time: bool = True) -> None:
    events = trajectory.events or {}
    mapping = [
        ("liftoff_s", "Liftoff", "#2ca02c"),
        ("max_q_s", "Max-Q", "#ff7f0e"),
        ("meco_s", "Engine cutoff", "#d62728"),
        ("apogee_s", "Apogee", "#1f77b4"),
        ("impact_s", "Impact", "#7f7f7f"),
    ]
    ymax = ax.get_ylim()[1]
    for key, label, color in mapping:
        t = events.get(key)
        if t is None:
            continue
        x = t if x_is_time else np.interp(t, trajectory.time, trajectory.downrange)
        ax.axvline(x, color=color, linestyle="--", linewidth=0.9, alpha=0.85, label=label)
    ax.set_ylim(top=ymax)


def _save(fig: Figure, results_dir: Path, name: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / name
    fig.tight_layout()
    fig.savefig(path)
    return path


def plot_results(
    baseline: Trajectory,
    optimized: Trajectory,
    baseline_metrics: PerformanceMetrics,
    optimized_metrics: PerformanceMetrics,
    opt_result: OptimizationResult | None,
    results_dir: str | Path,
    sensitivity: dict[str, np.ndarray] | None = None,
    show: bool = False,
) -> list[Path]:
    results_dir = Path(results_dir)
    saved: list[Path] = []

    def time_series(traj: Trajectory, y, ylabel, title, filename, color):
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.plot(traj.time, y, color=color, label=title.split(":")[0])
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        _event_lines(ax, traj, x_is_time=True)
        ax.legend(loc="best")
        saved.append(_save(fig, results_dir, filename))
        plt.close(fig)

    time_series(
        baseline,
        baseline.altitude / 1000.0,
        "Altitude (km)",
        "Figure 1: Altitude vs Time (baseline)",
        "figure01_altitude_vs_time.png",
        "#1f77b4",
    )
    time_series(
        baseline,
        baseline.velocity,
        "Velocity (m/s)",
        "Figure 2: Velocity vs Time (baseline)",
        "figure02_velocity_vs_time.png",
        "#ff7f0e",
    )
    time_series(
        baseline,
        baseline.acceleration,
        "Acceleration magnitude (m/s²)",
        "Figure 3: Acceleration vs Time (baseline)",
        "figure03_acceleration_vs_time.png",
        "#d62728",
    )
    time_series(
        baseline,
        baseline.mass,
        "Mass (kg)",
        "Figure 4: Mass vs Time (baseline)",
        "figure04_mass_vs_time.png",
        "#9467bd",
    )
    time_series(
        baseline,
        baseline.propellant,
        "Propellant remaining (kg)",
        "Figure 5: Propellant Remaining vs Time (baseline)",
        "figure05_propellant_vs_time.png",
        "#8c564b",
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(baseline.time, baseline.thrust, label="Thrust (N)", color="#1f77b4")
    ax.plot(baseline.time, baseline.drag, label="Drag (N)", color="#d62728")
    ax.set_title("Figure 6: Thrust and Drag vs Time (baseline)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    _event_lines(ax, baseline)
    ax.legend(loc="best")
    saved.append(_save(fig, results_dir, "figure06_thrust_drag_vs_time.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(baseline.downrange / 1000.0, baseline.altitude / 1000.0, label="Baseline", color="#1f77b4")
    ax.plot(
        optimized.downrange / 1000.0,
        optimized.altitude / 1000.0,
        label="Optimized",
        color="#d62728",
    )
    t_a, h_a, x_a = find_apogee(baseline)
    t_o, h_o, x_o = find_apogee(optimized)
    ax.scatter([x_a / 1000.0], [h_a / 1000.0], marker="o", color="#1f77b4", zorder=5, label="Baseline apogee")
    ax.scatter([x_o / 1000.0], [h_o / 1000.0], marker="s", color="#d62728", zorder=5, label="Optimized apogee")
    ax.set_title("Figure 7: Altitude vs Downrange Distance")
    ax.set_xlabel("Downrange distance (km)")
    ax.set_ylabel("Altitude (km)")
    ax.set_aspect("auto")
    ax.legend(loc="best")
    saved.append(_save(fig, results_dir, "figure07_trajectory_2d.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(baseline.time, baseline.altitude / 1000.0, label="Baseline", color="#1f77b4")
    ax.plot(optimized.time, optimized.altitude / 1000.0, label="Optimized", color="#d62728")
    ax.axhline(
        baseline_metrics.maximum_altitude_m / 1000.0,
        color="#1f77b4",
        linestyle=":",
        linewidth=0.8,
        label="Baseline apogee altitude",
    )
    ax.axhline(
        optimized_metrics.maximum_altitude_m / 1000.0,
        color="#d62728",
        linestyle=":",
        linewidth=0.8,
        label="Optimized apogee altitude",
    )
    ax.set_title("Figure 8: Baseline vs Optimized Altitude vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.legend(loc="best")
    saved.append(_save(fig, results_dir, "figure08_compare_altitude.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(baseline.time, baseline.velocity, label="Baseline", color="#1f77b4")
    ax.plot(optimized.time, optimized.velocity, label="Optimized", color="#d62728")
    ax.set_title("Figure 9: Baseline vs Optimized Velocity vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="best")
    saved.append(_save(fig, results_dir, "figure09_compare_velocity.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(11.0, 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    if opt_result and opt_result.history_best_altitude:
        gens = np.arange(1, len(opt_result.history_best_altitude) + 1)
        ax1.plot(gens, np.asarray(opt_result.history_best_altitude) / 1000.0, marker="o", ms=3)
        ax1.set_title("Optimization convergence")
        ax1.set_xlabel("Recorded iterate")
        ax1.set_ylabel("Best altitude (km)")
        ax1.grid(True, alpha=0.35)
    else:
        ax1.text(0.5, 0.5, "No optimizer history", ha="center", va="center")
        ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 2, 2)
    if opt_result is not None and opt_result.heatmap_altitude is not None:
        ang = opt_result.heatmap_angles
        thr = opt_result.heatmap_thrusts
        z = opt_result.heatmap_altitude / 1000.0
        mesh = ax2.pcolormesh(ang, thr, z, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax2, label="Max altitude (km)")
        ax2.scatter(
            [optimized.config.launch_angle_deg],
            [optimized.config.thrust],
            color="red",
            marker="x",
            s=60,
            label="Optimized point",
        )
        ax2.set_title("Design space: altitude vs angle & thrust")
        ax2.set_xlabel("Launch angle (deg)")
        ax2.set_ylabel("Thrust (N)")
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "Design-space map unavailable", ha="center", va="center")
        ax2.set_axis_off()
    fig.suptitle("Figure 10: Optimization / design-space visualization")
    saved.append(_save(fig, results_dir, "figure10_optimization_design_space.png"))
    plt.close(fig)

    # Extra supporting plots (density, Monte Carlo, 3-D trajectory)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    order = np.argsort(baseline.altitude)
    ax.plot(
        baseline.altitude[order] / 1000.0,
        baseline.density[order],
        label="Baseline path",
        color="#1f77b4",
    )
    order_o = np.argsort(optimized.altitude)
    ax.plot(
        optimized.altitude[order_o] / 1000.0,
        optimized.density[order_o],
        label="Optimized path",
        color="#d62728",
    )
    ax.set_title("Atmospheric density vs altitude")
    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("Density (kg/m³)")
    ax.set_yscale("log")
    ax.legend(loc="best")
    saved.append(_save(fig, results_dir, "figure_density_vs_altitude.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(8.0, 6.0))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot(baseline.downrange / 1000.0, np.zeros_like(baseline.downrange), baseline.altitude / 1000.0, label="Baseline")
    ax3.plot(
        optimized.downrange / 1000.0,
        np.zeros_like(optimized.downrange),
        optimized.altitude / 1000.0,
        label="Optimized",
    )
    ax3.set_xlabel("Downrange (km)")
    ax3.set_ylabel("Crossrange (km)")
    ax3.set_zlabel("Altitude (km)")
    ax3.set_title("Optional 3-D view of the planar trajectory")
    ax3.legend(loc="best")
    saved.append(_save(fig, results_dir, "figure_optional_3d_trajectory.png"))
    plt.close(fig)

    if sensitivity and "maximum_altitude_m" in sensitivity:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.hist(sensitivity["maximum_altitude_m"] / 1000.0, bins=12, color="#1f77b4", edgecolor="white")
        ax.axvline(
            optimized_metrics.maximum_altitude_m / 1000.0,
            color="#d62728",
            linestyle="--",
            label="Nominal optimized altitude",
        )
        ax.set_title("Monte Carlo altitude sensitivity")
        ax.set_xlabel("Maximum altitude (km)")
        ax.set_ylabel("Count")
        ax.legend(loc="best")
        saved.append(_save(fig, results_dir, "figure_monte_carlo_altitude.png"))
        plt.close(fig)

        # One-at-a-time tornado around the optimized point is plotted in main via csv;
        # keep the histogram as the primary sensitivity graphic.

    if show:
        plt.show()
    return saved
