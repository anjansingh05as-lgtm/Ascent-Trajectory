"""User-editable launch-vehicle configuration and optimization bounds.

Edit BASELINE_CONFIG (and the bound fields) to test different designs.
Engineering values used by the simulation should be changed here, not
inside the physics or integrator modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

ObjectiveName = Literal["max_altitude", "max_downrange", "min_propellant"]
AtmosphereName = Literal["exponential", "standard"]
IntegratorName = Literal["rk45", "euler"]


@dataclass
class RocketConfig:
    """Complete vehicle, environment, simulation, and optimizer settings."""

    # --- Vehicle (baseline design point) ---
    dry_mass: float = 160.0  # kg, structural / unburned mass
    propellant_mass: float = 180.0  # kg loaded at liftoff
    thrust: float = 6200.0  # N, constant while burning
    specific_impulse: float = 245.0  # s, vacuum-equivalent Isp
    burn_time: float = 42.0  # s, commanded burn (ends earlier if propellant is gone)
    drag_coefficient: float = 0.42  # dimensionless
    reference_area: float = 0.18  # m^2
    launch_angle_deg: float = 72.0  # deg from local horizon (90 = vertical)

    # --- Initial state ---
    initial_altitude: float = 0.0  # m
    initial_downrange: float = 0.0  # m
    initial_velocity: float = 0.0  # m/s along the launch-angle direction

    # --- Environment ---
    atmosphere_model: AtmosphereName = "standard"
    sea_level_density: float = 1.225  # kg/m^3
    scale_height: float = 8500.0  # m, exponential atmosphere
    sea_level_gravity: float = 9.80665  # m/s^2, also g0 for Isp
    earth_radius: float = 6_371_000.0  # m
    vary_gravity_with_altitude: bool = True

    # --- Limited resources / engineering constraints ---
    available_propellant: float = 260.0  # kg, hard resource cap
    maximum_initial_mass: float = 450.0  # kg
    maximum_thrust: float = 14_000.0  # N
    maximum_burn_time: float = 90.0  # s
    maximum_allowable_vehicle_mass: float = 450.0  # kg (same role as initial mass cap)
    target_altitude: float = 40_000.0  # m, used by min-propellant objective
    user_maximum_altitude: float = 2_000_000.0  # m, safety stop (not an apogee target)

    # Optimizer search bounds
    min_thrust: float = 3500.0
    max_thrust: float = 14_000.0
    min_propellant_mass: float = 80.0
    max_propellant_mass: float = 260.0
    min_burn_time: float = 15.0
    max_burn_time: float = 90.0
    min_launch_angle_deg: float = 55.0
    max_launch_angle_deg: float = 90.0
    min_drag_coefficient: float = 0.22
    max_drag_coefficient: float = 0.55
    min_dry_mass: float = 130.0
    max_dry_mass: float = 220.0

    # Which decision variables the optimizer may change
    optimize_thrust: bool = True
    optimize_propellant: bool = True
    optimize_burn_time: bool = True
    optimize_launch_angle: bool = True
    optimize_drag_coefficient: bool = True
    optimize_dry_mass: bool = True

    # --- Simulation numerics ---
    timestep: float = 0.05  # s, output sampling / Euler step
    max_simulation_time: float = 800.0  # s
    integrator: IntegratorName = "rk45"
    rk45_rtol: float = 1e-6
    rk45_atol: float = 1e-8
    # Looser tolerances used only inside the optimizer inner loop
    optimizer_rtol: float = 1e-4
    optimizer_atol: float = 1e-6
    optimizer_timestep: float = 0.10

    # --- Optimization ---
    objective: ObjectiveName = "max_altitude"
    de_maxiter: int = 18
    de_popsize: int = 9
    de_seed: int = 42
    refine_locally: bool = True
    enable_monte_carlo: bool = True
    monte_carlo_samples: int = 40
    monte_carlo_relative_std: float = 0.05

    # --- Output ---
    results_dir: str = "results"
    save_json: bool = True
    show_plots: bool = False

    notes: str = field(
        default=(
            "Educational 2-D point-mass ascent model. Not a representation of "
            "any operational launch vehicle."
        )
    )

    @property
    def initial_mass(self) -> float:
        return self.dry_mass + self.propellant_mass

    @property
    def g0(self) -> float:
        return self.sea_level_gravity

    @property
    def mass_flow_rate(self) -> float:
        """Propellant mass flow implied by thrust and specific impulse."""
        isp_g0 = self.specific_impulse * self.g0
        if isp_g0 <= 0.0:
            return 0.0
        return self.thrust / isp_g0

    def copy(self, **changes) -> "RocketConfig":
        return replace(self, **changes)

    def effective_propellant_cap(self) -> float:
        return min(self.available_propellant, self.max_propellant_mass)

    def effective_thrust_cap(self) -> float:
        return min(self.maximum_thrust, self.max_thrust)

    def effective_burn_time_cap(self) -> float:
        return min(self.maximum_burn_time, self.max_burn_time)


# Default challenge baseline. Change this object (or construct a new
# RocketConfig in main.py) to try another vehicle.
BASELINE_CONFIG = RocketConfig()
