# CHALLENGE 01 – Ascent Trajectory Optimisation

Educational **2-D point-mass** launch-vehicle ascent simulation and design-space optimiser.

This repository is a computational engineering study. It is **not** a model of any operational rocket, flight software, or certified trajectory-design tool. The dynamics are simplified on purpose so the physics, constraints, and optimiser remain transparent.

## Objective

Primary question: *how do you make a rocket reach maximum altitude with limited resources?*

The program:

1. Simulates a user-defined **baseline** vehicle.
2. Runs a **bounded global optimiser** (`scipy.optimize.differential_evolution`, optionally refined with SLSQP).
3. Re-simulates the optimiser’s design with the **same physics model**.
4. Writes comparison tables, CSV time histories, JSON, and engineering plots.

Optional objectives (CLI):

- `--objective max_altitude` (default)
- `--objective max_downrange`
- `--objective min_propellant` (minimum propellant that still reaches `target_altitude`)

## Physics

Planar point-mass equations in the downrange / vertical plane.

Thrust (while burning and propellant remains):

\[
T_x = T\cos\theta,\qquad T_y = T\sin\theta
\]

\(\theta\) is the launch angle from the local horizon (\(90^\circ\) is vertical). This model uses a **constant thrust direction** (no gravity-turn steering law).

Aerodynamic drag opposes the velocity vector:

\[
D = \tfrac{1}{2}\rho v^2 C_D A,\qquad
D_x = D\frac{v_x}{v},\qquad
D_y = D\frac{v_y}{v}
\]

Gravity (optional inverse-square):

\[
g(h) = g_0\left(\frac{R}{R+h}\right)^2
\]

Translational dynamics:

\[
a_x = \frac{T_x - D_x}{m},\qquad
a_y = \frac{T_y - D_y}{m} - g(h)
\]

Mass flow from specific impulse:

\[
\dot m = \frac{T}{I_{sp}\,g_0},\qquad
m(t) = m_{\mathrm{dry}} + m_{\mathrm{prop}}(t)
\]

Burnout occurs at the earlier of commanded `burn_time` and propellant exhaustion. Unused propellant remains as inert mass — wasting tanked propellant is therefore a real penalty, which is the point of the limited-resource constraint.

Atmosphere:

- `standard` — US Standard Atmosphere 1976 layers to 86 km, exponential above
- `exponential` — \(\rho(h)=\rho_0 e^{-h/H}\)

Dynamic pressure (Max-Q):

\[
q = \tfrac{1}{2}\rho v^2
\]

Default integrator: **RK45** (`scipy.integrate.solve_ivp`). Euler is available with `--euler`.

Termination: ground impact (after liftoff), maximum simulation time, or a user altitude ceiling.

## Assumptions and limitations

- Single stage, constant thrust magnitude and constant thrust *direction*.
- Point mass: no rotational dynamics, no aero moments, no winds, no Earth rotation.
- No nozzle altitude compensation (Isp is constant).
- No structural, thermal, or aeroelastic constraints beyond the bounds you set.
- Drag coefficient is a single number, not Mach- or altitude-dependent.
- Results are for coursework / computational demonstration only.

## Changing the vehicle

Edit `BASELINE_CONFIG` in `config.py` (or construct a `RocketConfig(...)` in `main.py`). All important engineering numbers live on that dataclass: masses, thrust, Isp, burn time, \(C_D\), area, launch angle, atmosphere, gravity, resource caps, optimiser bounds, and timestep.

Do not scatter new constants inside `physics.py` / `simulation.py`.

Resource caps the optimiser must honour:

- `available_propellant`
- `maximum_initial_mass`
- `maximum_thrust`
- `maximum_burn_time`

plus min/max bounds on thrust, propellant, burn time, launch angle, \(C_D\), and dry mass.

## Install

Python 3.10+ recommended.

```bash
cd rocket-ascent-optimization
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Other useful invocations:

```bash
python main.py --objective max_downrange
python main.py --objective min_propellant
python main.py --skip-optimize
python main.py --skip-monte-carlo
python main.py --show-plots
python main.py --euler
```

Pipeline:

1. Load baseline configuration  
2. Simulate baseline  
3. Print metrics and validation warnings  
4. Optimise  
5. Simulate the optimised design  
6. Compare  
7. Save CSV / JSON / figures under `results/`

The printed “optimized” numbers are **not** hand-picked. They are the design vector returned by the optimiser, then re-evaluated with the production integrator settings.

## Output files (`results/`)

| File | Contents |
|---|---|
| `baseline_trajectory.csv` | Full baseline time history |
| `optimized_trajectory.csv` | Full optimised time history |
| `optimization_results.csv` | Decision variables and optimiser stats |
| `performance_comparison.csv` | Baseline vs optimised table |
| `monte_carlo_samples.csv` | Optional sensitivity samples |
| `summary.json` | Machine-readable summary |
| `figure01_...png` … `figure10_...png` | Required plots |

## What each graph shows

1. **Altitude vs time** — climb and coast to apogee, then descent. Event markers: liftoff, Max-Q, engine cutoff, apogee, impact.  
2. **Velocity vs time** — inertial speed in the 2-D plane.  
3. **Acceleration vs time** — magnitude \(\sqrt{a_x^2+a_y^2}\).  
4. **Mass vs time** — decreases only while the engine is firing.  
5. **Propellant remaining vs time** — must not go negative; flattens at burnout.  
6. **Thrust and drag vs time** — thrust drops to zero at MECO; drag peaks near Max-Q.  
7. **Altitude vs downrange** — the 2-D trajectory shape.  
8. **Baseline vs optimised altitude** — same physics, two designs.  
9. **Baseline vs optimised velocity** — comparison of speed histories.  
10. **Optimisation visualisation** — convergence of recorded iterates and a coarse altitude heatmap over launch angle and thrust (other parameters held at the optimum).  

Additional figures: density vs altitude, optional 3-D view of the planar path, Monte Carlo altitude histogram.

Apogee is the vertical-velocity zero-crossing (interpolated), which should coincide with maximum altitude.

## Optimization method

Decision variables (each can be frozen in `RocketConfig`):

- thrust  
- propellant mass  
- burn time  
- launch angle  
- drag coefficient  
- dry mass  

Objective for maximum altitude:

\[
J = -h_{\max}
\]

(scipy minimizers minimise \(J\)). Designs that violate resource caps receive a large linear penalty. After differential evolution, a local SLSQP pass refines the point subject to \(m_{\mathrm{dry}}+m_{\mathrm{prop}}\le m_{\mathrm{initial,max}}\).

Because \(C_D\) and dry mass appear linearly in the losses, a well-posed bounded search will typically ride those lower bounds. Thrust, burn time, propellant, and angle remain the interesting trade-off under the propellant and mass caps.

## Interpreting constraint checks

The final banner prints PASS/FAIL for propellant, initial mass, thrust, burn time, and launch angle against the limits in `RocketConfig`. A FAIL means the reported design is outside the challenge resource box and should not be trusted as a feasible answer.
