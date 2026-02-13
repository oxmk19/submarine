# Submarine Search RL (PyBullet + Custom Water Physics)

This project implements a Gymnasium-compatible underwater search environment where a submarine rigid body must find a hidden target. PyBullet handles rigid-body dynamics and collision, while custom code applies water effects (buoyancy, drag, damping, current, optional restoring torque) each simulation step.

## Project Structure

```text
.
├── assets/
│   ├── submarine.urdf
│   └── submarine_fallback.urdf
├── envs/
│   ├── __init__.py
│   └── submarine_search_env.py
├── physics/
│   ├── __init__.py
│   ├── thrusters.py
│   └── water_forces.py
├── scripts/
│   ├── evaluate_policy.py
│   ├── qt_motor_controller.py
│   ├── run_manual_control.py
│   ├── run_random_policy.py
│   └── train_sb3.py
├── requirements.txt
└── README.md
```

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start Commands

1. Random-policy stability check (headless):
```bash
python scripts/run_random_policy.py
```

2. Manual keyboard control (GUI):
```bash
python scripts/run_manual_control.py
```

3. Qt motor-by-motor controller (GUI):
```bash
python scripts/qt_motor_controller.py
```

4. Train SAC (default):
```bash
python scripts/train_sb3.py --algo sac --total-timesteps 300000
```

5. Evaluate trained policy:
```bash
python scripts/evaluate_policy.py --algo sac --model-path models/sac_submarine_search.zip --episodes 20
```

## Environment Summary

- World bounds: `x,y in [-20, 20]`, `z in [-20, -1]`.
- Episode ends on success (`distance < 1.5 m`) or max steps (`2000`).
- Observation (17 floats):
  - position `(3)`
  - linear velocity `(3)`
  - angular velocity `(3)`
  - orientation quaternion `(4)`
  - noisy sonar-like unit direction to target `(3)`
  - noisy sonar-like range `(1)`
- Action (4 continuous thrusters in `[-1, 1]`):
  - left forward, right forward, front vertical, rear vertical
  - mapped to force by `F = thrust_max * action`

## Reward

- Progress: `d_prev - d_now`
- Success bonus: `+10`
- Energy penalty: `-lambda * sum(a^2)` (`lambda=0.01` default)
- Optional angular-speed penalty via config

Info dict includes:
- `distance`
- `success`
- `energy`

## Physics Model

Custom water physics is applied each substep with `applyExternalForce` / `applyExternalTorque`:

1. Buoyancy  
   - `F_b = rho * g * V_displaced` upward (+z)
   - applied at center of buoyancy offset (not necessarily COM)

2. Drag (relative to current)  
   - current field `v_current(x)` is a gentle analytic swirl
   - relative velocity `v_rel = v - v_current`
   - default anisotropic linear drag in body frame:
     - `F_body = -k_drag * v_body`
   - optional quadratic mode:
     - `F_body = -0.5 * rho * Cd * A * v * |v|`

3. Angular damping  
   - `tau = -k_omega * omega`

4. Optional restoring torque (off by default)  
   - stabilizes roll/pitch toward zero:
   - `tau_restore = -k_p*[roll,pitch,0] - k_d*[w_roll,w_pitch,0]`

## Tuning Guide

Key parameters live in:
- `envs/submarine_search_env.py` (`SubmarineEnvConfig`)
- `physics/water_forces.py` (`WaterPhysicsConfig`)
- `physics/thrusters.py` (`ThrusterConfig`)

Recommended tuning order:

1. Neutral buoyancy:
   - set `displaced_volume ~= mass / rho`
   - env now auto-sets this by default on reset (`auto_neutral_buoyancy=True`) using loaded URDF mass

2. Drag:
   - increase `linear_drag` if drift is too high
   - keep lateral/vertical drag (`ky`, `kz`) higher than forward (`kx`)

3. Thruster strength:
   - adjust `thrust_max` if agent cannot maneuver or is too aggressive

4. Stability:
   - increase `angular_damping` for spin/jitter
   - optionally enable restoring torque for extra roll/pitch stability

## Troubleshooting

1. Exploding motion / unstable simulation
   - reduce `thrust_max`
   - increase drag/damping
   - reduce control aggressiveness
   - use smaller step (`sim_dt`) or higher `sim_substeps`

2. Excessive drift
   - increase linear drag
   - reduce current strength

3. Slow learning
   - increase training timesteps
   - normalize with `VecNormalize` (enabled by default in `train_sb3.py`)
   - tune reward penalty weights

4. No GUI appears
   - run `run_manual_control.py`
   - ensure desktop/driver supports OpenGL for PyBullet GUI

5. URDF fails to load with mesh errors
   - ensure `assets/meshes/*.STL` files referenced by `assets/submarine.urdf` are present
   - env automatically falls back to `assets/submarine_fallback.urdf` if mesh URDF load fails
