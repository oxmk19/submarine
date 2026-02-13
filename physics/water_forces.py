from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pybullet as p


@dataclass
class WaterPhysicsConfig:
    rho: float = 1000.0
    gravity: float = 9.81
    displaced_volume: float = 0.262
    center_of_buoyancy_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.05], dtype=np.float64)
    )
    linear_drag: np.ndarray = field(
        default_factory=lambda: np.array([120.0, 220.0, 220.0], dtype=np.float64)
    )
    use_quadratic_drag: bool = False
    drag_coefficient: np.ndarray = field(
        default_factory=lambda: np.array([0.7, 1.0, 1.0], dtype=np.float64)
    )
    reference_area: np.ndarray = field(
        default_factory=lambda: np.array([0.8, 1.0, 1.0], dtype=np.float64)
    )
    angular_damping: np.ndarray = field(
        default_factory=lambda: np.array([32.0, 32.0, 20.0], dtype=np.float64)
    )
    current_strength: float = 0.35
    enable_restoring_torque: bool = False
    restoring_kp: np.ndarray = field(
        default_factory=lambda: np.array([80.0, 80.0, 0.0], dtype=np.float64)
    )
    restoring_kd: np.ndarray = field(
        default_factory=lambda: np.array([15.0, 15.0, 0.0], dtype=np.float64)
    )
    max_force: float = 4000.0
    max_torque: float = 2200.0


def _rotation_matrix(quat: np.ndarray) -> np.ndarray:
    m = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64)
    return m.reshape(3, 3)


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= max_norm:
        return v
    if norm < 1e-12:
        return np.zeros_like(v)
    return v * (max_norm / norm)


def current_velocity_field(position_world: np.ndarray, config: WaterPhysicsConfig) -> np.ndarray:
    """Simple gentle swirl current field in world frame."""
    x, y, z = position_world
    s = config.current_strength
    vx = -s * (y / 20.0)
    vy = s * (x / 20.0)
    vz = 0.05 * s * np.sin(0.2 * z)
    return np.array([vx, vy, vz], dtype=np.float64)


def compute_buoyancy_force(config: WaterPhysicsConfig) -> np.ndarray:
    # Buoyancy acts opposite gravity (+z in this world setup).
    return np.array([0.0, 0.0, config.rho * config.gravity * config.displaced_volume], dtype=np.float64)


def compute_drag_force(
    lin_vel_world: np.ndarray,
    quat_world: np.ndarray,
    position_world: np.ndarray,
    config: WaterPhysicsConfig,
) -> np.ndarray:
    v_current_world = current_velocity_field(position_world=position_world, config=config)
    v_rel_world = lin_vel_world - v_current_world
    r_world_from_body = _rotation_matrix(quat_world)
    v_rel_body = r_world_from_body.T @ v_rel_world

    if config.use_quadratic_drag:
        f_body = -0.5 * config.rho * config.drag_coefficient * config.reference_area * v_rel_body * np.abs(v_rel_body)
    else:
        f_body = -config.linear_drag * v_rel_body

    f_world = r_world_from_body @ f_body
    return _clip_norm(f_world, config.max_force)


def compute_angular_damping_torque(ang_vel_world: np.ndarray, config: WaterPhysicsConfig) -> np.ndarray:
    tau = -config.angular_damping * ang_vel_world
    return _clip_norm(tau, config.max_torque)


def compute_restoring_torque(
    quat_world: np.ndarray,
    ang_vel_world: np.ndarray,
    config: WaterPhysicsConfig,
) -> np.ndarray:
    if not config.enable_restoring_torque:
        return np.zeros(3, dtype=np.float64)

    roll, pitch, _ = p.getEulerFromQuaternion(quat_world.tolist())
    r_world_from_body = _rotation_matrix(quat_world)
    omega_body = r_world_from_body.T @ ang_vel_world

    error = np.array([roll, pitch, 0.0], dtype=np.float64)
    damping = np.array([omega_body[0], omega_body[1], 0.0], dtype=np.float64)
    tau_body = -config.restoring_kp * error - config.restoring_kd * damping
    tau_world = r_world_from_body @ tau_body
    return _clip_norm(tau_world, config.max_torque)


def apply_water_forces(client_id: int, body_id: int, config: WaterPhysicsConfig) -> Dict[str, np.ndarray]:
    """
    Apply buoyancy, drag, angular damping, and optional restoring torque.

    Returns a dictionary with the individual force/torque vectors (world frame).
    """
    pos, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
    lin_vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=client_id)
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    lin_vel = np.asarray(lin_vel, dtype=np.float64)
    ang_vel = np.asarray(ang_vel, dtype=np.float64)

    buoyancy = compute_buoyancy_force(config=config)
    drag = compute_drag_force(
        lin_vel_world=lin_vel,
        quat_world=quat,
        position_world=pos,
        config=config,
    )
    damping_torque = compute_angular_damping_torque(ang_vel_world=ang_vel, config=config)
    restoring_torque = compute_restoring_torque(
        quat_world=quat,
        ang_vel_world=ang_vel,
        config=config,
    )

    r_world_from_body = _rotation_matrix(quat)
    cob_world = pos + (r_world_from_body @ config.center_of_buoyancy_local)

    p.applyExternalForce(
        objectUniqueId=body_id,
        linkIndex=-1,
        forceObj=buoyancy.tolist(),
        posObj=cob_world.tolist(),
        flags=p.WORLD_FRAME,
        physicsClientId=client_id,
    )
    p.applyExternalForce(
        objectUniqueId=body_id,
        linkIndex=-1,
        forceObj=drag.tolist(),
        posObj=pos.tolist(),
        flags=p.WORLD_FRAME,
        physicsClientId=client_id,
    )
    p.applyExternalTorque(
        objectUniqueId=body_id,
        linkIndex=-1,
        torqueObj=damping_torque.tolist(),
        flags=p.WORLD_FRAME,
        physicsClientId=client_id,
    )
    p.applyExternalTorque(
        objectUniqueId=body_id,
        linkIndex=-1,
        torqueObj=restoring_torque.tolist(),
        flags=p.WORLD_FRAME,
        physicsClientId=client_id,
    )

    return {
        "buoyancy": buoyancy,
        "drag": drag,
        "damping_torque": damping_torque,
        "restoring_torque": restoring_torque,
    }
