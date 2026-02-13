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
    body_up_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64)
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
    current_strength: float = 0.0
    enable_restoring_torque: bool = True
    restoring_kp: np.ndarray = field(
        default_factory=lambda: np.array([80.0, 80.0, 0.0], dtype=np.float64)
    )
    restoring_kd: np.ndarray = field(
        default_factory=lambda: np.array([15.0, 15.0, 0.0], dtype=np.float64)
    )
    enable_attitude_hold: bool = True
    attitude_hold_kp: float = 220.0
    attitude_hold_kd: float = 55.0
    trim_quat_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
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

    r_world_from_body = _rotation_matrix(quat_world)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    body_up_local = np.asarray(config.body_up_axis_local, dtype=np.float64)
    body_up_norm = np.linalg.norm(body_up_local)
    if body_up_norm < 1e-9:
        body_up_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        body_up_local = body_up_local / body_up_norm
    body_up_world = r_world_from_body @ body_up_local

    # Axis-angle style tilt error that is independent of URDF forward-axis convention.
    tilt_error_world = np.cross(body_up_world, world_up)
    # Dampen only roll/pitch-rate component (exclude yaw about world-up).
    omega_tilt_world = ang_vel_world - np.dot(ang_vel_world, world_up) * world_up

    kp = float(np.mean(config.restoring_kp[:2]))
    kd = float(np.mean(config.restoring_kd[:2]))
    tau_world = kp * tilt_error_world - kd * omega_tilt_world
    return _clip_norm(tau_world, config.max_torque)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def compute_attitude_hold_torque(
    quat_world: np.ndarray,
    ang_vel_world: np.ndarray,
    config: WaterPhysicsConfig,
) -> np.ndarray:
    """
    PD torque driving current orientation toward trim_quat_world.
    Uses quaternion error to avoid axis-convention ambiguity.
    """
    if not config.enable_attitude_hold:
        return np.zeros(3, dtype=np.float64)

    q_current = np.asarray(quat_world, dtype=np.float64)
    q_target = np.asarray(config.trim_quat_world, dtype=np.float64)
    q_target = q_target / max(np.linalg.norm(q_target), 1e-12)
    q_current = q_current / max(np.linalg.norm(q_current), 1e-12)

    q_err = _quat_multiply(q_target, _quat_conjugate(q_current))
    if q_err[3] < 0.0:
        q_err = -q_err

    # Small-angle approximation: 2 * vector-part is orientation error in radians.
    rot_err = 2.0 * q_err[:3]
    tau = config.attitude_hold_kp * rot_err - config.attitude_hold_kd * ang_vel_world
    return _clip_norm(tau, config.max_torque)


def apply_water_forces(client_id: int, body_id: int, config: WaterPhysicsConfig) -> Dict[str, np.ndarray]:
    """
    Apply buoyancy, drag, angular damping, and optional restoring torque.

    Returns a dictionary with the individual force/torque vectors (world frame).
    """
    pos, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
    lin_vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=client_id)
    com_local = np.asarray(p.getDynamicsInfo(body_id, -1, physicsClientId=client_id)[3], dtype=np.float64)
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
    attitude_hold_torque = compute_attitude_hold_torque(
        quat_world=quat,
        ang_vel_world=ang_vel,
        config=config,
    )

    r_world_from_body = _rotation_matrix(quat)
    com_world = pos + (r_world_from_body @ com_local)
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
        posObj=com_world.tolist(),
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
    p.applyExternalTorque(
        objectUniqueId=body_id,
        linkIndex=-1,
        torqueObj=attitude_hold_torque.tolist(),
        flags=p.WORLD_FRAME,
        physicsClientId=client_id,
    )

    return {
        "buoyancy": buoyancy,
        "drag": drag,
        "damping_torque": damping_torque,
        "restoring_torque": restoring_torque,
        "attitude_hold_torque": attitude_hold_torque,
    }
