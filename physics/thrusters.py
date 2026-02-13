from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybullet as p


@dataclass
class ThrusterConfig:
    thrust_max: float = 260.0
    local_positions: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.58027, 0.12677, 0.00545],  # left forward (matches Motor_left joint region)
                [0.61465, 0.12677, 0.00545],  # right forward (matches Motor_right joint region)
                [0.01719, 0.50339, -0.04238],  # front vertical (between up/down front motors)
                [0.01719, -0.25861, -0.04238],  # rear vertical (between up/down rear motors)
            ],
            dtype=np.float64,
        )
    )
    local_directions: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.0, 1.0, 0.0],  # left forward
                [0.0, 1.0, 0.0],  # right forward
                [0.0, 0.0, 1.0],  # front vertical
                [0.0, 0.0, 1.0],  # rear vertical
            ],
            dtype=np.float64,
        )
    )


def _rotation_matrix(quat: np.ndarray) -> np.ndarray:
    m = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64)
    return m.reshape(3, 3)


def apply_thrusters(
    client_id: int,
    body_id: int,
    action: np.ndarray,
    config: ThrusterConfig,
) -> np.ndarray:
    """
    Apply 4 thruster forces to a single rigid body in world frame.

    Returns the clipped action used for force generation.
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] != 4:
        raise ValueError(f"Expected action shape (4,), got {action.shape}")

    clipped = np.clip(action, -1.0, 1.0)
    base_pos, base_quat = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
    base_pos = np.asarray(base_pos, dtype=np.float64)
    base_quat = np.asarray(base_quat, dtype=np.float64)

    r_world_from_body = _rotation_matrix(base_quat)
    local_positions = np.asarray(config.local_positions, dtype=np.float64)
    local_directions = np.asarray(config.local_directions, dtype=np.float64)

    for i in range(4):
        force_body = local_directions[i] * (config.thrust_max * clipped[i])
        force_world = r_world_from_body @ force_body
        force_pos_world = base_pos + (r_world_from_body @ local_positions[i])
        p.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=-1,
            forceObj=force_world.tolist(),
            posObj=force_pos_world.tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=client_id,
        )

    return clipped
