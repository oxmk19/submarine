from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

from physics.thrusters import ThrusterConfig, apply_thrusters
from physics.water_forces import WaterPhysicsConfig, apply_water_forces


@dataclass
class SubmarineEnvConfig:
    world_min: np.ndarray = field(default_factory=lambda: np.array([-20.0, -20.0, -20.0], dtype=np.float64))
    world_max: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, -1.0], dtype=np.float64))
    start_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -5.0], dtype=np.float64))
    start_quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    sim_dt: float = 1.0 / 60.0
    sim_substeps: int = 4
    max_steps: int = 2000
    target_radius: float = 1.5
    sonar_dir_noise_std: float = 0.03
    sonar_range_noise_std: float = 0.15
    action_energy_lambda: float = 0.01
    angular_speed_penalty_lambda: float = 0.0
    seed: Optional[int] = None
    urdf_path: Optional[str] = None
    thruster: ThrusterConfig = field(default_factory=ThrusterConfig)
    water: WaterPhysicsConfig = field(default_factory=WaterPhysicsConfig)
    auto_neutral_buoyancy: bool = True


class SubmarineSearchEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(
        self,
        config: Optional[SubmarineEnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else SubmarineEnvConfig()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(self.config.seed)
        self.client_id: Optional[int] = None
        self.submarine_id: Optional[int] = None
        self.target_visual_id: Optional[int] = None
        self.target_pos = np.zeros(3, dtype=np.float64)
        self._last_distance = 0.0
        self._step_count = 0

        self._connect_if_needed()

    def _connect_if_needed(self) -> None:
        if self.client_id is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

    def _urdf_path(self) -> str:
        if self.config.urdf_path is not None:
            return self.config.urdf_path
        root = Path(__file__).resolve().parents[1]
        return str(root / "assets" / "submarine.urdf")

    def _reset_world(self) -> None:
        assert self.client_id is not None
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -self.config.water.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.config.sim_dt / self.config.sim_substeps, physicsClientId=self.client_id)
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)

        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.resetBasePositionAndOrientation(
            plane_id,
            posObj=[0.0, 0.0, -20.0],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )

        self.submarine_id = p.loadURDF(
            self._urdf_path(),
            basePosition=self.config.start_pos.tolist(),
            baseOrientation=self.config.start_quat.tolist(),
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        if self.config.auto_neutral_buoyancy:
            total_mass = self._compute_total_mass(self.submarine_id)
            self.config.water.displaced_volume = total_mass / self.config.water.rho
        p.resetBaseVelocity(
            self.submarine_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

    def _compute_total_mass(self, body_id: int) -> float:
        assert self.client_id is not None
        total = float(p.getDynamicsInfo(body_id, -1, physicsClientId=self.client_id)[0])
        joint_count = p.getNumJoints(body_id, physicsClientId=self.client_id)
        for j in range(joint_count):
            total += float(p.getDynamicsInfo(body_id, j, physicsClientId=self.client_id)[0])
        return total

    def _sample_target(self) -> np.ndarray:
        low = self.config.world_min
        high = self.config.world_max
        return self._rng.uniform(low=low, high=high).astype(np.float64)

    def _update_target_visual(self) -> None:
        assert self.client_id is not None
        if self.render_mode != "human":
            return
        if self.target_visual_id is not None:
            p.removeBody(self.target_visual_id, physicsClientId=self.client_id)
            self.target_visual_id = None
        vis = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.35,
            rgbaColor=[1.0, 0.1, 0.1, 1.0],
            physicsClientId=self.client_id,
        )
        self.target_visual_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=vis,
            baseCollisionShapeIndex=-1,
            basePosition=self.target_pos.tolist(),
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )

    def _get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.client_id is not None
        assert self.submarine_id is not None
        pos, quat = p.getBasePositionAndOrientation(self.submarine_id, physicsClientId=self.client_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.submarine_id, physicsClientId=self.client_id)
        return (
            np.asarray(pos, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
            np.asarray(lin_vel, dtype=np.float64),
            np.asarray(ang_vel, dtype=np.float64),
        )

    def _get_observation(self) -> np.ndarray:
        pos, quat, lin_vel, ang_vel = self._get_state()
        to_target = self.target_pos - pos
        dist = float(np.linalg.norm(to_target))
        if dist < 1e-8:
            dir_vec = np.zeros(3, dtype=np.float64)
        else:
            dir_vec = to_target / dist

        noisy_dir = dir_vec + self._rng.normal(0.0, self.config.sonar_dir_noise_std, size=3)
        norm_noisy_dir = np.linalg.norm(noisy_dir)
        if norm_noisy_dir > 1e-8:
            noisy_dir = noisy_dir / norm_noisy_dir
        noisy_range = dist + float(self._rng.normal(0.0, self.config.sonar_range_noise_std))

        obs = np.concatenate([pos, lin_vel, ang_vel, quat, noisy_dir, [noisy_range]]).astype(np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._connect_if_needed()
        self._reset_world()

        self.target_pos = self._sample_target()
        self._update_target_visual()
        self._step_count = 0

        sub_pos, _, _, _ = self._get_state()
        self._last_distance = float(np.linalg.norm(sub_pos - self.target_pos))
        obs = self._get_observation()
        info = {"distance": self._last_distance, "success": False, "energy": 0.0}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.client_id is not None
        assert self.submarine_id is not None

        action_arr = np.asarray(action, dtype=np.float64).reshape(4)
        action_clipped = np.clip(action_arr, -1.0, 1.0)

        for _ in range(self.config.sim_substeps):
            apply_thrusters(
                client_id=self.client_id,
                body_id=self.submarine_id,
                action=action_clipped,
                config=self.config.thruster,
            )
            apply_water_forces(
                client_id=self.client_id,
                body_id=self.submarine_id,
                config=self.config.water,
            )
            p.stepSimulation(physicsClientId=self.client_id)

        obs = self._get_observation()
        pos, _, _, ang_vel = self._get_state()
        distance = float(np.linalg.norm(pos - self.target_pos))
        success = distance < self.config.target_radius
        self._step_count += 1
        truncated = self._step_count >= self.config.max_steps
        terminated = success

        progress = self._last_distance - distance
        energy = float(np.sum(np.square(action_clipped)))
        ang_speed_penalty = float(np.sum(np.square(ang_vel)))

        reward = progress
        reward += 10.0 if success else 0.0
        reward -= self.config.action_energy_lambda * energy
        reward -= self.config.angular_speed_penalty_lambda * ang_speed_penalty

        self._last_distance = distance
        info = {"distance": distance, "success": success, "energy": energy}

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        return None

    def close(self) -> None:
        if self.client_id is not None and p.isConnected(physicsClientId=self.client_id):
            p.disconnect(physicsClientId=self.client_id)
        self.client_id = None
        self.submarine_id = None
        self.target_visual_id = None
