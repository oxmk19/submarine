from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import pybullet_data
from PyQt5 import QtCore, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.water_forces import WaterPhysicsConfig, apply_water_forces


@dataclass
class MotorSpec:
    name: str
    local_position: np.ndarray
    local_direction: np.ndarray
    max_force: float


def _rotation_matrix(quat: np.ndarray) -> np.ndarray:
    m = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64)
    return m.reshape(3, 3)


def _mesh_paths_exist(urdf_path: Path) -> bool:
    if not urdf_path.exists():
        return False
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError:
        return False
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename and not (urdf_path.parent / filename).exists():
            return False
    return True


def build_motor_specs() -> List[MotorSpec]:
    # Local positions from your URDF joint origins.
    return [
        # Surge/Yaw: rear + bow thrusters along +Y.
        MotorSpec("Main_motor", np.array([0.017189, -1.8618, 0.033821]), np.array([0.0, 1.0, 0.0]), 320.0),
        MotorSpec("Motor_left", np.array([-0.58027, 0.12677, 0.0054523]), np.array([0.0, 1.0, 0.0]), 180.0),
        MotorSpec("Motor_right", np.array([0.61465, 0.12677, 0.0054523]), np.array([0.0, 1.0, 0.0]), 180.0),
        # Heave/Pitch: vertical thrusters (+Z for positive command).
        MotorSpec("Motor_up1", np.array([0.017189, 0.50339, 0.033821]), np.array([0.0, 0.0, 1.0]), 140.0),
        MotorSpec("Motor_up2", np.array([0.017189, -0.25861, 0.033821]), np.array([0.0, 0.0, 1.0]), 140.0),
        MotorSpec("Motor_down1", np.array([0.017189, 0.50339, -0.11858]), np.array([0.0, 0.0, 1.0]), 140.0),
        MotorSpec("Motor_down2", np.array([0.017189, -0.25861, -0.11858]), np.array([0.0, 0.0, 1.0]), 140.0),
        # Sway/Roll: side array as lateral thrusters.
        MotorSpec("Motor_right1", np.array([0.16959, 0.73199, 0.033821]), np.array([1.0, 0.0, 0.0]), 120.0),
        MotorSpec("Motor_right2", np.array([0.16959, -0.48721, 0.033821]), np.array([1.0, 0.0, 0.0]), 120.0),
        MotorSpec("Motor_left1", np.array([-0.13013, 0.73199, 0.033821]), np.array([-1.0, 0.0, 0.0]), 120.0),
        MotorSpec("Motor_left2", np.array([-0.13267, -0.48721, 0.033821]), np.array([-1.0, 0.0, 0.0]), 120.0),
    ]


class SubmarineQtSimulator:
    def __init__(self, sim_dt: float = 1.0 / 60.0, sim_substeps: int = 4) -> None:
        self.sim_dt = sim_dt
        self.sim_substeps = sim_substeps
        self.client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.client_id)

        self.urdf_path = ROOT / "assets" / "submarine.urdf"
        self.fallback_urdf_path = ROOT / "assets" / "submarine_fallback.urdf"
        self.motor_specs = build_motor_specs()
        self.motor_commands: Dict[str, float] = {m.name: 0.0 for m in self.motor_specs}

        self.water = WaterPhysicsConfig()
        self.water.current_strength = 0.0
        self.water.enable_attitude_hold = False
        self.water.enable_restoring_torque = False
        self.water.angular_damping = np.array([6.0, 6.0, 4.0], dtype=np.float64)

        self.body_id = -1
        self.reset()

    def _compute_total_mass(self, body_id: int) -> float:
        total = float(p.getDynamicsInfo(body_id, -1, physicsClientId=self.client_id)[0])
        for j in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
            total += float(p.getDynamicsInfo(body_id, j, physicsClientId=self.client_id)[0])
        return total

    def reset(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -self.water.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.sim_dt / self.sim_substeps, physicsClientId=self.client_id)

        plane_urdf = str(Path(pybullet_data.getDataPath()) / "plane.urdf")
        plane_id = p.loadURDF(plane_urdf, physicsClientId=self.client_id)
        p.resetBasePositionAndOrientation(plane_id, [0.0, 0.0, -20.0], [0.0, 0.0, 0.0, 1.0], physicsClientId=self.client_id)

        start_pos = [0.0, 0.0, -5.0]
        start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        if _mesh_paths_exist(self.urdf_path):
            urdf_to_load = str(self.urdf_path)
        else:
            urdf_to_load = str(self.fallback_urdf_path)
            print(f"[qt_motor_controller] Using fallback URDF: {urdf_to_load}")

        self.body_id = p.loadURDF(
            urdf_to_load,
            basePosition=start_pos,
            baseOrientation=start_quat,
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(self.body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=self.client_id)

        total_mass = self._compute_total_mass(self.body_id)
        self.water.displaced_volume = total_mass / self.water.rho
        com_local = np.asarray(p.getDynamicsInfo(self.body_id, -1, physicsClientId=self.client_id)[3], dtype=np.float64)
        self.water.center_of_buoyancy_local = com_local.copy()
        self.water.trim_quat_world = np.array(start_quat, dtype=np.float64)

        for k in self.motor_commands:
            self.motor_commands[k] = 0.0

    def set_motor(self, name: str, command: float) -> None:
        self.motor_commands[name] = float(np.clip(command, -1.0, 1.0))

    def apply_motor_forces(self) -> None:
        pos, quat = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        r_world_from_body = _rotation_matrix(quat)

        for spec in self.motor_specs:
            cmd = self.motor_commands[spec.name]
            if abs(cmd) < 1e-8:
                continue
            force_body = spec.local_direction * (spec.max_force * cmd)
            force_world = r_world_from_body @ force_body
            force_pos_world = pos + (r_world_from_body @ spec.local_position)
            p.applyExternalForce(
                objectUniqueId=self.body_id,
                linkIndex=-1,
                forceObj=force_world.tolist(),
                posObj=force_pos_world.tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.client_id,
            )

    def step(self) -> None:
        for _ in range(self.sim_substeps):
            self.apply_motor_forces()
            apply_water_forces(self.client_id, self.body_id, self.water)
            p.stepSimulation(physicsClientId=self.client_id)

        pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=8.0,
            cameraYaw=35.0,
            cameraPitch=-20.0,
            cameraTargetPosition=[float(pos[0]), float(pos[1]), float(pos[2])],
            physicsClientId=self.client_id,
        )

    def close(self) -> None:
        if p.isConnected(physicsClientId=self.client_id):
            p.disconnect(physicsClientId=self.client_id)


class MotorControllerWindow(QtWidgets.QWidget):
    def __init__(self, sim: SubmarineQtSimulator) -> None:
        super().__init__()
        self.sim = sim
        self.setWindowTitle("Submarine Motor Qt Controller")
        self.setMinimumWidth(520)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(QtWidgets.QLabel("Motor Commands (-1.0 to +1.0)"))

        self.sliders: Dict[str, QtWidgets.QSlider] = {}
        self.value_labels: Dict[str, QtWidgets.QLabel] = {}

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)

        for spec in self.sim.motor_specs:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)
            slider.valueChanged.connect(self._make_slider_handler(spec.name))
            value = QtWidgets.QLabel("0.00")
            value.setFixedWidth(48)

            row_layout.addWidget(slider)
            row_layout.addWidget(value)
            form.addRow(spec.name, row)

            self.sliders[spec.name] = slider
            self.value_labels[spec.name] = value

        scroll.setWidget(panel)
        main_layout.addWidget(scroll)

        button_row = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._on_pause_toggled)
        button_row.addWidget(self.pause_button)

        zero_btn = QtWidgets.QPushButton("Zero All Motors")
        zero_btn.clicked.connect(self.zero_all)
        button_row.addWidget(zero_btn)

        reset_btn = QtWidgets.QPushButton("Reset Simulation")
        reset_btn.clicked.connect(self.reset_sim)
        button_row.addWidget(reset_btn)

        main_layout.addLayout(button_row)

        self.status = QtWidgets.QLabel("Running")
        main_layout.addWidget(self.status)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(self.sim.sim_dt * 1000))

    def _make_slider_handler(self, motor_name: str):
        def _handler(value: int) -> None:
            cmd = value / 100.0
            self.sim.set_motor(motor_name, cmd)
            self.value_labels[motor_name].setText(f"{cmd:+.2f}")

        return _handler

    def _on_pause_toggled(self, paused: bool) -> None:
        self.pause_button.setText("Resume" if paused else "Pause")
        self.status.setText("Paused" if paused else "Running")

    def zero_all(self) -> None:
        for name, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            self.sim.set_motor(name, 0.0)
            self.value_labels[name].setText("0.00")

    def reset_sim(self) -> None:
        self.sim.reset()
        self.zero_all()
        self.status.setText("Reset complete")

    def _tick(self) -> None:
        if not self.pause_button.isChecked():
            self.sim.step()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.sim.close()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    sim = SubmarineQtSimulator()
    w = MotorControllerWindow(sim)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
