from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.submarine_search_env import SubmarineSearchEnv


def main() -> None:
    env = SubmarineSearchEnv(render_mode="human")
    obs, info = env.reset()
    del obs
    print("Manual control:")
    print("W/S: forward/back, A/D: yaw left/right")
    print("I/K: up/down, J/L: pitch up/down, R: reset")
    print("Ctrl+C to quit.")

    step = 0
    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=env.client_id)
            action = np.zeros(4, dtype=np.float32)

            if ord("w") in keys and keys[ord("w")] & p.KEY_IS_DOWN:
                action[0] += 0.7
                action[1] += 0.7
            if ord("s") in keys and keys[ord("s")] & p.KEY_IS_DOWN:
                action[0] -= 0.7
                action[1] -= 0.7
            if ord("a") in keys and keys[ord("a")] & p.KEY_IS_DOWN:
                action[0] -= 0.5
                action[1] += 0.5
            if ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:
                action[0] += 0.5
                action[1] -= 0.5

            if ord("i") in keys and keys[ord("i")] & p.KEY_IS_DOWN:
                action[2] += 0.7
                action[3] += 0.7
            if ord("k") in keys and keys[ord("k")] & p.KEY_IS_DOWN:
                action[2] -= 0.7
                action[3] -= 0.7
            if ord("j") in keys and keys[ord("j")] & p.KEY_IS_DOWN:
                action[2] += 0.6
                action[3] -= 0.6
            if ord("l") in keys and keys[ord("l")] & p.KEY_IS_DOWN:
                action[2] -= 0.6
                action[3] += 0.6

            if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                env.reset()

            _, reward, terminated, truncated, info = env.step(action)
            step += 1
            if step % 120 == 0:
                print(
                    f"step={step} distance={info['distance']:.2f} reward={reward:.3f} "
                    f"success={info['success']}"
                )

            if terminated or truncated:
                print(f"Episode ended. success={info['success']} distance={info['distance']:.2f}")
                env.reset()

            time.sleep(env.config.sim_dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
