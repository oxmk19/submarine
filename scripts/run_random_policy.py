from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.submarine_search_env import SubmarineSearchEnv


def main() -> None:
    env = SubmarineSearchEnv(render_mode=None)
    rng = np.random.default_rng(7)
    episodes = 10

    success_count = 0
    returns = []
    max_abs_vel = 0.0

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            done = False
            ep_return = 0.0
            step_count = 0

            while not done:
                action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_return += reward
                step_count += 1

                if not np.all(np.isfinite(obs)):
                    raise RuntimeError(f"Non-finite observation at episode {ep}, step {step_count}")

                pos = obs[0:3]
                vel = obs[3:6]
                ang_vel = obs[6:9]
                if not np.all(np.isfinite(pos)):
                    raise RuntimeError(f"Non-finite position at episode {ep}, step {step_count}")
                if np.max(np.abs(pos)) > 1e3:
                    raise RuntimeError(f"Position diverged at episode {ep}, step {step_count}: {pos}")

                speed_metric = max(float(np.max(np.abs(vel))), float(np.max(np.abs(ang_vel))))
                max_abs_vel = max(max_abs_vel, speed_metric)
                if np.max(np.abs(vel)) > 100.0 or np.max(np.abs(ang_vel)) > 50.0:
                    raise RuntimeError(
                        f"Velocity bound exceeded at episode {ep}, step {step_count}: vel={vel}, ang_vel={ang_vel}"
                    )

            success_count += int(info["success"])
            returns.append(ep_return)
            print(
                f"Episode {ep + 1}/{episodes} return={ep_return:.2f} "
                f"steps={step_count} success={info['success']}"
            )

        print("\nRandom policy stability check passed.")
        print(f"Successes: {success_count}/{episodes}")
        print(f"Average return: {float(np.mean(returns)):.2f}")
        print(f"Max abs vel observed: {max_abs_vel:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
