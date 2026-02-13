from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.submarine_search_env import SubmarineSearchEnv


def make_env():
    def _init():
        return SubmarineSearchEnv(render_mode=None)

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained policy on SubmarineSearchEnv.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .zip model file.")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vecnorm-path", type=str, default="models/vecnormalize.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    vecnorm_path = Path(args.vecnorm_path)

    vec_env = DummyVecEnv([make_env()])
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), vec_env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats: {vecnorm_path}")
    else:
        env = vec_env
        print("VecNormalize stats not found, evaluating without normalization.")

    if args.algo == "sac":
        model = SAC.load(str(model_path), env=env)
    else:
        model = PPO.load(str(model_path), env=env)

    rng = np.random.default_rng(args.seed)
    successes = 0
    returns = []
    lengths = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            ep_ret += float(reward[0])
            ep_len += 1
            ep_success = ep_success or bool(infos[0].get("success", False))

            if done:
                break

        successes += int(ep_success)
        returns.append(ep_ret)
        lengths.append(ep_len)
        print(f"Episode {ep + 1}/{args.episodes}: return={ep_ret:.2f} steps={ep_len} success={ep_success}")

        env.seed(int(rng.integers(0, 1_000_000)))

    success_rate = successes / args.episodes
    print("\nEvaluation summary")
    print(f"Success rate: {successes}/{args.episodes} = {100.0 * success_rate:.1f}%")
    print(f"Average return: {float(np.mean(returns)):.2f}")
    print(f"Average episode length: {float(np.mean(lengths)):.1f}")

    env.close()


if __name__ == "__main__":
    main()
