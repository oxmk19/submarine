from __future__ import annotations

import argparse
from pathlib import Path
import sys

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.submarine_search_env import SubmarineSearchEnv


def make_env(seed: int):
    def _init():
        env = SubmarineSearchEnv(render_mode=None)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train SB3 policy on SubmarineSearchEnv.")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--no-vecnorm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([make_env(args.seed)])
    if args.no_vecnorm:
        env = vec_env
        vecnorm = None
    else:
        vecnorm = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        env = vecnorm

    if args.algo == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=args.seed,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=args.seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
        )

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    model_path = model_dir / f"{args.algo}_submarine_search"
    model.save(str(model_path))
    if vecnorm is not None:
        vecnorm.save(str(model_dir / "vecnormalize.pkl"))

    print(f"Saved model to: {model_path}.zip")
    if vecnorm is not None:
        print(f"Saved VecNormalize stats to: {model_dir / 'vecnormalize.pkl'}")

    env.close()


if __name__ == "__main__":
    main()
