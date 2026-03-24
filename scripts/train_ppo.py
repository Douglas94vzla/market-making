"""
train_ppo.py — Train the PPO agent using Stable-Baselines3.

Usage:
    uv run python scripts/train_ppo.py --total-timesteps 2000000 --n-envs 8

Takes ~6h on a standard CPU VPS (4 vCPU, 8 GB RAM).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.env.market_making_env import MarketMakingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO market making agent")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = make_vec_env(MarketMakingEnv, n_envs=args.n_envs)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.15,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=args.log_dir,
        device="cpu",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_mm",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(os.path.join(args.checkpoint_dir, "ppo_mm_final"))
    print(f"Training complete. Model saved to {args.checkpoint_dir}/ppo_mm_final")


if __name__ == "__main__":
    main()
