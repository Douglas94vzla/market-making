import os
from typing import Dict, Any
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Ensure the module can find MarketMakingEnv
from src.env.market_making_env import MarketMakingEnv

def env_creator(env_config: Dict[str, Any]):
    return MarketMakingEnv(
        max_inventory=env_config.get("max_inventory", 1000.0),
        max_qty=env_config.get("max_qty", 10.0),
        sigma_window_60s=env_config.get("sigma_window_60s", 10.0)
    )

def get_ppo_config() -> PPOConfig:
    """
    Returns the PPO Configuration strictly matching Section 3.3 specifications.
    lr=3e-4, clip_param=0.15, train_batch_size=4096, gamma=0.999, lambda=0.95, grad_clip=0.5
    """
    config = (
        PPOConfig()
        .environment(env="MarketMakingEnv")
        .framework("torch")
        .training(
            lr=3e-4,
            clip_param=0.15,
            train_batch_size=4096,
            gamma=0.999,
            lambda_=0.95,
            grad_clip=0.5,
            vf_clip_param=10.0, # Standard scaling for value function clip
            model={
                # Baseline relies on standard MLP until S4 architecture in Phase 7
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
        # Using simple rollouts for the baseline test
        .rollouts(num_rollout_workers=1)
    )
    return config

def train_baseline(iterations: int = 1, checkpoint_dir: str = "./ray_results"):
    """
    Initializes Ray, Registers the Env, and runs PPO training iterations.
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    register_env("MarketMakingEnv", env_creator)
    
    config = get_ppo_config()
    algo = config.build()
    
    print(f"Starting PPO Baseline training for {iterations} iterations...")
    for i in range(iterations):
        result = algo.train()
        print(f"Iteration {i+1}: reward_mean={result.get('env_runners', {}).get('episode_reward_mean', result.get('episode_reward_mean', 'N/A'))}")
        
    checkpoint_path = algo.save(checkpoint_dir)
    print(f"Checkpoint saved at {checkpoint_path}")
    algo.stop()
    ray.shutdown()
    
    return checkpoint_path

if __name__ == "__main__":
    train_baseline(iterations=5)

# --- Design Decisions Note ---
# 1. We strictly bound `clip_param=0.15` to ensure the mathematical policy stability requested for Market Making sensitive actions.
# 2. `train_batch_size=4096` requires a bit of memory but guarantees the batch has sufficient variance for gradient steps.
# 3. Model defaults to standard MLP for the Baseline. The architectural (S4 state-space) upgrade happens strictly in Phase 7 MARL jerárquico.
