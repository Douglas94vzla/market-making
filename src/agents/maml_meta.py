import torch
import torch.nn as nn
from typing import Dict

class MetaController(nn.Module):
    """
    Mock network acting as the Meta-Controller parametrizing regime behavior
    (e.g., bounds for lambda_base, kappa over the HMM state space).
    """
    def __init__(self, in_features: int = 4, out_features: int = 2):
        super().__init__()
        # Input features: 4-dim (State Probabilities from MarketRegimeHMM)
        # Output features: 2-dim (lambda_base, kappa bounds)
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        """Outputs modified hyperparameters for the target regime."""
        out = self.fc(regime_probs)
        # Clamp outputs to safe RL shaping regimes
        # Column 0: lambda_base (expected around 0.001)
        # Column 1: kappa (expected around 0.0007)
        return torch.sigmoid(out) * 0.01

def compute_regime_loss(params: Dict[str, torch.Tensor], support_batch: torch.Tensor, target_rewards: torch.Tensor) -> torch.Tensor:
    """
    Dummy evaluation proxy for MAML. Simulates predicting the optimal hyperparameters 
    that maximize proxy returns (MSE of predicted hyperparameters vs target ones).
    In the real RL training script, this would be computed by policy trajectories.
    """
    # Simply using the linear layer mathematically
    pred = torch.nn.functional.linear(support_batch, params["fc.weight"], params["fc.bias"])
    pred = torch.sigmoid(pred) * 0.01
    loss = torch.nn.functional.mse_loss(pred, target_rewards)
    return loss

def maml_inner_loop(
    meta_params: Dict[str, torch.Tensor],
    support_batch: torch.Tensor,
    target_rewards: torch.Tensor,
    inner_lr: float = 0.01,
    n_inner_steps: int = 5,
) -> Dict[str, torch.Tensor]:
    """
    The MAML fast-adaptation step (inner loop) specifically mapped from Section 3.2.2.
    It takes the base model parameters and computes N gradient descent steps using the support data 
    while *maintaining the computation graph* for an outer loop optimization.
    
    Args:
        meta_params: Dictionary of parameters {name: tensor}.
        support_batch: 500 ticks for adaptation batch.
        target_rewards: Synthetic expected reward mapping (for validation test bounds)
        inner_lr: Step size \alpha for the adaptation step.
        n_inner_steps: Number of MAML iterations.
        
    Returns:
        adapted_params: Dictionary of adapted parameters.
    """
    # Start with a reference to the initial parameters
    adapted_params = {k: v.clone() for k, v in meta_params.items()}
    
    for step in range(n_inner_steps):
        # 1. Evaluate loss on current param snapshot
        loss = compute_regime_loss(adapted_params, support_batch, target_rewards)
        
        # 2. Compute derivatives directly via autograd while maintaining create_graph=True
        # This allows outer-loop meta-optimization to flow back through this trajectory
        grads = torch.autograd.grad(
            loss, 
            adapted_params.values(), 
            create_graph=True,
            retain_graph=True
        )
        
        # 3. Apply SGD manual update: theta'_i = theta_i - alpha * grad
        adapted_params = {
            k: v - inner_lr * g
            for (k, v), g in zip(adapted_params.items(), grads)
        }
        
    return adapted_params

# --- Design Decisions Note ---
# 1. Used explicit `torch.autograd.grad` rather than `optimizer.step()` as MAML requires graph retention through `create_graph=True`.
# 2. Re-created exactly the mapping syntax `adapted_params.items()` + zip `grads` mentioned in 3.2.2.
