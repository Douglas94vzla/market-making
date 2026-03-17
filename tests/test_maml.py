import pytest
import torch
import torch.nn as nn
from src.agents.maml_meta import MetaController, maml_inner_loop, compute_regime_loss

@pytest.fixture
def dummy_maml_env():
    torch.manual_seed(42)
    # 4 state regimes input probability from HMM
    model = MetaController(in_features=4, out_features=2)
    
    # 5 samples representing a 500-tick regime "support batch" in the inner loop
    support_batch = torch.rand((5, 4))
    
    # Target values (we want the model to learn to output these hyper-parameters)
    target_rewards = torch.tensor([[0.001, 0.0007]] * 5)
    
    return model, support_batch, target_rewards

def test_maml_gradient_tracking(dummy_maml_env):
    """
    Validates that the MAML inner loop preserves the PyTorch computation graph
    for outer-loop Meta-Optimization backpropagation.
    """
    model, support_batch, target_rewards = dummy_maml_env
    
    # Enable requires_grad for outer parameters tracking
    meta_params = {k: v.clone().detach().requires_grad_(True) for k, v in model.named_parameters()}
    
    # Step 1: Run inner loop adaptation
    adapted_params = maml_inner_loop(
        meta_params=meta_params,
        support_batch=support_batch,
        target_rewards=target_rewards,
        inner_lr=1000.0, # High LR simply to mathematically escape torch.allclose bounds
        n_inner_steps=15
    )
    
    # Step 2: The parameters should have drifted
    assert not torch.allclose(adapted_params["fc.weight"], meta_params["fc.weight"])
    
    # Step 3: Compute outer query loss on new test batch
    query_batch = torch.rand((2, 4))
    query_targets = torch.tensor([[0.001, 0.0007], [0.001, 0.0007]])
    
    outer_loss = compute_regime_loss(adapted_params, query_batch, query_targets)
    
    # Step 4: Validate outer backward pass propagates through inner loop
    outer_loss.backward()
    
    # We must observe gradients strictly flowing back to the original `meta_params`
    # Otherwise, MAML logic is broken.
    for k, v in meta_params.items():
        assert v.grad is not None, f"Gradient vanished for parameter {k} in Meta-Controller"
        assert torch.sum(torch.abs(v.grad)) > 0.0, f"Outer gradients didn't modify parameter {k}"

# --- Design Decisions Note ---
# 1. We strictly re-created the `test_maml.py` constraints verifying MAML topology:
#    a. `adapted_params` separate from `meta_params`.
#    b. Check that gradients bridge the adapted gap back to `meta_params`.
# 2. Used standard `torch.nn.functional` proxy loss for rapid testing.
