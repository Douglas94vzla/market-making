import pytest
import torch
from src.agents.s4_model import SpreadAgentNetwork, S4Layer

@pytest.fixture
def network_setup():
    torch.manual_seed(42)
    # Instantiate the network using default parameters strictly from Document configuration
    # obs_dim = 47, action_dim = 3, seq_len = 64
    network = SpreadAgentNetwork(
        obs_dim=47,
        action_dim=3,
        d_model=256,
        n_layers=4,
        seq_len=64
    )
    
    # Generate dummy batch mimicking real Gym Rollouts
    # (batch_size, sequence_length, observation_dimension)
    batch_size = 10
    dummy_obs_seq = torch.rand((batch_size, 64, 47))
    dummy_actions = torch.rand((batch_size, 3))
    
    return network, dummy_obs_seq, dummy_actions

def test_s4_layer_shape_preservation():
    """
    Validates that the deep convolution 1D proxy for S4 perfectly maps temporal outputs
    without mutating the `seq_len` or dimension vectors.
    """
    s4_block = S4Layer(d_model=256)
    
    # Fake embedding inside network loop -> (batch, seq, d_model)
    dummy_input = torch.rand((5, 64, 256))
    
    out = s4_block(dummy_input)
    
    assert out.shape == (5, 64, 256), "S4 Proxy altered the tensor temporal alignment."

def test_spread_network_actor_forward(network_setup):
    """
    Validates the Policy Actor mapping 64-length sequences down to continuous Gaussian params
    `mean` (3,) and `std` (3,).
    """
    network, dummy_obs_seq, _ = network_setup
    
    mean, std = network(dummy_obs_seq)
    
    # Actor returns predictions for action_dim length
    assert mean.shape == (10, 3), f"Expected tensor shape (10, 3) got {mean.shape}"
    assert std.shape == (10, 3), f"Expected tensor shape (10, 3) got {std.shape}"
    
    # Output of Tanh must be dynamically within [-1, 1] bounds, ready to be expanded per Gym
    assert torch.all(mean >= -1.0) and torch.all(mean <= 1.0)
    
    # Standard deviation clamps are [1e-3, 1.0] from Spec
    assert torch.all(std >= 1e-3) and torch.all(std <= 1.0)

def test_spread_network_critic_forward(network_setup):
    """
    Validates the Q-Twins evaluating State+Action tuples correctly.
    """
    network, dummy_obs_seq, dummy_actions = network_setup
    
    q1, q2 = network.evaluate_critic(dummy_obs_seq, dummy_actions)
    
    # Critic must evaluate a singular dimensional Q scalar representing expected return
    assert q1.shape == (10, 1), f"Expected (10, 1) Q-values got {q1.shape}"
    assert q2.shape == (10, 1)

# --- Design Decisions Note ---
# 1. Used `1D Conv` structure to prove dimensional outputs, keeping computation extremely swift while validating parameters tracking.
# 2. Confirmed Actor constraint maps to strict Tanh ranges required for safe limit order placements (Actions bound 0.0 to 1.0 / 0.0 to 5.0 later projected linearly from the RL continuous network distributions [-1, 1]).
