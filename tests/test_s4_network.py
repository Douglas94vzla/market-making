"""
Tests for S4 sequence model and the split PPO/SAC agent networks.

SpreadAgentNetwork was split into SpreadAgentPPO and SpreadAgentSAC because
the original class mixed incompatible PPO and SAC heads.
"""
import pytest
import torch
from src.agents.s4_model import SpreadAgentPPO, SpreadAgentSAC, SpreadAgentS4, S4Layer

def test_s4_layer_shape_preservation():
    """S4Layer must preserve (batch, seq_len, d_model) shape exactly."""
    s4_block = S4Layer(d_model=256)
    dummy_input = torch.rand((5, 64, 256))
    out = s4_block(dummy_input)
    assert out.shape == (5, 64, 256), "S4 proxy altered the tensor temporal alignment."


def test_s4_encoder_output_shape():
    """SpreadAgentS4 encoder returns (batch, d_model) for latest tick."""
    torch.manual_seed(42)
    encoder = SpreadAgentS4(obs_dim=47, d_model=256, n_layers=4)
    obs_seq = torch.rand((10, 64, 47))
    x_last = encoder.encode(obs_seq)
    assert x_last.shape == (10, 256)


class TestSpreadAgentPPO:
    @pytest.fixture
    def ppo(self):
        torch.manual_seed(42)
        return SpreadAgentPPO(obs_dim=47, action_dim=3, d_model=256, n_layers=4)

    def test_forward_shapes(self, ppo):
        obs = torch.rand((10, 47))
        mean, std, value = ppo(obs)
        assert mean.shape == (10, 3)
        assert std.shape == (3,)
        assert value.shape == (10, 1)

    def test_actor_mean_bounded(self, ppo):
        obs = torch.rand((10, 47))
        mean, _, _ = ppo(obs)
        assert torch.all(mean >= -1.0) and torch.all(mean <= 1.0)

    def test_actor_std_clamped(self, ppo):
        obs = torch.rand((10, 47))
        _, std, _ = ppo(obs)
        assert torch.all(std >= 1e-3) and torch.all(std <= 1.0)

    def test_has_no_twin_q_heads(self, ppo):
        assert not hasattr(ppo, 'critic_q1'), "PPO should not have critic_q1"
        assert not hasattr(ppo, 'critic_q2'), "PPO should not have critic_q2"


class TestSpreadAgentSAC:
    @pytest.fixture
    def sac(self):
        torch.manual_seed(42)
        return SpreadAgentSAC(obs_dim=47, action_dim=3, d_model=256, n_layers=4)

    def test_actor_shapes(self, sac):
        obs = torch.rand((10, 47))
        mean, std = sac.actor_forward(obs)
        assert mean.shape == (10, 3)
        assert std.shape == (10, 3)

    def test_actor_mean_bounded(self, sac):
        obs = torch.rand((10, 47))
        mean, _ = sac.actor_forward(obs)
        assert torch.all(mean >= -1.0) and torch.all(mean <= 1.0)

    def test_twin_critic_shapes(self, sac):
        obs = torch.rand((10, 47))
        action = torch.rand((10, 3))
        q1, q2 = sac.critic_forward(obs, action)
        assert q1.shape == (10, 1)
        assert q2.shape == (10, 1)

    def test_twin_critics_differ(self, sac):
        torch.manual_seed(0)
        obs = torch.rand((10, 47))
        action = torch.rand((10, 3))
        q1, q2 = sac.critic_forward(obs, action)
        assert not torch.allclose(q1, q2), "Twin critics should not be identical"

    def test_has_no_ppo_value_head(self, sac):
        assert not hasattr(sac, 'critic_value'), "SAC should not have critic_value"
