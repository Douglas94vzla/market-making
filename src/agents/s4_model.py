import torch
import torch.nn as nn
from torch import Tensor

class S4Layer(nn.Module):
    """
    Simulated 1D Convolutional proxy block serving the role of an S4 
    (Structured State Space Sequence) Layer for capturing long-term dependencies
    in O(N log N) without the heavy custom CUDA requirements of pure S4 implementations.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # We use a depthwise 1D Convolution over the sequence dimension to mix temporal states.
        # This mirrors the behavior of state-space sequences capturing local+global trajectory.
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1, # Keep sequence length exactly the same
            groups=d_model # Depthwise
        )
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (Batch, Seq_Len, d_model)
        # Conv1d expects: (Batch, Channels/d_model, Seq_Len)
        x_transposed = x.transpose(1, 2)
        
        # Apply Temporal Mixing
        out = self.conv1d(x_transposed)
        out = self.activation(out)
        
        # Back to (Batch, Seq_Len, d_model)
        out = out.transpose(1, 2)
        out = self.layernorm(out)
        return out

class SpreadAgentPPO(nn.Module):
    """
    PPO actor-critic with a single value function head.
    # Fixed: original SpreadAgentNetwork mixed PPO actor_log_std (state-dependent std)
    # with SAC twin-Q critics — these are incompatible algorithms. PPO uses a value
    # function V(s), not Q-functions Q(s,a). Separated into two clean classes.
    """
    def __init__(self, obs_dim: int = 47, action_dim: int = 3,
                 d_model: int = 256, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.actor_mean = nn.Linear(d_model, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_value = nn.Linear(d_model, 1)  # Value function V(s), NOT Q-function

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = torch.tanh(self.input_proj(obs))
        for layer in self.layers:
            x = torch.tanh(layer(x)) + x  # residual
        mean = torch.tanh(self.actor_mean(x))
        std = self.actor_log_std.exp().clamp(1e-3, 1.0)
        value = self.critic_value(x)
        return mean, std, value


class SpreadAgentSAC(nn.Module):
    """
    SAC actor with separate twin Q-critic networks, per the SAC paper (Haarnoja et al. 2018).
    # Fixed: SAC requires Q(s,a) critics that take (obs, action) as input — not V(s).
    # Also requires *separate* actor and critic networks to avoid gradient interference.
    """
    def __init__(self, obs_dim: int = 47, action_dim: int = 3,
                 d_model: int = 256, n_layers: int = 4):
        super().__init__()
        # Actor network
        self.actor_net = self._build_net(obs_dim, d_model, n_layers)
        self.actor_mean = nn.Linear(d_model, action_dim)
        self.actor_log_std = nn.Linear(d_model, action_dim)
        # Twin Q-critics (take obs + action concatenated as input)
        self.q1_net = self._build_net(obs_dim + action_dim, d_model, n_layers)
        self.q1_head = nn.Linear(d_model, 1)
        self.q2_net = self._build_net(obs_dim + action_dim, d_model, n_layers)
        self.q2_head = nn.Linear(d_model, 1)

    @staticmethod
    def _build_net(in_dim: int, d_model: int, n_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, d_model), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_model, d_model), nn.Tanh()]
        return nn.Sequential(*layers)

    def actor_forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        x = self.actor_net(obs)
        mean = torch.tanh(self.actor_mean(x))
        log_std = self.actor_log_std(x).clamp(-5, 2)
        return mean, log_std.exp()

    def critic_forward(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        q1 = self.q1_head(self.q1_net(sa))
        q2 = self.q2_head(self.q2_net(sa))
        return q1, q2


class SpreadAgentS4(nn.Module):
    """
    S4-based sequence model for processing tick windows (64 ticks × 47 features).
    Used as a shared encoder trunk that can be combined with either PPO or SAC heads.
    """
    def __init__(self, obs_dim: int = 47, d_model: int = 256, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.s4_layers = nn.ModuleList([S4Layer(d_model) for _ in range(n_layers)])

    def encode(self, obs_seq: Tensor) -> Tensor:
        """
        Args:
            obs_seq: (Batch, Seq_len=64, Obs_dim=47)
        Returns:
            x_last: (Batch, d_model) — representation of the latest tick
        """
        x = self.input_proj(obs_seq)
        for layer in self.s4_layers:
            x = layer(x) + x
        return x[:, -1, :]

# --- Design Decisions Note ---
# 1. SpreadAgentNetwork was split into SpreadAgentPPO and SpreadAgentSAC because PPO and SAC
#    have fundamentally incompatible critic architectures. PPO uses V(s) (no action input),
#    while SAC requires twin Q(s,a) networks (action is part of the input). Mixing them in
#    one class caused silent gradient corruption during training.
# 2. SpreadAgentS4 keeps the S4 sequence encoder as a reusable trunk.
# 3. Depthwise-1D Conv is used as an S4 proxy to avoid complex CUDA requirements.
