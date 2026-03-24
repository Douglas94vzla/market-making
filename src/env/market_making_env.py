import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MarketMakingEnv(gym.Env):
    """
    Market Making Environment for Deep Reinforcement Learning.
    Follows Phase 2 specifications (Capa 1).
    """

    def __init__(self, 
                 max_inventory: float = 1000.0,
                 max_qty: float = 10.0,
                 sigma_window_60s: float = 10.0,
                 maker_rebate: float = 0.0002,
                 taker_fee: float = 0.0004,
                 lambda_base: float = 0.001,
                 kappa: float = 0.0007,
                 rho: float = 0.0001):
        super().__init__()
        
        self.max_inventory = max_inventory
        self.max_qty = max_qty
        self.sigma = sigma_window_60s
        self.maker_rebate = maker_rebate
        self.taker_fee = taker_fee
        
        # Reward shaping parameters
        self.lambda_base = lambda_base
        self.kappa = kappa
        self.rho = rho
        
        # Action Space: 3D Continuous (delta_bid, delta_ask, qty)
        # delta ranges [0, 5], will be scaled by sigma internally
        # qty ranges [0, 1], will be scaled by max_qty internally
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([5.0, 5.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Observation Space: 47D Continuous [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(47,), 
            dtype=np.float32
        )
        
        # Internal State (Dummy proxy for Phase 2 before Rust integration)
        self.current_inventory = 0.0
        self.avg_entry_price = 0.0
        self.mid_price = 100000.0 # e.g. BTC
        self.realized_pnl = 0.0
        
        self.current_step = 0
        self.max_steps = 1000

        # Sigma clip threshold (p95 of historical sigma) for reward normalization
        # Fixed: prevents reward collapse to 0 during flash crashes
        self._sigma_max_p95: float = 0.002

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_inventory = 0.0
        self.avg_entry_price = 0.0
        self.mid_price = 100000.0
        self.realized_pnl = 0.0
        self.current_step = 0
        
        # Return dummy observation for now within [-1, 1] bounds
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        """
        Action: [delta_bid_sigma, delta_ask_sigma, qty_pct]
        """
        # Un-normalize actions
        delta_bid = action[0] * self.sigma
        delta_ask = action[1] * self.sigma
        qty = action[2] * self.max_qty
        
        # Place orders based on current mid_price
        bid_price = self.mid_price - delta_bid
        ask_price = self.mid_price + delta_ask
        
        # Simulated Market Dynamics (Random Walk)
        # Volatility ~ N(0, 1) * sigma_tick
        tick_change = self.np_random.normal(0, self.sigma / 10.0)
        new_mid = self.mid_price + tick_change
        
        # Simulated Fills (Naive model for test phase: if price crosses our limit)
        # Real backtesting with microstructure happens in Phase 3
        fill_occurred = False
        step_realized_pnl = 0.0
        taker_cost = 0.0
        
        # Check sell fill (market bought from us)
        if new_mid > ask_price and qty > 0:
            fill_occurred = True
            # We sold at ask_price (we are shorting or closing long)
            if self.current_inventory > 0: # Closing long
                step_realized_pnl += (ask_price - self.avg_entry_price) * min(qty, self.current_inventory)
                step_realized_pnl += (ask_price * min(qty, self.current_inventory)) * self.maker_rebate
            
            self.current_inventory -= qty
            
            if self.current_inventory < 0 and self.avg_entry_price == 0:
                self.avg_entry_price = ask_price # Opening new short
            elif self.current_inventory < 0: # Adding to short
                pass # simplification for avg price

        # Check buy fill (market sold to us)
        elif new_mid < bid_price and qty > 0:
            fill_occurred = True
            # We bought at bid_price (we are longing or closing short)
            if self.current_inventory < 0: # Closing short
                step_realized_pnl += (self.avg_entry_price - bid_price) * min(qty, abs(self.current_inventory))
                step_realized_pnl += (bid_price * min(qty, abs(self.current_inventory))) * self.maker_rebate
                
            self.current_inventory += qty
            
            if self.current_inventory > 0 and self.avg_entry_price == 0:
                self.avg_entry_price = bid_price # Opening new long
            elif self.current_inventory > 0: # Adding to long
                pass # simplification for avg price

        self.mid_price = new_mid
        self.realized_pnl += step_realized_pnl
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Normalization of inventory [-1, 1]
        norm_inv = np.clip(self.current_inventory / self.max_inventory, -1.0, 1.0)
        
        # We need realistic standard deviation for reward normalization
        sigma_30s = self.sigma * np.sqrt(30) # arbitrary approximation for testing
        
        # Compute Reward
        reward = self.compute_reward(
            inventory=norm_inv,
            realized_pnl=step_realized_pnl,
            taker_cost=taker_cost,
            fill_occurred=fill_occurred,
            sigma_30s=sigma_30s,
            lambda_base=self.lambda_base,
            kappa=self.kappa,
            rho=self.rho,
            sigma_max_p95=self._sigma_max_p95,
        )
        
        obs = self.observation_space.sample() # Phase 8 connection required
        
        return obs, reward, terminated, truncated, {}

    @staticmethod
    def compute_reward(
        inventory: float,
        realized_pnl: float,
        taker_cost: float,
        fill_occurred: bool,
        sigma_30s: float,
        lambda_base: float = 0.001,
        kappa: float = 0.0007,
        rho: float = 0.0001,
        sigma_max_p95: float = 0.002,
    ) -> float:
        """
        Computes the reward function EXACTLY as specified in Section 2.3.
        inventory is expected to be normalized [-1, 1].
        """
        # Asymmetric quadratic penalty (Crypto funding constraint context)
        if inventory > 0:
            inv_penalty = lambda_base * (inventory ** 2)
        else:
            inv_penalty = 1.5 * lambda_base * (inventory ** 2)
            
        # Quartic scaling for extreme inventory (CRÍTICO)
        if abs(inventory) > 0.8:
            inv_penalty *= (abs(inventory) / 0.8) ** 2
            
        fill_bonus = rho if fill_occurred else 0.0
        raw_reward = realized_pnl - inv_penalty - kappa * taker_cost + fill_bonus
        
        # Normalization by volatility — sigma clipped at p95 to prevent collapse during flash crashes
        # Fixed: dividing by raw sigma_30s collapses reward to ~0 when sigma spikes 10-100x
        sigma_clipped = min(sigma_30s + 1e-8, sigma_max_p95)
        return raw_reward / sigma_clipped

    def update_sigma_max(self, sigma_history: list) -> None:
        """
        Update the p95 sigma clip threshold from a history of realized volatilities.
        Call after every episode to adapt to current market conditions.
        # Fixed: static 0.002 threshold ignores regime changes; p95 adapts correctly
        """
        import numpy as np
        if len(sigma_history) >= 20:
            self._sigma_max_p95 = float(np.percentile(sigma_history, 95))

# --- Design Decisions Note ---
# 1. Action bounds: Used [0, 5] for δ_bid/ask to match 5σ explicitly, and [0, 1] for quantity to represent a percentage of max_qty. This improves neural net exploration vs unbounded outputs.
# 2. Avg Entry Price updating is simplified in the dummy logic. Full VWAP logic applies in execution module later.
# 3. compute_reward is a `@staticmethod` for easier isolated unit testing without instantiating the gym env.
