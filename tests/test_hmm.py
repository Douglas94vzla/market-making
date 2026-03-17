import pytest
import numpy as np
from src.features.hmm_regime import MarketRegimeHMM

@pytest.fixture
def synthetic_data():
    """Generates synthetic price, volume, and spread ticks."""
    np.random.seed(42)
    n_ticks = 500
    
    # Regime 1: Low vol, stable upward drift
    regime1_prices = np.exp(np.cumsum(np.random.normal(0.0001, 0.001, n_ticks // 2))) * 10000.0
    regime1_vols = np.random.uniform(0.1, 1.0, n_ticks // 2)
    regime1_spreads = np.random.uniform(0.5, 2.0, n_ticks // 2)
    
    # Regime 2: High vol, mean reverting chunks
    regime2_prices = np.exp(np.cumsum(np.random.normal(0.0000, 0.005, n_ticks // 2))) * regime1_prices[-1]
    regime2_vols = np.random.uniform(1.0, 5.0, n_ticks // 2)
    regime2_spreads = np.random.uniform(2.0, 10.0, n_ticks // 2)
    
    prices = np.concatenate([regime1_prices, regime2_prices])
    volumes = np.concatenate([regime1_vols, regime2_vols])
    spreads = np.concatenate([regime1_spreads, regime2_spreads])
    
    return prices, volumes, spreads

def test_hmm_initialization():
    hmm_detector = MarketRegimeHMM(n_regimes=4)
    assert hmm_detector.n_regimes == 4
    assert not hmm_detector.is_fitted
    
def test_hmm_feature_preparation(synthetic_data):
    prices, volumes, spreads = synthetic_data
    hmm_detector = MarketRegimeHMM()
    
    features = hmm_detector._prepare_features(prices, volumes, spreads)
    
    assert features.shape == (500, 3) # (n_ticks, log_ret + vol + spread)
    # Ensure first log return is zeroed out safely (no NaNs)
    assert features[0, 0] == 0.0
    assert not np.isnan(features).any()

def test_hmm_train_and_predict(synthetic_data):
    prices, volumes, spreads = synthetic_data
    hmm_detector = MarketRegimeHMM(n_regimes=4, random_state=42)
    
    # Before training it should raise error
    with pytest.raises(RuntimeError):
         hmm_detector.predict_regime_probabilities(prices, volumes, spreads)
         
    # Training
    hmm_detector.train(prices, volumes, spreads)
    assert hmm_detector.is_fitted
    
    # Prediction shape and constraints check
    # We pass the last 100 ticks to estimate the current state
    probabilities = hmm_detector.predict_regime_probabilities(prices[-100:], volumes[-100:], spreads[-100:])
    
    assert probabilities.shape == (4,)
    
    # Test stochastic vector properties:
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    # The sum of probabilities for the hidden states must be exactly 1.0
    assert np.isclose(np.sum(probabilities), 1.0)

# --- Design Decisions Note ---
# 1. Added generated synthetic data mimicking 2 distinct macro-regimes to make the EM algorithm naturally segregate component distributions.
# 2. Asserted the required structural shape: Returns 4-dimensional continuous arrays for Phase 1 Vector Integration.
