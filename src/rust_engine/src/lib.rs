use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::BTreeMap;

/// FastOrderBook replacing Python dictionaries with BTreeMap.
/// Provides O(log N) insertion and fast iterations natively.
#[pyclass]
pub struct FastOrderBook {
    /// Bids stored with negative keys so that standard iteration yields highest to lowest
    bids: BTreeMap<i64, f64>,
    /// Asks stored with positive keys, lowest to highest
    asks: BTreeMap<i64, f64>,
}

#[pymethods]
impl FastOrderBook {
    #[new]
    pub fn new() -> Self {
        FastOrderBook {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    /// Process snapshot mapping Python tuples directly. Notice prices are translated to i64 limits
    /// using a fixed precision factor for BTreeMap Hash ordering (e.g. price * 100_000).
    pub fn apply_snapshot(&mut self, bids_py: &PyList, asks_py: &PyList) -> PyResult<()> {
        self.bids.clear();
        self.asks.clear();

        for item in bids_py {
            let tuple: (f64, f64) = item.extract()?;
            let price_key = -(tuple.0 * 100_000.0) as i64;
            self.bids.insert(price_key, tuple.1);
        }

        for item in asks_py {
            let tuple: (f64, f64) = item.extract()?;
            let price_key = (tuple.0 * 100_000.0) as i64;
            self.asks.insert(price_key, tuple.1);
        }

        Ok(())
    }

    /// O(log N) operations for delta updates
    pub fn apply_delta(&mut self, is_bid: bool, price: f64, qty: f64) {
        if is_bid {
            let price_key = -(price * 100_000.0) as i64;
            if qty == 0.0 {
                self.bids.remove(&price_key);
            } else {
                self.bids.insert(price_key, qty);
            }
        } else {
            let price_key = (price * 100_000.0) as i64;
            if qty == 0.0 {
                self.asks.remove(&price_key);
            } else {
                self.asks.insert(price_key, qty);
            }
        }
    }

    /// Computes the 47-dimensional State Vector defined in Gym Observation exactly.
    /// This iteration runs purely in machine code avoiding Python loop overheads.
    pub fn compute_state(&self) -> Vec<f64> {
        let mut features = Vec::with_capacity(47);

        // 1. Get Top 10 Bids and Top 10 Asks
        let mut top_bids_prices = Vec::new();
        let mut top_bids_vols = Vec::new();
        for (k, v) in self.bids.iter().take(10) {
            top_bids_prices.push((-(*k) as f64) / 100_000.0);
            top_bids_vols.push(*v);
        }
        
        let mut top_asks_prices = Vec::new();
        let mut top_asks_vols = Vec::new();
        for (k, v) in self.asks.iter().take(10) {
            top_asks_prices.push((*k as f64) / 100_000.0);
            top_asks_vols.push(*v);
        }
        
        // Pad with zeros if Book is extremely empty
        while top_bids_prices.len() < 10 {
            top_bids_prices.push(0.0);
            top_bids_vols.push(0.0);
        }
        while top_asks_prices.len() < 10 {
            top_asks_prices.push(0.0);
            top_asks_vols.push(0.0);
        }

        // Fill 40 features (10 bid prices, 10 bid vols, 10 ask prices, 10 ask vols)
        features.extend(top_bids_prices.iter());
        features.extend(top_bids_vols.iter());
        features.extend(top_asks_prices.iter());
        features.extend(top_asks_vols.iter());

        // 2. Compute Mid Price and Spread (Feature 41, 42)
        let best_bid = top_bids_prices[0];
        let best_ask = top_asks_prices[0];
        
        // Safety Fallback for empty book scenario
        let mid_price = if best_bid > 0.0 && best_ask > 0.0 {
            (best_ask + best_bid) / 2.0
        } else {
            0.0
        };
        let spread = if best_ask > 0.0 && best_bid > 0.0 {
            best_ask - best_bid
        } else {
            0.0
        };

        features.push(mid_price);
        features.push(spread);

        // 3. Compute VWAP for Bids and Asks (Feature 43, 44)
        let total_bid_vol: f64 = top_bids_vols.iter().sum();
        let vw_bid = if total_bid_vol > 0.0 {
            top_bids_prices.iter().zip(top_bids_vols.iter()).map(|(p, v)| p * v).sum::<f64>() / total_bid_vol
        } else {
            0.0
        };

        let total_ask_vol: f64 = top_asks_vols.iter().sum();
        let vw_ask = if total_ask_vol > 0.0 {
            top_asks_prices.iter().zip(top_asks_vols.iter()).map(|(p, v)| p * v).sum::<f64>() / total_ask_vol
        } else {
            0.0
        };

        features.push(vw_bid);
        features.push(vw_ask);

        // 4. Compute Volume Imbalance (Feature 45)
        let imbalance = if (total_bid_vol + total_ask_vol) > 0.0 {
            (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        } else {
            0.0
        };
        features.push(imbalance);

        // Feature 46, 47 are reserved for external inputs (HMM regime_trend, current_inventory)
        // Set to 0.0 allowing Agent to inject them in the wrapper externally.
        features.push(0.0);
        features.push(0.0);

        features
    }
}

/// A Python module mapped natively to Rust API via PyO3.
#[pymodule]
fn rust_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FastOrderBook>()?;
    Ok(())
}
