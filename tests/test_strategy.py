import numpy as np
import pandas as pd
import pytest
from backtester.strategy import VolatilityBreakoutStrategy


class TestVolatilityBreakoutStrategyInit:
    """Test strategy initialization"""
    
    def test_default_initialization(self):
        """Test default initialization with default window"""
        strategy = VolatilityBreakoutStrategy()
        assert strategy.window == 20
        assert strategy.min_periods == 20
    
    def test_custom_window(self):
        """Test initialization with custom window"""
        strategy = VolatilityBreakoutStrategy(window=10)
        assert strategy.window == 10
        assert strategy.min_periods == 10
    
    def test_custom_min_periods(self):
        """Test initialization with custom min_periods"""
        strategy = VolatilityBreakoutStrategy(window=20, min_periods=10)
        assert strategy.window == 20
        assert strategy.min_periods == 10
    
    def test_window_type_coercion(self):
        """Test that window is coerced to int"""
        strategy = VolatilityBreakoutStrategy(window=10.5)
        assert strategy.window == 10
        assert isinstance(strategy.window, int)
    
    def test_min_periods_type_coercion(self):
        """Test that min_periods is coerced to int"""
        strategy = VolatilityBreakoutStrategy(window=20, min_periods=15.7)
        assert strategy.min_periods == 15
        assert isinstance(strategy.min_periods, int)


class TestVolatilityBreakoutStrategySignals:
    """Test signal generation"""
    
    def test_signals_length(self, strategy, prices):
        """Test that signals have same length as prices"""
        sig = strategy.signals(prices)
        assert len(sig) == len(prices)
    
    def test_signals_type(self, strategy, prices):
        """Test that signals return a pandas Series"""
        sig = strategy.signals(prices)
        assert isinstance(sig, pd.Series)
    
    def test_signals_dtype(self, strategy, prices):
        """Test that signals are int64 type"""
        sig = strategy.signals(prices)
        assert sig.dtype == np.dtype('int64')
    
    def test_signals_values_range(self, strategy, prices):
        """Test that signals only contain -1, 0, or 1"""
        sig = strategy.signals(prices)
        assert sig.isin([-1, 0, 1]).all()
    
    def test_signals_index_preserved(self, strategy):
        """Test that output index matches input index"""
        custom_index = pd.date_range('2020-01-01', periods=50)
        prices = pd.Series(np.linspace(100, 120, 50), index=custom_index)
        sig = strategy.signals(prices)
        assert sig.index.equals(prices.index)
    
    def test_rising_prices_generates_buy_signals(self):
        """Test that strong upward moves generate buy signals"""
        # Create a price series with a strong upward jump
        prices = pd.Series([100.0] * 25 + [110.0] * 25)
        strategy = VolatilityBreakoutStrategy(window=20, min_periods=10)
        sig = strategy.signals(prices)
        # After the jump, there should be at least one buy signal
        assert (sig == 1).any()
    
    def test_falling_prices_generates_sell_signals(self):
        """Test that strong downward moves generate sell signals"""
        # Create a price series with a strong downward jump
        prices = pd.Series([100.0] * 25 + [90.0] * 25)
        strategy = VolatilityBreakoutStrategy(window=20, min_periods=10)
        sig = strategy.signals(prices)
        # After the drop, there should be at least one sell signal
        assert (sig == -1).any()
    
    def test_first_signal_is_zero(self, strategy, prices):
        """Test that first signal is always 0 (no previous return)"""
        sig = strategy.signals(prices)
        assert sig.iloc[0] == 0
    
    def test_signals_before_min_periods_are_zero(self):
        """Test that signals before min_periods are zero"""
        prices = pd.Series(np.linspace(100, 120, 30))
        strategy = VolatilityBreakoutStrategy(window=20, min_periods=15)
        sig = strategy.signals(prices)
        # First min_periods+1 signals should be 0 (need +1 for the shift and +1 for pct_change)
        assert sig.iloc[:16].eq(0).all()


class TestVolatilityBreakoutStrategyEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_series(self, strategy):
        """Test handling of empty price series"""
        empty_prices = pd.Series([], dtype=float)
        sig = strategy.signals(empty_prices)
        assert isinstance(sig, pd.Series)
        assert len(sig) == 0
        assert sig.dtype == np.dtype('int64')
    
    def test_single_price(self, strategy):
        """Test handling of single price point"""
        prices = pd.Series([100.0])
        sig = strategy.signals(prices)
        assert len(sig) == 1
        assert sig.iloc[0] == 0
    
    def test_two_prices(self):
        """Test handling of two price points"""
        prices = pd.Series([100.0, 105.0])
        strategy = VolatilityBreakoutStrategy(window=5, min_periods=2)
        sig = strategy.signals(prices)
        assert len(sig) == 2
        # Both should be 0 because volatility needs more data
        assert sig.iloc[0] == 0
        assert sig.iloc[1] == 0
    
    def test_constant_prices(self, strategy):
        """Test handling of constant price series"""
        prices = pd.Series([100.0] * 50)
        sig = strategy.signals(prices)
        # All signals should be 0 for constant prices
        assert (sig == 0).all()
    
    def test_prices_with_nans_at_start(self):
        """Test handling of NaNs at the beginning"""
        prices = pd.Series([np.nan, np.nan, np.nan] + list(np.linspace(100, 120, 47)))
        strategy = VolatilityBreakoutStrategy(window=10, min_periods=5)
        sig = strategy.signals(prices)
        assert len(sig) == 50
        # Signals corresponding to NaN prices should be 0
        assert sig.iloc[:3].eq(0).all()
    
    def test_very_short_series_with_large_window(self):
        """Test handling when series is shorter than window"""
        prices = pd.Series([100.0, 101.0, 102.0])
        strategy = VolatilityBreakoutStrategy(window=20, min_periods=2)
        sig = strategy.signals(prices)
        assert len(sig) == 3
        # Should still work with min_periods
        assert sig.dtype == np.dtype('int64')
    
    def test_invalid_input_type_raises_error(self, strategy):
        """Test that non-Series input raises TypeError"""
        with pytest.raises(TypeError, match="prices must be a pandas Series"):
            strategy.signals([100, 101, 102])
    
    def test_invalid_input_type_list_raises_error(self, strategy):
        """Test that list input raises TypeError"""
        with pytest.raises(TypeError, match="prices must be a pandas Series"):
            strategy.signals([100.0, 101.0, 102.0])
    
    def test_invalid_input_type_array_raises_error(self, strategy):
        """Test that numpy array input raises TypeError"""
        with pytest.raises(TypeError, match="prices must be a pandas Series"):
            strategy.signals(np.array([100.0, 101.0, 102.0]))
    
    def test_prices_with_zero_volatility_period(self):
        """Test handling when volatility is zero for some period"""
        # Create prices with constant segment followed by movement
        prices = pd.Series([100.0] * 15 + list(np.linspace(100, 110, 15)))
        strategy = VolatilityBreakoutStrategy(window=10, min_periods=5)
        sig = strategy.signals(prices)
        assert len(sig) == 30
        # Should handle zero volatility gracefully
        assert sig.dtype == np.dtype('int64')


class TestVolatilityBreakoutStrategyLogic:
    """Test the actual volatility breakout logic"""
    
    def test_buy_signal_when_return_exceeds_volatility(self):
        """Test that buy signal is generated when return > volatility"""
        # Create a controlled scenario
        prices = pd.Series([100.0] * 10 + [100.0, 100.0, 100.0, 100.0, 100.0,
                                           100.0, 100.0, 100.0, 100.0, 105.0])
        strategy = VolatilityBreakoutStrategy(window=10, min_periods=5)
        sig = strategy.signals(prices)
        # The large jump at the end should trigger a buy signal
        assert sig.iloc[-1] == 1
    
    def test_sell_signal_when_return_below_negative_volatility(self):
        """Test that sell signal is generated when return < -volatility"""
        # Create a controlled scenario with sudden drop
        prices = pd.Series([100.0] * 10 + [100.0, 100.0, 100.0, 100.0, 100.0,
                                           100.0, 100.0, 100.0, 100.0, 95.0])
        strategy = VolatilityBreakoutStrategy(window=10, min_periods=5)
        sig = strategy.signals(prices)
        # The large drop at the end should trigger a sell signal
        assert sig.iloc[-1] == -1
    
    def test_no_signal_when_return_within_volatility_range(self):
        """Test that no signal when return is within normal volatility"""
        # Create prices with consistent small movements
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(50) * 0.5)
        strategy = VolatilityBreakoutStrategy(window=20, min_periods=10)
        sig = strategy.signals(prices)
        # Most signals should be 0 for small movements
        zero_ratio = (sig == 0).sum() / len(sig)
        assert zero_ratio > 0.5  # At least 50% should be neutral
    
    def test_volatility_calculation_uses_correct_window(self):
        """Test that strategy uses the specified window for volatility"""
        prices = pd.Series(np.linspace(100, 120, 100))
        
        # Test with different windows
        strat_short = VolatilityBreakoutStrategy(window=5, min_periods=3)
        strat_long = VolatilityBreakoutStrategy(window=50, min_periods=10)
        
        sig_short = strat_short.signals(prices)
        sig_long = strat_long.signals(prices)
        
        # Different windows should produce different signals
        # They shouldn't be identical (though this is probabilistic)
        assert not sig_short.equals(sig_long)
    
    def test_min_periods_affects_signal_start(self):
        """Test that min_periods determines when signals can start"""
        prices = pd.Series(np.linspace(100, 120, 30))
        
        strat_high_min = VolatilityBreakoutStrategy(window=20, min_periods=15)
        strat_low_min = VolatilityBreakoutStrategy(window=20, min_periods=5)
        
        sig_high = strat_high_min.signals(prices)
        sig_low = strat_low_min.signals(prices)
        
        # With higher min_periods, more early signals should be 0
        early_zeros_high = (sig_high.iloc[:15] == 0).sum()
        early_zeros_low = (sig_low.iloc[:15] == 0).sum()
        
        assert early_zeros_high >= early_zeros_low


class TestVolatilityBreakoutStrategyIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_realistic_price_series(self):
        """Test with realistic stock-like price movements"""
        np.random.seed(123)
        # Simulate random walk with drift
        returns = np.random.randn(200) * 0.02 + 0.0005
        prices = pd.Series(100 * np.exp(np.cumsum(returns)))
        
        strategy = VolatilityBreakoutStrategy(window=20, min_periods=10)
        sig = strategy.signals(prices)
        
        assert len(sig) == 200
        assert sig.dtype == np.dtype('int64')
        assert sig.isin([-1, 0, 1]).all()
        # Should have some non-zero signals in volatile data
        assert (sig != 0).any()
    
    def test_trending_market(self):
        """Test strategy in trending market"""
        # Strong uptrend
        prices = pd.Series(100 * (1.01 ** np.arange(100)))
        strategy = VolatilityBreakoutStrategy(window=10, min_periods=5)
        sig = strategy.signals(prices)
        
        # In a strong trend, should generate some signals
        assert (sig == 1).any() or (sig == -1).any()
    
    def test_mean_reverting_market(self):
        """Test strategy in mean-reverting market"""
        # Oscillating prices
        prices = pd.Series(100 + 10 * np.sin(np.linspace(0, 4 * np.pi, 100)))
        strategy = VolatilityBreakoutStrategy(window=10, min_periods=5)
        sig = strategy.signals(prices)
        
        # Should generate both buy and sell signals in oscillating market
        assert (sig == 1).any()
        assert (sig == -1).any()
    
    def test_with_custom_date_index(self):
        """Test that strategy works with date index"""
        date_index = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.Series(np.linspace(100, 120, 100), index=date_index)
        
        strategy = VolatilityBreakoutStrategy(window=20, min_periods=10)
        sig = strategy.signals(prices)
        
        assert sig.index.equals(date_index)
        assert len(sig) == 100
    
    def test_signal_consistency(self):
        """Test that signals are deterministic"""
        prices = pd.Series(np.linspace(100, 120, 50))
        strategy = VolatilityBreakoutStrategy(window=10, min_periods=5)
        
        sig1 = strategy.signals(prices)
        sig2 = strategy.signals(prices)
        
        # Same input should always produce same output
        assert sig1.equals(sig2)

