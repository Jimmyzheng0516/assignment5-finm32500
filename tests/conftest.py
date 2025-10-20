import numpy as np, pandas as pd, pytest
from backtester.strategy import VolatilityBreakoutStrategy
from backtester.broker import Broker
from backtester.engine import Backtester


# ===== Price Fixtures =====

@pytest.fixture
def prices():
    """Standard deterministic rising price series"""
    return pd.Series(np.linspace(100, 120, 200))


@pytest.fixture
def constant_prices():
    """Constant price series for testing zero volatility"""
    return pd.Series([100.0] * 100)


@pytest.fixture
def volatile_prices():
    """Volatile price series with random movements"""
    np.random.seed(42)
    returns = np.random.randn(200) * 0.02
    return pd.Series(100 * np.exp(np.cumsum(returns)))


@pytest.fixture
def trending_prices():
    """Strong uptrend price series"""
    return pd.Series(100 * (1.005 ** np.arange(100)))


@pytest.fixture
def short_prices():
    """Very short price series for edge case testing"""
    return pd.Series([100.0, 102.0, 101.0, 103.0, 104.0])


@pytest.fixture
def empty_prices():
    """Empty price series"""
    return pd.Series([], dtype=float)


@pytest.fixture
def prices_with_nans():
    """Price series with NaN values at the beginning"""
    return pd.Series([np.nan, np.nan, np.nan] + list(np.linspace(100, 120, 47)))


@pytest.fixture
def price_with_date_index():
    """Price series with datetime index"""
    date_index = pd.date_range('2020-01-01', periods=100, freq='D')
    return pd.Series(np.linspace(100, 120, 100), index=date_index)


# ===== Strategy Fixtures =====

@pytest.fixture
def strategy():
    """Default volatility breakout strategy"""
    return VolatilityBreakoutStrategy()


@pytest.fixture
def strategy_short_window():
    """Strategy with shorter window for faster signals"""
    return VolatilityBreakoutStrategy(window=10, min_periods=5)


@pytest.fixture
def strategy_long_window():
    """Strategy with longer window for slower signals"""
    return VolatilityBreakoutStrategy(window=50, min_periods=20)


# ===== Broker Fixtures =====

@pytest.fixture
def broker():
    """Default broker with limited cash"""
    return Broker(cash=1_000)


@pytest.fixture
def broker_high_cash():
    """Broker with high cash balance"""
    return Broker(cash=1_000_000)


@pytest.fixture
def broker_low_cash():
    """Broker with very low cash balance"""
    return Broker(cash=100)


@pytest.fixture
def broker_with_position():
    """Broker with existing position"""
    b = Broker(cash=1_000)
    b.position = 10  # Start with 10 shares
    return b


@pytest.fixture
def broker_short_allowed():
    """Broker that allows short selling"""
    return Broker(cash=1_000, allow_short=True)



@pytest.fixture
def backtester(strategy, broker):
    """Default backtester instance"""
    return Backtester(strategy, broker)


@pytest.fixture
def backtester_with_unit_size(strategy_short_window, broker_high_cash):
    """Backtester with larger unit size"""
    return Backtester(strategy_short_window, broker_high_cash, unit_size=10)
