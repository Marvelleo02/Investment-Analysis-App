import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import time

def retry_api_call(func, max_retries=3, delay=1):
    """Retry API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** attempt))
            continue
    return None

def get_demo_data(symbol, period='1y'):
    """Generate demo stock data with realistic patterns"""
    try:
        # Generate 2 years of data to ensure sufficient points for technical indicators
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Using business days

        # Ensure consistent random seed for each symbol
        np.random.seed(hash(symbol) % 2**32)

        # Generate base price and trends
        base_price = 100 + np.random.rand() * 200
        trend = np.random.choice([-1, 1]) * np.random.uniform(0.0001, 0.0003)
        volatility = np.random.uniform(0.01, 0.02)

        # Generate more realistic price movements with seasonality
        t = np.linspace(0, len(dates)/252, len(dates))  # Time in years
        seasonal = 0.1 * np.sin(2 * np.pi * t) + 0.05 * np.sin(4 * np.pi * t)  # Annual and semi-annual cycles
        returns = np.random.normal(trend, volatility, len(dates)) + seasonal/252
        price_path = base_price * np.exp(np.cumsum(returns))

        # Generate other price components with realistic spreads
        daily_volatility = np.random.uniform(0.005, 0.015, len(dates))
        high = price_path * (1 + daily_volatility)
        low = price_path * (1 - daily_volatility)
        volume = np.random.lognormal(15, 0.5, len(dates))

        df = pd.DataFrame({
            'Open': price_path,
            'High': high,
            'Low': low,
            'Close': price_path,
            'Volume': volume
        }, index=dates)

        # Sort index to ensure chronological order
        df = df.sort_index()

        # Validate data
        if df['Close'].isna().any():
            st.error(f"Demo data contains NaN values for {symbol}")
            return None

        return df
    except Exception as e:
        st.error(f"Error generating demo data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=300)  # 5 minute cache for live data
def get_stock_data(symbol, period='1y', use_demo=True):
    """Fetch stock data with intelligent caching"""
    cache_key = f"{symbol}_{period}_{use_demo}"
    
    # Check if data exists in cache and is not stale
    cached_data = st.session_state.get(cache_key)
    if cached_data and (datetime.now() - cached_data['timestamp']).seconds < 300:
        return cached_data['data']
    """Fetch stock data with multiple sources and fallback"""
    try:
        if use_demo:
            return get_demo_data(symbol, period)

        # Try primary source (Yahoo Finance)
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if not df.empty and len(df) >= 50:  # Ensure sufficient data points
                df = df.sort_index()
                return df
        except Exception as e:
            st.warning(f"Could not fetch live data for {symbol} from primary source: {str(e)}")

        # Try secondary source (Alpha Vantage)
        try:
            import requests
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey=demo"
            r = requests.get(url)
            data = r.json()

            if 'Time Series (Daily)' in data:
                df = pd.DataFrame(data['Time Series (Daily)']).T
                df.index = pd.to_datetime(df.index)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df.astype(float)
                if len(df) >= 50:
                    return df.sort_index()
        except Exception as e:
            st.warning(f"Could not fetch live data for {symbol} from secondary source: {str(e)}")

        # Fallback to demo data with warning
        st.warning(f"Using demo data as fallback for {symbol} (API limits or connectivity issues)")
        return get_demo_data(symbol, period)

    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    try:
        if df is None or df.empty or len(df) < 50:
            return None

        df = df.copy()

        # Ensure data is sorted
        df = df.sort_index()

        # Calculate Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200, min_periods=200).mean()

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20, min_periods=20).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Remove any remaining NaN values
        df = df.dropna()

        if len(df) < 50:
            st.error("Insufficient data points after calculating indicators")
            return None

        return df

    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def calculate_portfolio_metrics(symbols, weights, investment_amount, use_demo=True):
    """Calculate comprehensive portfolio metrics"""
    try:
        # Validate inputs
        if not symbols or not weights or len(symbols) != len(weights):
            st.error("Invalid portfolio parameters")
            return None, None, None

        if not np.isclose(sum(weights), 1.0, rtol=1e-5):
            st.error("Portfolio weights must sum to 1.0")
            return None, None, None

        portfolio_data = {}
        full_data = {}

        # Fetch and validate data for all symbols
        for symbol in symbols:
            df = get_stock_data(symbol, use_demo=use_demo)
            if df is not None and not df.empty:
                portfolio_data[symbol] = df['Close']
                full_data[symbol] = calculate_technical_indicators(df)
            else:
                st.error(f"Could not process data for {symbol}")
                return None, None, None

        # Create portfolio DataFrame
        portfolio = pd.DataFrame(portfolio_data)

        # Validate portfolio data
        if portfolio.empty:
            st.error("No valid data available for portfolio construction")
            return None, None, None

        # Calculate returns with explicit handling of missing values
        returns = portfolio.pct_change()
        returns = returns.dropna()

        if returns.empty:
            st.error("Unable to calculate returns from the data")
            return None, None, None

        # Calculate portfolio metrics
        weights = np.array(weights)
        daily_returns = returns.dot(weights)

        if len(daily_returns) == 0:
            st.error("No valid returns available for calculation")
            return None, None, None

        portfolio_return = np.mean(daily_returns) * 252
        portfolio_std = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std != 0 else 0

        # Calculate additional metrics
        max_drawdown = calculate_max_drawdown(daily_returns)
        value_at_risk = calculate_value_at_risk(daily_returns, investment_amount)

        metrics = {
            'Expected Annual Return': portfolio_return,
            'Annual Volatility': portfolio_std,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Value at Risk (95%)': value_at_risk,
            'Investment Value': investment_amount * (1 + portfolio_return)
        }

        return metrics, portfolio, full_data

    except Exception as e:
        st.error(f"Error calculating portfolio metrics: {str(e)}")
        return None, None, None

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    try:
        if len(returns) == 0:
            return 0

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return float(drawdowns.min())
    except Exception as e:
        st.error(f"Error calculating max drawdown: {str(e)}")
        return 0

def calculate_value_at_risk(returns, investment):
    """Calculate Value at Risk using historical method"""
    try:
        if len(returns) == 0:
            return 0

        return float(np.percentile(returns, 5) * investment)
    except Exception as e:
        st.error(f"Error calculating VaR: {str(e)}")
        return 0

def convert_currency(amount, from_currency, to_currency):
    """Simple currency conversion (placeholder)"""
    rates = {
        'USD': 1.0,
        'EUR': 0.85,
        'GBP': 0.73,
        'JPY': 110.0
    }

    if from_currency not in rates or to_currency not in rates:
        return amount

    usd_amount = amount / rates[from_currency]
    target_amount = usd_amount * rates[to_currency]
    return target_amount