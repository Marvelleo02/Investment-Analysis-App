import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def determine_arima_params(data):
    """Determine optimal ARIMA parameters"""
    try:
        # Check for stationarity
        adf_result = adfuller(data)
        d = 0 if adf_result[1] < 0.05 else 1

        # Simple parameter selection
        p = 1  # Start with simple AR term
        q = 1  # Start with simple MA term

        return p, d, q
    except:
        return 1, 1, 1  # Default parameters if determination fails

def fit_arima(data):
    """Fit ARIMA model with improved error handling"""
    try:
        if data is None or len(data) < 10:
            return None

        # Determine parameters
        p, d, q = determine_arima_params(data)

        # Try fitting the model with determined parameters
        try:
            model = ARIMA(data, order=(p, d, q))
            fitted_model = model.fit()
            return fitted_model
        except:
            # If first attempt fails, try simpler model
            try:
                model = ARIMA(data, order=(1, 1, 0))
                fitted_model = model.fit()
                return fitted_model
            except:
                return None

    except Exception as e:
        st.warning(f"Could not fit ARIMA model: {str(e)}")
        return None

def fit_garch(returns):
    """Fit GARCH model with improved error handling"""
    try:
        if returns is None or len(returns) < 10:
            return None

        model = arch_model(returns, vol='Garch', p=1, q=1)
        fitted_model = model.fit(disp='off', show_warning=False)
        return fitted_model
    except Exception as e:
        st.warning(f"Could not fit GARCH model: {str(e)}")
        return None

def black_scholes_prediction(S, r, sigma, T):
    """Calculate Black-Scholes prediction with validation"""
    try:
        if not all(np.isfinite([S, r, sigma, T])):
            return None
        if S <= 0 or sigma <= 0 or T <= 0:
            return None

        d1 = (np.log(S) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        result = S * norm.cdf(d1)

        return result if np.isfinite(result) and result > 0 else None
    except:
        return None

def train_prediction_model(df):
    """Train prediction model on historical data"""
    if df is None or df.empty:
        return None, None

    try:
        X_scaled, y, scaler = prepare_data(df)
        if X_scaled is None or y is None:
            return None, None

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        return model, scaler
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

def create_features(df, symbol=None):
    """Create enhanced technical indicators for prediction"""
    if df is None or df.empty:
        return None
        
    try:
        from utils.sentiment_analysis import get_news_sentiment, get_market_indicators
        
        # Get sentiment and market data
        sentiment_data = get_news_sentiment(symbol) if symbol else None
        market_data = get_market_indicators()
        
        # Add sentiment and market features if available
        if sentiment_data:
            df['Sentiment_Score'] = sentiment_data['sentiment_score']
            df['News_Volume'] = sentiment_data['sentiment_volume']
            
        if market_data:
            df['VIX'] = market_data['vix']
            df['Market_Momentum'] = market_data['market_momentum']

        # Technical indicators
        window_short = min(20, len(df) // 4)
        window_long = min(50, len(df) // 2)
        df = df.copy()

        # Technical indicators
        df['SMA_20'] = df['Close'].rolling(window=window_short, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=window_long, min_periods=1).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2

        return df
    except Exception as e:
        print(f"Error creating features: {str(e)}")
        return None

def prepare_data(df):
    """Prepare data for prediction models"""
    if df is None or df.empty:
        return None, None, None

    df_features = create_features(df)
    if df_features is None:
        return None, None, None

    df_features = df_features.dropna()
    if df_features.empty:
        return None, None, None

    features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volatility']
    X = df_features[features]
    y = df_features['Close']

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler
    except Exception as e:
        print(f"Error in data preparation: {str(e)}")
        return None, None, None

def predict_future_prices(df, days_ahead, model, scaler):
    """Predict future prices using ensemble of models with improved error handling"""
    if df is None or df.empty:
        st.warning("Insufficient data for predictions")
        return None
        
    # Ensure minimum data points
    min_points = min(50, len(df) - 1)
    if min_points < 10:
        st.warning("Insufficient data points for prediction")
        return None

    try:
        predictions = []
        current_data = df.tail(50).copy()

        # Calculate returns for GARCH
        returns = df['Close'].pct_change().dropna()

        # Fit models with improved error handling
        arima_model = fit_arima(df['Close'])
        if arima_model is None:
            st.info("Using simplified prediction model due to ARIMA fitting issues")

        garch_model = fit_garch(returns)
        if garch_model is None:
            st.info("Using historical volatility due to GARCH fitting issues")

        # Current price and volatility
        current_price = df['Close'].iloc[-1]
        current_vol = returns.std()

        for i in range(days_ahead):
            try:
                # Get base prediction
                features = create_features(current_data)
                if features is None:
                    continue

                feature_names = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volatility']
                last_features = features[feature_names].iloc[-1:]

                if last_features.isnull().any().any():
                    continue

                scaled_features = scaler.transform(last_features)
                base_pred = model.predict(scaled_features)[0]

                if not np.isfinite(base_pred) or base_pred <= 0:
                    continue

                # Get ARIMA prediction with fallback
                if arima_model:
                    try:
                        arima_pred = arima_model.forecast(1)[0]
                        if not np.isfinite(arima_pred) or arima_pred <= 0:
                            arima_pred = base_pred
                    except:
                        arima_pred = base_pred
                else:
                    arima_pred = base_pred

                # Get volatility forecast
                try:
                    if garch_model:
                        forecast = garch_model.forecast(horizon=1)
                        vol_forecast = np.sqrt(forecast.variance.values[-1][0])
                        if not np.isfinite(vol_forecast) or vol_forecast <= 0:
                            vol_forecast = current_vol
                    else:
                        vol_forecast = current_vol
                except:
                    vol_forecast = current_vol

                # Get Black-Scholes prediction
                bs_pred = black_scholes_prediction(current_price, 0.02, vol_forecast, 1/252)
                if bs_pred is None:
                    bs_pred = base_pred

                # Ensemble prediction with validation
                prediction = 0.4 * base_pred + 0.3 * arima_pred + 0.3 * bs_pred

                if np.isfinite(prediction) and prediction > 0:
                    predictions.append(prediction)

                    # Update data for next prediction
                    new_row = current_data.iloc[-1:].copy()
                    new_row.index = [new_row.index[-1] + pd.Timedelta(days=1)]
                    new_row['Close'] = prediction
                    current_data = pd.concat([current_data, new_row])

            except Exception as e:
                continue

        if not predictions:
            st.warning("Could not generate valid predictions")
            return None

        return predictions

    except Exception as e:
        st.error(f"Error in prediction process: {str(e)}")
        return None