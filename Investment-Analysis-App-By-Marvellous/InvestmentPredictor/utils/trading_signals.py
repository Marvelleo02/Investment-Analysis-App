import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

def calculate_macd_signal(data):
    """Calculate MACD-based trading signals"""
    if len(data) < 26:
        return None

    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    # Generate signals
    buy_signal = macd > signal
    sell_signal = macd < signal

    return {
        'MACD': macd,
        'Signal': signal,
        'Buy': buy_signal,
        'Sell': sell_signal
    }

def calculate_rsi_signal(data, overbought=70, oversold=30):
    """Calculate RSI-based trading signals"""
    if len(data) < 14:
        return None

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    buy_signal = rsi < oversold
    sell_signal = rsi > overbought

    return {
        'RSI': rsi,
        'Buy': buy_signal,
        'Sell': sell_signal
    }

def calculate_bollinger_signal(data, window=20):
    """Calculate Bollinger Bands trading signals"""
    if len(data) < window:
        return None

    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std

    buy_signal = data['Close'] < lower_band
    sell_signal = data['Close'] > upper_band

    return {
        'Upper': upper_band,
        'Lower': lower_band,
        'Middle': sma,
        'Buy': buy_signal,
        'Sell': sell_signal
    }

def calculate_volume_signal(data, window=20):
    """Calculate volume-based trading signals"""
    if len(data) < window:
        return None

    volume_sma = data['Volume'].rolling(window=window).mean()
    volume_ratio = data['Volume'] / volume_sma

    buy_signal = (volume_ratio > 2) & (data['Close'].pct_change() > 0)
    sell_signal = (volume_ratio > 2) & (data['Close'].pct_change() < 0)

    return {
        'VolumeRatio': volume_ratio,
        'Buy': buy_signal,
        'Sell': sell_signal
    }

def generate_trading_signals(data, min_confidence=0.6):
    """Generate combined trading signals with confidence levels"""
    if data is None or data.empty or len(data) < 26:  # Minimum required for MACD
        return None

    try:
        # Calculate individual signals
        macd = calculate_macd_signal(data)
        rsi = calculate_rsi_signal(data)
        bollinger = calculate_bollinger_signal(data)
        volume = calculate_volume_signal(data)

        if not all([macd, rsi, bollinger, volume]):
            return None

        # Combine signals
        buy_signals = pd.DataFrame({
            'MACD': macd['Buy'],
            'RSI': rsi['Buy'],
            'Bollinger': bollinger['Buy'],
            'Volume': volume['Buy']
        })

        sell_signals = pd.DataFrame({
            'MACD': macd['Sell'],
            'RSI': rsi['Sell'],
            'Bollinger': bollinger['Sell'],
            'Volume': volume['Sell']
        })

        # Calculate signal strength
        buy_strength = buy_signals.sum(axis=1) / len(buy_signals.columns)
        sell_strength = sell_signals.sum(axis=1) / len(sell_signals.columns)

        # Generate final signals
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 'HOLD'
        signals['Confidence'] = 0.0

        # Buy signals
        buy_mask = buy_strength >= min_confidence
        signals.loc[buy_mask, 'Signal'] = 'BUY'
        signals.loc[buy_mask, 'Confidence'] = buy_strength[buy_mask]

        # Sell signals
        sell_mask = sell_strength >= min_confidence
        signals.loc[sell_mask, 'Signal'] = 'SELL'
        signals.loc[sell_mask, 'Confidence'] = sell_strength[sell_mask]

        # Add indicator values
        signals['MACD'] = macd['MACD']  # Changed from MACD_Value
        signals['RSI'] = rsi['RSI']
        signals['BB_Position'] = (data['Close'] - bollinger['Lower']) / (bollinger['Upper'] - bollinger['Lower'])
        signals['Volume_Ratio'] = volume['VolumeRatio']

        return signals

    except Exception as e:
        st.error(f"Error generating trading signals: {str(e)}")
        return None

def get_latest_signal(data):
    """Get the most recent trading signal with details"""
    signals = generate_trading_signals(data)
    if signals is None or signals.empty:
        return None

    latest = signals.iloc[-1]
    return {
        'signal': latest['Signal'],
        'confidence': latest['Confidence'],
        'timestamp': signals.index[-1],
        'details': {
            'MACD': latest['MACD'],  # Changed from MACD_Value
            'RSI': latest['RSI'],
            'BB_Position': latest['BB_Position'],
            'Volume_Ratio': latest['Volume_Ratio']
        }
    }

def format_signal_message(signal_data, stock):
    """Format trading signal into a readable message"""
    if not signal_data:
        return f"No trading signals available for {stock}"

    signal = signal_data['signal']
    confidence = signal_data['confidence']
    timestamp = signal_data['timestamp']
    details = signal_data['details']

    message = f"""
    📊 Trading Signal for {stock}

    Signal: {signal} {'🟢' if signal == 'BUY' else '🔴' if signal == 'SELL' else '⚪️'}
    Confidence: {confidence:.1%}
    Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

    Technical Indicators:
    - RSI: {details['RSI']:.2f}
    - MACD: {details['MACD']:.2f}
    - Bollinger Position: {details['BB_Position']:.2f}
    - Volume Ratio: {details['Volume_Ratio']:.2f}
    """

    return message