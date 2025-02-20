import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

def calculate_performance_metrics(data):
    """Calculate key performance metrics for the carousel"""
    try:
        daily_returns = data['Close'].pct_change()
        metrics = {
            'Daily Return': daily_returns.iloc[-1],
            'Weekly Return': data['Close'].pct_change(periods=5).iloc[-1],
            'Monthly Return': data['Close'].pct_change(periods=20).iloc[-1],
            'Volatility': daily_returns.std() * np.sqrt(252),
            'Average Volume': data['Volume'].mean()
        }
        return metrics
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None

def create_trend_chart(data, symbol, window=10):
    """Create an animated trend chart"""
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Price',
        line=dict(color='#1f77b4')
    ))

    # Moving average
    ma = data['Close'].rolling(window=window).mean()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ma,
        name=f'{window}-day MA',
        line=dict(color='#ff7f0e', dash='dash')
    ))

    fig.update_layout(
        template='plotly_dark',
        showlegend=True,
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def create_volume_chart(data, symbol):
    """Create an animated volume chart"""
    fig = go.Figure()

    # Volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color='#2ecc71'
    ))

    # Moving average of volume
    vol_ma = data['Volume'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=vol_ma,
        name='Volume MA',
        line=dict(color='#e74c3c')
    ))

    fig.update_layout(
        template='plotly_dark',
        showlegend=True,
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def animate_value(start, end, duration=1000):
    """Animate value changes"""
    import time
    steps = 20
    step_time = duration / steps
    value_step = (end - start) / steps
    
    placeholder = st.empty()
    for i in range(steps + 1):
        current = start + (value_step * i)
        placeholder.metric("Value", f"{current:.2f}")
        time.sleep(step_time / 1000)
    return placeholder

def display_performance_metrics(metrics, symbol):
    """Display metrics with animations"""
    """Display performance metrics with animations"""
    cols = st.columns(3)

    with cols[0]:
        st.metric(
            "Daily Return",
            f"{metrics['Daily Return']:.2%}",
            delta=f"{metrics['Daily Return']:.2%}"
        )

    with cols[1]:
        st.metric(
            "Monthly Return",
            f"{metrics['Monthly Return']:.2%}",
            delta=f"{metrics['Monthly Return']:.2%}"
        )

    with cols[2]:
        st.metric(
            "Volatility",
            f"{metrics['Volatility']:.2%}"
        )

def display_trend_analysis(data, symbol):
    """Display trend analysis charts"""
    st.plotly_chart(
        create_trend_chart(data, symbol),
        use_container_width=True,
        key=f"{symbol}_trend_chart"
    )

def display_volume_analysis(data, symbol):
    """Display volume analysis"""
    st.plotly_chart(
        create_volume_chart(data, symbol),
        use_container_width=True,
        key=f"{symbol}_volume_chart"
    )

def display_carousel(symbol, data):
    """Display animated carousel of stock insights"""
    if data is None or data.empty:
        st.error(f"No data available for {symbol}")
        return

    # Calculate metrics
    metrics = calculate_performance_metrics(data)
    if not metrics:
        return

    # Create container for each section with unique keys
    st.subheader("Performance Overview")
    with st.container():
        display_performance_metrics(metrics, symbol)

    st.subheader("Price Trend Analysis")
    with st.container():
        display_trend_analysis(data, symbol)

    st.subheader("Volume Analysis")
    with st.container():
        display_volume_analysis(data, symbol)