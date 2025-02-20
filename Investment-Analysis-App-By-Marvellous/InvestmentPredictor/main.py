import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import json
import base64
import io

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

from utils.data_processing import get_stock_data, calculate_portfolio_metrics
from utils.prediction import train_prediction_model, predict_future_prices
from utils.storytelling import generate_portfolio_story, generate_action_items
from utils.carousel import display_carousel
from utils.sentiment_analysis import get_news_sentiment, get_market_indicators
from utils.trading_signals import generate_trading_signals, get_latest_signal, format_signal_message


# Page configuration
st.set_page_config(
    page_title="Investment Analysis Tool By Marvellous Idowu",
    layout="wide"
)

# Load custom CSS
try:
    css_path = os.path.join(os.path.dirname(__file__), 'styles', 'custom.css')
    print("Attempting to load CSS from:", css_path)
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception as e:
    print("Error loading CSS:", str(e))
    st.warning("Custom CSS file not found. Using default styles.")

# Initialize session state for watchlists and alerts
if 'watchlists' not in st.session_state:
    st.session_state.watchlists = {'Default': []}
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = {}
if 'custom_indicators' not in st.session_state:
    st.session_state.custom_indicators = {
        'SMA': True,
        'EMA': False,
        'RSI': True,
        'MACD': False,
        'Bollinger': True
    }

# Title
st.title("Investment Analysis Tool By Marvellous Idowu")
st.markdown("---")

# Sidebar
st.sidebar.header("Portfolio Settings")

# Add demo mode toggle
use_demo = st.sidebar.checkbox("Use Demo Data", value=True,
    help="Toggle between demo data and live market data")

if not use_demo:
    st.sidebar.warning("Live data mode may be limited by API rate restrictions")

# Watchlist Management
st.sidebar.subheader("Watchlist Management")
watchlist_name = st.sidebar.selectbox("Select Watchlist", 
                                    list(st.session_state.watchlists.keys()))
new_watchlist_name = st.sidebar.text_input("Create New Watchlist")
if st.sidebar.button("Create Watchlist") and new_watchlist_name:
    st.session_state.watchlists[new_watchlist_name] = []

# Get available stocks
available_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V']
currencies = ['USD', 'EUR', 'GBP', 'JPY']

# User inputs
selected_stocks = st.sidebar.multiselect(
    "Select Stocks",
    available_stocks,
    default=['AAPL', 'MSFT']
)

# Add to Watchlist button
if st.sidebar.button("Add to Watchlist"):
    st.session_state.watchlists[watchlist_name] = list(set(
        st.session_state.watchlists[watchlist_name] + selected_stocks
    ))

# Price Alert System
st.sidebar.subheader("Price Alerts")
alert_stock = st.sidebar.selectbox("Select Stock for Alert", selected_stocks if selected_stocks else [''])
if alert_stock:
    alert_price = st.sidebar.number_input(f"Alert Price for {alert_stock}", min_value=0.0, value=100.0)
    alert_type = st.sidebar.selectbox("Alert Type", ["Above", "Below"])
    if st.sidebar.button("Set Alert"):
        st.session_state.price_alerts[alert_stock] = {
            'price': alert_price,
            'type': alert_type
        }

# Technical Indicator Customization
st.sidebar.subheader("Technical Indicators")
for indicator in st.session_state.custom_indicators:
    st.session_state.custom_indicators[indicator] = st.sidebar.checkbox(
        indicator, 
        value=st.session_state.custom_indicators[indicator]
    )

# Dynamic weight inputs
weights = []
if selected_stocks:
    st.sidebar.subheader("Portfolio Weights")
    remaining_weight = 100
    for i, stock in enumerate(selected_stocks[:-1]):
        weight = st.sidebar.slider(
            f"{stock} Weight (%)",
            0,
            remaining_weight,
            value=100 // len(selected_stocks),
            key=f"weight_{stock}"
        )
        weights.append(weight / 100)
        remaining_weight -= weight

    weights.append(remaining_weight / 100)
    st.sidebar.text(f"{selected_stocks[-1]} Weight: {remaining_weight}%")

investment_amount = st.sidebar.number_input(
    "Investment Amount",
    min_value=1000,
    value=10000,
    key="investment_amount"
)

investment_currency = st.sidebar.selectbox(
    "Investment Currency",
    currencies,
    index=0,
    key="investment_currency"
)

investment_duration = st.sidebar.slider(
    "Investment Duration (Days)",
    min_value=30,
    max_value=365,
    value=180,
    key="investment_duration"
)

# Export Functionality
def export_data(data, format='csv'):
    if format == 'csv':
        return data.to_csv(index=True)
    elif format == 'json':
        return data.to_json(orient='split', date_format='iso')
    return None

# Main content
if selected_stocks and weights:
    try:
        metrics, portfolio_data, full_data = calculate_portfolio_metrics(
            selected_stocks,
            weights,
            investment_amount,
            use_demo=use_demo
        )

        if metrics and portfolio_data is not None:
            tabs = st.tabs([
                "Portfolio Analysis",
                "Technical Analysis",
                "Price Predictions",
                "Your Investment Story",
                "Performance Insights",
                "Stock Comparison",
                "News & Sentiment",
                "Correlation Analysis",
                "Risk Dashboard",
                "Industry Analysis",
                "Watchlists",
                "Trading Signals"  # New tab
            ])

            # Portfolio Analysis Tab
            with tabs[0]:
                st.subheader("Portfolio Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Portfolio Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': metrics.keys(),
                        'Value': [f"{v:.2%}" if k != 'Investment Value' and k != 'Value at Risk (95%)'
                                 else f"${abs(v):,.2f}" for k, v in metrics.items()]
                    })
                    st.table(metrics_df)

                    # Export buttons
                    if st.button("Export Metrics (CSV)"):
                        csv = export_data(metrics_df, 'csv')
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="portfolio_metrics.csv">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)

                with col2:
                    st.subheader("Portfolio Performance")
                    fig = go.Figure()
                    for stock in selected_stocks:
                        if stock in portfolio_data.columns:
                            fig.add_trace(go.Scatter(
                                x=portfolio_data.index,
                                y=portfolio_data[stock],
                                name=stock
                            ))
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Technical Analysis Tab
            with tabs[1]:
                st.subheader("Technical Analysis")
                show_events = st.checkbox("Show Historical Events")

                for stock in selected_stocks:
                    if stock in full_data:
                        stock_data = full_data[stock]

                        # Price with Selected Indicators
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Price'))

                        if st.session_state.custom_indicators['SMA']:
                            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_20'], name='SMA 20'))
                            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], name='SMA 50'))
                            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_200'], name='SMA 200'))

                        if st.session_state.custom_indicators['Bollinger']:
                            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_upper'], name='BB Upper'))
                            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_middle'], name='BB Middle'))
                            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_lower'], name='BB Lower'))

                        if st.session_state.custom_indicators['RSI']:
                            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], name='RSI'))
                            fig1.add_hline(y=70, line_dash="dash", line_color="red")
                            fig1.add_hline(y=30, line_dash="dash", line_color="green")


                        if show_events:
                            # Add example historical events (replace with real data)
                            events = {
                                '2024-01-15': 'Earnings Report',
                                '2024-02-01': 'Product Launch'
                            }
                            for date, event in events.items():
                                fig1.add_annotation(
                                    x=date,
                                    y=stock_data['Close'].max(),
                                    text=event,
                                    showarrow=True,
                                    arrowhead=1
                                )

                        fig1.update_layout(
                            title=f"{stock} - Technical Analysis",
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=400
                        )
                        st.plotly_chart(fig1, use_container_width=True)

            with tabs[2]:
                st.subheader("Price Predictions")
                prediction_results = []
                for stock in selected_stocks:
                    stock_data = full_data.get(stock)
                    if stock_data is not None:
                        model, scaler = train_prediction_model(stock_data)
                        if model and scaler:
                            predictions = predict_future_prices(stock_data, investment_duration, model, scaler)

                            if predictions:
                                # Prediction Chart
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['Close'],
                                    name='Historical'
                                ))

                                future_dates = [stock_data.index[-1] + timedelta(days=x)
                                                for x in range(1, len(predictions) + 1)]
                                fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=predictions,
                                    name='Predicted',
                                    line=dict(dash='dash')
                                ))

                                fig.update_layout(
                                    title=f"{stock} - Price Prediction",
                                    template='plotly_dark',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Calculate prediction metrics
                                current_price = stock_data['Close'].iloc[-1]
                                predicted_price = predictions[-1]
                                predicted_return = (predicted_price - current_price) / current_price
                                predicted_value = investment_amount * weights[selected_stocks.index(stock)] * (
                                            1 + predicted_return)

                                prediction_results.append({
                                    'Stock': stock,
                                    'Current Price': f"${current_price:.2f}",
                                    'Predicted Price': f"${predicted_price:.2f}",
                                    'Predicted Return': f"{predicted_return:.2%}",
                                    'Predicted Value': f"${predicted_value:.2f}",
                                    'Status': '📈 Profit' if predicted_return > 0 else '📉 Loss'
                                })

                # Prediction Summary Table
                if prediction_results:
                    st.subheader("Prediction Summary")
                    pred_df = pd.DataFrame(prediction_results)
                    st.table(pred_df)

                    # Final Portfolio Prediction
                    total_predicted_value = sum(
                        [float(result['Predicted Value'].replace('$', '').replace(',', ''))
                         for result in prediction_results])
                    total_return = (total_predicted_value - investment_amount) / investment_amount

                    st.metric(
                        "Predicted Portfolio Value",
                        f"${total_predicted_value:,.2f}",
                        f"{total_return:+.2%}",
                        delta_color="normal"
                    )

                    st.subheader("Final Analysis")
                    if total_return > 0:
                        st.success(f"📈 Projected Profit: ${total_predicted_value - investment_amount:,.2f}")
                    else:
                        st.error(f"📉 Projected Loss: ${total_predicted_value - investment_amount:,.2f}")

            with tabs[3]:
                st.subheader("📈 Your Investment Story")

                # Generate narrative content
                if prediction_results:
                    story = generate_portfolio_story(
                        metrics, selected_stocks, weights,
                        full_data, prediction_results
                    )
                    actions = generate_action_items(metrics, prediction_results)

                    # Display narrative sections
                    for section in story:
                        with st.expander(f"📊 {section['title']}", expanded=True):
                            st.write(section['content'])

                    # Display action items
                    if actions:
                        st.subheader("💡 Recommended Actions")
                        for action in actions:
                            if action['type'] == 'positive':
                                st.success(f"**{action['title']}**\n\n{action['content']}")
                            elif action['type'] == 'warning':
                                st.warning(f"**{action['title']}**\n\n{action['content']}")
                            else:
                                st.info(f"**{action['title']}**\n\n{action['content']}")

                    # Add interaction hints
                    st.info("""
                        💡 **Tips:**
                        - Expand each section to dive deeper into your portfolio analysis
                        - Review recommended actions for potential portfolio adjustments
                        - Toggle between stocks to see detailed technical insights
                    """)

                if use_demo:
                    st.info("Currently using demo data. Toggle 'Use Demo Data' in the sidebar to switch to live market data.")

            with tabs[4]:
                st.subheader("📊 Performance Insights")
                if selected_stocks:
                    for stock in selected_stocks:
                        if stock in full_data:
                            st.markdown(f"### {stock} Performance Carousel")
                            display_carousel(stock, full_data[stock])
                else:
                    st.warning("Please select stocks to view performance insights.")

            with tabs[5]:
                st.subheader("Stock Performance Comparison")

                if len(selected_stocks) < 2:
                    st.warning("Please select at least two stocks to compare.")
                else:
                    # Normalize prices for comparison
                    comparison_df = pd.DataFrame()
                    base_dates = {}

                    for stock in selected_stocks:
                        if stock in full_data:
                            stock_data = full_data[stock]
                            # Normalize to percentage change from first day
                            first_price = stock_data['Close'].iloc[0]
                            comparison_df[f"{stock}_price"] = (stock_data['Close'] / first_price - 1) * 100
                            comparison_df[f"{stock}_volume"] = stock_data['Volume']
                            comparison_df[f"{stock}_rsi"] = stock_data['RSI']
                            base_dates[stock] = stock_data.index[0]

                    # Price Comparison Chart
                    st.subheader("Relative Price Performance (%)")
                    fig_price = go.Figure()
                    for stock in selected_stocks:
                        if f"{stock}_price" in comparison_df.columns:
                            fig_price.add_trace(go.Scatter(
                                y=comparison_df[f"{stock}_price"],
                                name=f"{stock} ({base_dates[stock].strftime('%Y-%m-%d')})",
                                mode='lines'
                            ))

                    fig_price.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        yaxis_title="Change from Base (%)"
                    )
                    st.plotly_chart(fig_price, use_container_width=True)

                    # Volume Comparison
                    st.subheader("Volume Comparison")
                    fig_volume = go.Figure()
                    for stock in selected_stocks:
                        if f"{stock}_volume" in comparison_df.columns:
                            fig_volume.add_trace(go.Scatter(
                                y=comparison_df[f"{stock}_volume"],
                                name=stock,
                                mode='lines'
                            ))

                    fig_volume.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        yaxis_title="Volume"
                    )
                    st.plotly_chart(fig_volume, use_container_width=True)

                    # RSI Comparison
                    st.subheader("RSI Comparison")
                    fig_rsi = go.Figure()
                    for stock in selected_stocks:
                        if f"{stock}_rsi" in comparison_df.columns:
                            fig_rsi.add_trace(go.Scatter(
                                y=comparison_df[f"{stock}_rsi"],
                                name=stock,
                                mode='lines'
                            ))

                    # Add RSI threshold lines
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

                    fig_rsi.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        yaxis_title="RSI"
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)

                    # Comparison Metrics Table
                    st.subheader("Performance Metrics Comparison")
                    metrics_comparison = {}

                    for stock in selected_stocks:
                        if stock in full_data:
                            stock_data = full_data[stock]
                            returns = stock_data['Close'].pct_change()
                            metrics_comparison[stock] = {
                                'Total Return (%)': f"{((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100:.2f}%",
                                'Volatility (%)': f"{returns.std() * np.sqrt(252) * 100:.2f}%",
                                'Current RSI': f"{stock_data['RSI'].iloc[-1]:.2f}",
                                'Avg Volume': f"{stock_data['Volume'].mean():,.0f}",
                                'Price': f"${stock_data['Close'].iloc[-1]:.2f}"
                            }

                    metrics_df = pd.DataFrame(metrics_comparison).T
                    st.table(metrics_df)

            # News & Sentiment Tab
            with tabs[6]:
                st.subheader("News & Sentiment Analysis")
                for stock in selected_stocks:
                    st.write(f"### {stock} Sentiment Analysis")
                    sentiment_data = get_news_sentiment(stock)

                    if sentiment_data:
                        cols = st.columns(3)
                        cols[0].metric("Sentiment Score", f"{sentiment_data['sentiment_score']:.2f}")
                        cols[1].metric("News Volume", sentiment_data['sentiment_volume'])
                        cols[2].metric("Article Count", sentiment_data['article_count'])

                        st.write("### Latest Headlines")
                        for headline in sentiment_data['latest_headlines']:
                            st.write(f"- {headline}")

            # Correlation Analysis Tab
            with tabs[7]:
                st.subheader("Correlation Analysis")
                if len(selected_stocks) > 1:
                    returns_data = pd.DataFrame()
                    for stock in selected_stocks:
                        if stock in full_data:
                            returns_data[stock] = full_data[stock]['Close'].pct_change()

                    correlation_matrix = returns_data.corr()

                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_matrix,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        colorscale='RdBu'
                    ))

                    fig.update_layout(
                        title="Stock Returns Correlation Matrix",
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least two stocks for correlation analysis")

            # Risk Dashboard Tab
            with tabs[8]:
                st.subheader("Risk Analysis Dashboard")
                for stock in selected_stocks:
                    if stock in full_data:
                        stock_data = full_data[stock]
                        returns = stock_data['Close'].pct_change().dropna()

                        col1, col2, col3 = st.columns(3)

                        beta = returns.cov(returns) / returns.var()
                        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                        var_95 = np.percentile(returns, 5)

                        col1.metric("Beta", f"{beta:.2f}")
                        col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        col3.metric("Value at Risk (95%)", f"{var_95:.2%}")

            # Industry Analysis Tab
            with tabs[9]:
                st.subheader("Industry Analysis")
                industry_data = {
                    'AAPL': 'Technology',
                    'GOOGL': 'Technology',
                    'MSFT': 'Technology',
                    'JPM': 'Financial',
                    'V': 'Financial'
                }

                for stock in selected_stocks:
                    if stock in industry_data:
                        st.write(f"### {stock} - {industry_data[stock]} Sector")
                        # Add industry average comparisons here

            # Watchlists Tab
            with tabs[10]:
                st.subheader("Your Watchlists")
                for watchlist_name, stocks in st.session_state.watchlists.items():
                    with st.expander(f"📋 {watchlist_name}"):
                        if stocks:
                            for stock in stocks:
                                st.write(f"- {stock}")
                        else:
                            st.write("No stocks in this watchlist")

            # Trading Signals Tab
            with tabs[11]:
                st.subheader("📈 Real-Time Trading Signals")

                # Signal confidence threshold
                confidence_threshold = st.slider(
                    "Signal Confidence Threshold",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.6,
                    step=0.1,
                    help="Minimum confidence level required for generating signals"
                )

                # Auto-refresh interval
                auto_refresh = st.checkbox("Auto-refresh signals", value=True)
                if auto_refresh:
                    st.empty()  # Placeholder for refresh

                # Generate and display signals for each stock
                for stock in selected_stocks:
                    if stock in full_data:
                        stock_data = full_data[stock]
                        latest_signal = get_latest_signal(stock_data)

                        with st.expander(f"📊 {stock} Trading Signals", expanded=True):
                            # Display current signal
                            signal_message = format_signal_message(latest_signal, stock)
                            st.markdown(signal_message)

                            # Signal history chart
                            signals = generate_trading_signals(stock_data, min_confidence=confidence_threshold)
                            if signals is not None:
                                fig = go.Figure()

                                # Price line
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['Close'],
                                    name='Price',
                                    line=dict(color='gray', width=1)
                                ))

                                # Buy signals
                                buy_points = signals[signals['Signal'] == 'BUY']
                                fig.add_trace(go.Scatter(
                                    x=buy_points.index,
                                    y=stock_data.loc[buy_points.index, 'Close'],
                                    mode='markers',
                                    name='Buy Signals',
                                    marker=dict(
                                        color='green',
                                        size=10,
                                        symbol='triangle-up'
                                    )
                                ))

                                # Sell signals
                                sell_points = signals[signals['Signal'] == 'SELL']
                                fig.add_trace(go.Scatter(
                                    x=sell_points.index,
                                    y=stock_data.loc[sell_points.index, 'Close'],
                                    mode='markers',
                                    name='Sell Signals',
                                    marker=dict(
                                        color='red',
                                        size=10,
                                        symbol='triangle-down'
                                    )
                                ))

                                fig.update_layout(
                                    title=f"{stock} - Trading Signals",
                                    template='plotly_dark',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Signal statistics
                                total_signals = len(signals[signals['Signal'] != 'HOLD'])
                                buy_signals = len(signals[signals['Signal'] == 'BUY'])
                                sell_signals = len(signals[signals['Signal'] == 'SELL'])

                                col1, col2, col3 = st.columns(3)
                                col1.metric("Total Signals", total_signals)
                                col2.metric("Buy Signals", buy_signals)
                                col3.metric("Sell Signals", sell_signals)

        else:
            st.error("Unable to calculate portfolio metrics. Please try different stocks or check your settings.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please select at least one stock to analyze.")

# Check price alerts
if st.session_state.price_alerts:
    for stock in selected_stocks:
        if stock in st.session_state.price_alerts:
            alert = st.session_state.price_alerts[stock]
            current_price = full_data[stock]['Close'].iloc[-1] if stock in full_data else None

            if current_price:
                if (alert['type'] == 'Above' and current_price > alert['price']) or \
                   (alert['type'] == 'Below' and current_price < alert['price']):
                    st.warning(f"⚠️ Alert: {stock} price {alert['type'].lower()} ${alert['price']:.2f}")