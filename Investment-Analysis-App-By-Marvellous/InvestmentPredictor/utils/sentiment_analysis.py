import requests
import pandas as pd
from textblob import TextBlob
import streamlit as st
from datetime import datetime, timedelta
import os

def get_news_sentiment(symbol, days=30):
    """Analyze news sentiment for a stock using Alpha Vantage News API"""
    try:
        # Use Alpha Vantage API for news (using demo key for now)
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey=demo"
        response = requests.get(url)
        data = response.json()

        if 'feed' not in data:
            return get_demo_sentiment()  # Fallback to demo data if API limit reached

        news_items = data['feed']
        sentiments = []

        for item in news_items:
            # Extract sentiment from title and summary
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)

        if not sentiments:
            return get_demo_sentiment()

        # Calculate aggregated sentiment metrics
        avg_sentiment = sum(sentiments) / len(sentiments)
        sentiment_volume = len(sentiments)

        return {
            'sentiment_score': avg_sentiment,
            'sentiment_volume': sentiment_volume,
            'article_count': len(news_items),
            'latest_headlines': [item['title'] for item in news_items[:3]]
        }
    except Exception as e:
        st.warning(f"Could not fetch sentiment data: {str(e)}")
        return get_demo_sentiment()

def get_demo_sentiment():
    """Generate demo sentiment data when API is unavailable"""
    return {
        'sentiment_score': 0.2,  # Slightly positive
        'sentiment_volume': 25,
        'article_count': 25,
        'latest_headlines': [
            "Company announces strong quarterly results",
            "New product launch receives positive reviews",
            "Industry analysts remain optimistic about growth"
        ]
    }

def get_market_indicators():
    """Get broader market indicators including VIX and sector performance"""
    try:
        # Try to get real VIX data (using demo API)
        vix_url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=VIX&interval=5min&apikey=demo"
        response = requests.get(vix_url)
        data = response.json()

        if 'Time Series (5min)' in data:
            latest_vix = float(next(iter(data['Time Series (5min)'].values()))['4. close'])
        else:
            latest_vix = 20.0  # Default demo value

        # Calculate market momentum and sector performance
        indicators = {
            'vix': latest_vix,
            'market_momentum': calculate_market_momentum(),
            'sector_performance': get_sector_performance()
        }

        return indicators
    except Exception as e:
        st.warning(f"Could not fetch market indicators: {str(e)}")
        return get_demo_indicators()

def calculate_market_momentum():
    """Calculate market momentum indicator"""
    try:
        # Using S&P 500 as market benchmark
        spy_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&apikey=demo"
        response = requests.get(spy_url)
        data = response.json()

        if 'Time Series (Daily)' in data:
            daily_data = data['Time Series (Daily)']
            dates = sorted(daily_data.keys(), reverse=True)[:10]  # Last 10 days
            closes = [float(daily_data[date]['4. close']) for date in dates]

            # Calculate 10-day momentum
            momentum = (closes[0] - closes[-1]) / closes[-1]
            return momentum
        else:
            return 0.01  # Default demo value
    except Exception:
        return 0.01

def get_sector_performance():
    """Get sector performance data"""
    try:
        url = "https://www.alphavantage.co/query?function=SECTOR&apikey=demo"
        response = requests.get(url)
        data = response.json()

        if 'Rank A: Real-Time Performance' in data:
            sector_data = data['Rank A: Real-Time Performance']
            # Average of all sector performances
            performances = [float(perf.strip('%')) for perf in sector_data.values()]
            return sum(performances) / len(performances) / 100
        else:
            return 0.02  # Default demo value
    except Exception:
        return 0.02

def get_demo_indicators():
    """Generate demo market indicators"""
    return {
        'vix': 20.0,
        'market_momentum': 0.01,
        'sector_performance': 0.02
    }