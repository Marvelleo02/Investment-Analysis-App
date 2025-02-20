import pandas as pd
import numpy as np
from datetime import datetime

def generate_portfolio_story(metrics, stocks, weights, full_data, prediction_results):
    """Generate a personalized narrative about the portfolio"""
    if not metrics or not stocks or not weights or not full_data:
        return [{
            'section': 'Data Notice',
            'title': 'Portfolio Analysis',
            'content': 'Unable to generate complete analysis due to insufficient data. Please ensure all required data is available.'
        }]
    story = []
    
    # Portfolio Overview
    total_stocks = len(stocks)
    top_stock = max(zip(stocks, weights), key=lambda x: x[1])
    
    story.append({
        'section': 'Portfolio Overview',
        'title': 'Your Investment Strategy',
        'content': f"""Your diversified portfolio consists of {total_stocks} carefully selected stocks, 
        with {top_stock[0]} being your highest conviction pick at {top_stock[1]*100:.1f}% allocation."""
    })
    
    # Performance Story
    annual_return = metrics.get('Expected Annual Return', 0)
    volatility = metrics.get('Annual Volatility', 0)
    sharpe = metrics.get('Sharpe Ratio', 0)
    
    performance_narrative = f"""Your portfolio has demonstrated {
        'strong' if annual_return > 0.15 else 'moderate' if annual_return > 0 else 'challenging'
    } performance with an expected annual return of {annual_return:.1%}. """
    
    risk_narrative = f"""The portfolio's volatility is {
        'high' if volatility > 0.25 else 'moderate' if volatility > 0.15 else 'low'
    } at {volatility:.1%}, resulting in a Sharpe ratio of {sharpe:.2f}, which indicates {
        'excellent' if sharpe > 2 else 'good' if sharpe > 1 else 'moderate' if sharpe > 0 else 'challenging'
    } risk-adjusted returns."""
    
    story.append({
        'section': 'Performance Analysis',
        'title': 'Performance and Risk Profile',
        'content': performance_narrative + risk_narrative
    })
    
    # Technical Analysis Insights
    tech_insights = []
    for stock in stocks:
        if stock in full_data:
            data = full_data[stock]
            latest_rsi = data['RSI'].iloc[-1]
            latest_price = data['Close'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            
            rsi_status = (
                'potentially overbought' if latest_rsi > 70
                else 'potentially oversold' if latest_rsi < 30
                else 'neutral'
            )
            
            trend_status = (
                'above' if latest_price > sma_50
                else 'below'
            )
            
            tech_insights.append(
                f"{stock} is currently {rsi_status} with RSI at {latest_rsi:.0f}, "
                f"and trading {trend_status} its 50-day moving average."
            )
    
    story.append({
        'section': 'Technical Insights',
        'title': 'Technical Analysis Summary',
        'content': ' '.join(tech_insights)
    })
    
    # Future Outlook
    if prediction_results:
        positive_predictions = sum(1 for result in prediction_results 
                                 if float(result['Predicted Return'].strip('%'))/100 > 0)
        total_predictions = len(prediction_results)
        
        outlook = (
            'strongly positive' if positive_predictions == total_predictions
            else 'generally positive' if positive_predictions > total_predictions/2
            else 'mixed' if positive_predictions == total_predictions/2
            else 'challenging'
        )
        
        story.append({
            'section': 'Future Outlook',
            'title': 'Forward-Looking Analysis',
            'content': f"""Based on our predictive models, the outlook for your portfolio appears {outlook}, 
            with {positive_predictions} out of {total_predictions} stocks showing potential upside."""
        })
    
    return story

def generate_action_items(metrics, prediction_results):
    """Generate actionable insights based on portfolio analysis"""
    actions = []
    
    # Risk-based actions
    sharpe = metrics.get('Sharpe Ratio', 0)
    if sharpe < 1:
        actions.append({
            'type': 'warning',
            'title': 'Risk Management',
            'content': 'Consider rebalancing your portfolio to improve risk-adjusted returns.'
        })
    
    # Prediction-based actions
    if prediction_results:
        for result in prediction_results:
            return_pct = float(result['Predicted Return'].strip('%'))/100
            if abs(return_pct) > 0.1:  # 10% threshold
                action_type = 'positive' if return_pct > 0 else 'warning'
                actions.append({
                    'type': action_type,
                    'title': f'Price Movement Alert - {result["Stock"]}',
                    'content': f"""Significant {'upside' if return_pct > 0 else 'downside'} 
                    potential detected for {result['Stock']}. Consider {'increasing' if return_pct > 0 else 'reducing'} 
                    position if aligned with your strategy."""
                })
    
    return actions
