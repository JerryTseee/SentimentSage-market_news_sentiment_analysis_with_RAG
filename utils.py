"""
Utility functions for the Financial Sentiment Analyzer
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict, Any

def format_sentiment_label(label: str) -> str:
    """Format sentiment label with emoji"""
    emoji_map = {
        'positive': '🟢',
        'negative': '🔴',
        'neutral': '🟡'
    }
    return f"{emoji_map.get(label, '')} {label.title()}"

def calculate_sentiment_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate sentiment metrics from dataframe"""
    total = len(df)
    if total == 0:
        return {}
    
    sentiment_counts = df['label'].value_counts()
    
    return {
        'positive_pct': (sentiment_counts.get('positive', 0) / total) * 100,
        'negative_pct': (sentiment_counts.get('negative', 0) / total) * 100,
        'neutral_pct': (sentiment_counts.get('neutral', 0) / total) * 100,
        'avg_confidence': df['score'].mean(),
        'total_articles': total
    }

def get_investment_signal(positive_ratio: float, negative_ratio: float) -> tuple:
    """Get investment signal based on sentiment ratios"""
    if positive_ratio > 0.6:
        return "bullish", "🟢 Bullish Signal", "success"
    elif negative_ratio > 0.6:
        return "bearish", "🔴 Bearish Signal", "error"
    else:
        return "mixed", "🟡 Mixed Signal", "warning"

def create_advanced_sentiment_chart(df: pd.DataFrame) -> go.Figure:
    """Create an advanced sentiment visualization"""
    sentiment_counts = df['label'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=['#28a745' if x == 'positive' else '#dc3545' if x == 'negative' else '#ffc107' 
                         for x in sentiment_counts.index],
            text=sentiment_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Number of Articles",
        showlegend=False
    )
    
    return fig

def export_results_to_csv(df: pd.DataFrame, company: str) -> str:
    """Export results to CSV format"""
    csv = df.to_csv(index=False)
    return csv

def validate_ticker_symbol(ticker: str) -> bool:
    """Basic validation for ticker symbol"""
    if not ticker:
        return False
    
    # Basic checks
    if len(ticker) < 1 or len(ticker) > 5:
        return False
    
    if not ticker.isalpha():
        return False
    
    return True

@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load sample data for demonstration"""
    sample_data = {
        'text': [
            'Apple reports strong quarterly earnings beating expectations.',
            'Tesla faces production challenges in new factory.',
            'Microsoft announces new AI partnership deal.'
        ],
        'label': ['positive', 'negative', 'positive'],
        'score': [0.89, 0.76, 0.82]
    }
    return pd.DataFrame(sample_data)
