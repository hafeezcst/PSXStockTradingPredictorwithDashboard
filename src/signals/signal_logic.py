"""
Signal and analysis logic for PSX dashboard and analysis.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def analyze_rsi_trend(analysis_df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """Analyze RSI trends for accumulation/distribution."""
    try:
        recent_rsi = analysis_df['RSI_weekly'].tail(20).values
        rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0] if len(recent_rsi) > 1 else 0
        longer_rsi = analysis_df['RSI_weekly'].tail(60).values
        longer_rsi_trend = np.polyfit(range(len(longer_rsi)), longer_rsi, 1)[0] if len(longer_rsi) > 2 else 0
        current_rsi = recent_rsi[-1] if len(recent_rsi) > 0 else 50
        rsi_score = 0
        if current_rsi < 35 and rsi_trend > 0.1:
            rsi_score = 2.5
        elif current_rsi < 40 and rsi_trend > 0.05:
            rsi_score = 2
        elif current_rsi > 65 and rsi_trend < -0.1:
            rsi_score = -2.5
        elif current_rsi > 60 and rsi_trend < -0.05:
            rsi_score = -2
        elif current_rsi < 45 and rsi_trend > 0.03:
            rsi_score = 1
        elif current_rsi > 55 and rsi_trend < -0.03:
            rsi_score = -1
        elif current_rsi < 50 and longer_rsi_trend > 0:
            rsi_score = 0.5
        elif current_rsi > 50 and longer_rsi_trend < 0:
            rsi_score = -0.5
        details = {
            'current_rsi': round(current_rsi, 2),
            'rsi_trend': round(rsi_trend, 4),
            'longer_rsi_trend': round(longer_rsi_trend, 4)
        }
        if len(recent_rsi) > 10 and len(analysis_df['Close'].tail(20)) > 10:
            price_trend = np.polyfit(range(len(analysis_df['Close'].tail(20))), analysis_df['Close'].tail(20).values, 1)[0]
            if price_trend > 0 and rsi_trend < 0:
                rsi_score -= 0.5
                details['divergence'] = 'bearish'
            elif price_trend < 0 and rsi_trend > 0:
                rsi_score += 0.5
                details['divergence'] = 'bullish'
            else:
                details['divergence'] = 'none'
        if len(recent_rsi) > 10:
            rsi_volatility = np.std(recent_rsi)
            details['rsi_volatility'] = round(rsi_volatility, 4)
            if rsi_volatility > 5:
                rsi_score -= 0.3
            elif rsi_volatility < 3:
                rsi_score += 0.3
        return rsi_score, details
    except Exception as e:
        return 0, {'error': str(e)} 