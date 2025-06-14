from typing import Dict, List, Optional
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_technical_indicators(data: Dict) -> Dict:
    """Analyze technical indicators for a given stock data"""
    try:
        analysis = {
            'rsi_analysis': analyze_rsi(data.get('rsi')),
            'macd_analysis': analyze_macd(data.get('macd'), data.get('macd_signal'), data.get('macd_hist')),
            'moving_averages': analyze_moving_averages(
                data.get('close'),
                data.get('sma_20'), 
                data.get('sma_50'), 
                data.get('sma_200'),
                data.get('ema_20'),
                data.get('ema_50'),
                data.get('ema_200')
            ),
            'bollinger_bands': analyze_bollinger_bands(
                data.get('close'),
                data.get('bollinger_upper'),
                data.get('bollinger_middle'),
                data.get('bollinger_lower')
            ),
            'stochastic': analyze_stochastic(data.get('stoch_k'), data.get('stoch_d')),
            'ichimoku_cloud': analyze_ichimoku_cloud(
                data.get('close'),
                data.get('ichimoku_tenkan'),
                data.get('ichimoku_kijun'),
                data.get('ichimoku_senkou_span_a'),
                data.get('ichimoku_senkou_span_b'),
                data.get('ichimoku_cloud_green'),
                data.get('ichimoku_cloud_red')
            ),
            'support_resistance': analyze_support_resistance(
                data.get('close'),
                data.get('support_level'),
                data.get('resistance_level')
            ),
            'volume_analysis': analyze_volume(data.get('volume'), data.get('sma_20'))
        }
        
        # Generate overall technical score
        analysis['technical_score'] = calculate_technical_score(analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing technical indicators: {e}")
        return {}

def analyze_rsi(rsi: Optional[float]) -> Dict:
    """Analyze RSI values"""
    try:
        if rsi is None:
            return {'status': 'unknown', 'strength': 0.0}
            
        if rsi > 70:
            return {'status': 'overbought', 'strength': -0.3}
        elif rsi < 30:
            return {'status': 'oversold', 'strength': 0.3}
        elif rsi > 50:
            return {'status': 'bullish', 'strength': 0.1}
        elif rsi < 50:
            return {'status': 'bearish', 'strength': -0.1}
        else:
            return {'status': 'neutral', 'strength': 0.0}
            
    except Exception as e:
        logger.error(f"Error analyzing RSI: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_macd(macd: Optional[float], signal: Optional[float], hist: Optional[float]) -> Dict:
    """Analyze MACD values"""
    try:
        if macd is None or signal is None or hist is None:
            return {'status': 'unknown', 'strength': 0.0}
            
        if macd > signal and hist > 0:
            return {'status': 'bullish_crossover', 'strength': 0.2}
        elif macd < signal and hist < 0:
            return {'status': 'bearish_crossover', 'strength': -0.2}
        elif macd > 0 and signal > 0:
            return {'status': 'bullish_trend', 'strength': 0.1}
        elif macd < 0 and signal < 0:
            return {'status': 'bearish_trend', 'strength': -0.1}
        else:
            return {'status': 'neutral', 'strength': 0.0}
            
    except Exception as e:
        logger.error(f"Error analyzing MACD: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_moving_averages(close: Optional[float], sma_20: Optional[float], sma_50: Optional[float], 
                           sma_200: Optional[float], ema_20: Optional[float], ema_50: Optional[float], 
                           ema_200: Optional[float]) -> Dict:
    """Analyze moving averages"""
    try:
        if any(v is None for v in [close, sma_20, sma_50, sma_200, ema_20, ema_50, ema_200]):
            return {'status': 'unknown', 'strength': 0.0}
            
        # Check for golden cross (bullish)
        if sma_50 > sma_200 and ema_50 > ema_200 and close > sma_50:
            return {'status': 'golden_cross', 'strength': 0.3}
        # Check for death cross (bearish)
        elif sma_50 < sma_200 and ema_50 < ema_200 and close < sma_50:
            return {'status': 'death_cross', 'strength': -0.3}
        # Check if price is above key moving averages (bullish)
        elif close > sma_20 and close > sma_50 and close > sma_200:
            return {'status': 'bullish_trend', 'strength': 0.2}
        # Check if price is below key moving averages (bearish)
        elif close < sma_20 and close < sma_50 and close < sma_200:
            return {'status': 'bearish_trend', 'strength': -0.2}
        else:
            return {'status': 'mixed', 'strength': 0.0}
            
    except Exception as e:
        logger.error(f"Error analyzing moving averages: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_bollinger_bands(close: Optional[float], upper: Optional[float], 
                           middle: Optional[float], lower: Optional[float]) -> Dict:
    """Analyze Bollinger Bands"""
    try:
        if any(v is None for v in [close, upper, middle, lower]):
            return {'status': 'unknown', 'strength': 0.0}
            
        if close > upper:
            return {'status': 'overbought', 'strength': -0.2}
        elif close < lower:
            return {'status': 'oversold', 'strength': 0.2}
        elif close > middle:
            return {'status': 'bullish', 'strength': 0.1}
        elif close < middle:
            return {'status': 'bearish', 'strength': -0.1}
        else:
            return {'status': 'neutral', 'strength': 0.0}
            
    except Exception as e:
        logger.error(f"Error analyzing Bollinger Bands: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_stochastic(k: Optional[float], d: Optional[float]) -> Dict:
    """Analyze Stochastic Oscillator"""
    try:
        if k is None or d is None:
            return {'status': 'unknown', 'strength': 0.0}
            
        if k > 80 and d > 80:
            return {'status': 'overbought', 'strength': -0.2}
        elif k < 20 and d < 20:
            return {'status': 'oversold', 'strength': 0.2}
        elif k > d:
            return {'status': 'bullish_crossover', 'strength': 0.1}
        elif k < d:
            return {'status': 'bearish_crossover', 'strength': -0.1}
        else:
            return {'status': 'neutral', 'strength': 0.0}
            
    except Exception as e:
        logger.error(f"Error analyzing Stochastic Oscillator: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_ichimoku_cloud(close: Optional[float], tenkan: Optional[float], kijun: Optional[float], 
                          span_a: Optional[float], span_b: Optional[float], green: float, red: float) -> Dict:
    """Analyze Ichimoku Cloud"""
    try:
        if any(v is None for v in [close, tenkan, kijun, span_a, span_b]):
            return {'status': 'unknown', 'strength': 0.0}
            
        if green == 1 and close > span_a and close > span_b and close > tenkan and close > kijun:
            return {'status': 'strong_bullish', 'strength': 0.3}
        elif red == 1 and close < span_a and close < span_b and close < tenkan and close < kijun:
            return {'status': 'strong_bearish', 'strength': -0.3}
        elif close > span_a and close > span_b:
            return {'status': 'bullish', 'strength': 0.1}
        elif close < span_a and close < span_b:
            return {'status': 'bearish', 'strength': -0.1}
        else:
            return {'status': 'neutral', 'strength': 0.0}
            
    except Exception as e:
        logger.error(f"Error analyzing Ichimoku Cloud: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_support_resistance(close: Optional[float], support: Optional[float], resistance: Optional[float]) -> Dict:
    """Analyze support and resistance levels"""
    try:
        if close is None or support is None or resistance is None:
            return {'status': 'unknown', 'strength': 0.0}
            
        proximity_to_support = abs(close - support) / close if close != 0 else 0
        proximity_to_resistance = abs(close - resistance) / close if close != 0 else 0
        
        if proximity_to_support < 0.02:  # Within 2% of support
            return {'status': 'near_support', 'strength': 0.2}
        elif proximity_to_resistance < 0.02:  # Within 2% of resistance
            return {'status': 'near_resistance', 'strength': -0.2}
        elif close < support:
            return {'status': 'below_support', 'strength': -0.3}
        elif close > resistance:
            return {'status': 'above_resistance', 'strength': 0.3}
        else:
            return {'status': 'between_levels', 'strength': 0.0}
            
    except Exception as e:
        logger.error(f"Error analyzing support/resistance: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_volume(volume: Optional[float], sma_20: Optional[float]) -> Dict:
    """Analyze volume trends"""
    try:
        if volume is None or sma_20 is None:
            return {'status': 'unknown', 'strength': 0.0}
            
        volume_ratio = volume / sma_20 if sma_20 != 0 else 0
        
        if volume_ratio > 1.5:
            return {'status': 'high_volume', 'strength': 0.2}
        elif volume_ratio < 0.5:
            return {'status': 'low_volume', 'strength': -0.1}
        else:
            return {'status': 'normal_volume', 'strength': 0.0}
            
    except Exception as e:
        logger.error(f"Error analyzing volume: {e}")
        return {'status': 'error', 'strength': 0.0}

def calculate_technical_score(analysis: Dict) -> float:
    """Calculate overall technical score"""
    try:
        score = 0.0
        
        # Sum up strengths from all indicators
        indicators = [
            analysis.get('rsi_analysis', {}).get('strength', 0.0),
            analysis.get('macd_analysis', {}).get('strength', 0.0),
            analysis.get('moving_averages', {}).get('strength', 0.0),
            analysis.get('bollinger_bands', {}).get('strength', 0.0),
            analysis.get('stochastic', {}).get('strength', 0.0),
            analysis.get('ichimoku_cloud', {}).get('strength', 0.0),
            analysis.get('support_resistance', {}).get('strength', 0.0),
            analysis.get('volume_analysis', {}).get('strength', 0.0)
        ]
        
        score = sum(indicators)
        
        # Normalize score to be between -1 and 1
        score = max(min(score, 1.0), -1.0)
        
        return score
        
    except Exception as e:
        logger.error(f"Error calculating technical score: {e}")
        return 0.0
