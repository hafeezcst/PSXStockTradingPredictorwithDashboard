from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Handles technical analysis of stock data using various indicators.
    
    This class performs analysis on stock data using technical indicators like RSI, MACD,
    Bollinger Bands, and moving averages to generate trading signals and scores.
    
    Attributes:
        None
    """
    
    def __init__(self) -> None:
        """Initialize the TechnicalAnalyzer."""
        pass

    def _handle_error(self, error: Exception, context: str, default_return=None):
        """Utility method to handle exceptions with consistent logging.
        
        Args:
            error: The exception object caught.
            context: A string describing the context of the error.
            default_return: The value to return in case of error, if applicable.
        
        Returns:
            The default_return value if provided, otherwise None.
        """
        logger.error(f"Error in {context}: {str(error)}")
        return default_return

    def analyze_stock_indicators(self, stock_data: Dict) -> Dict:
        """Analyze stock data using multiple technical indicators.
        
        Args:
            stock_data (Dict): Dictionary containing stock data with indicators.
            
        Returns:
            Dict: Analysis results including signal type, strength, and scores.
        """
        try:
            symbol = stock_data['symbol']
            logger.info(f"Starting stock analysis for {symbol}")
            
            # Perform technical analysis
            analysis = self._perform_technical_analysis(stock_data)
            logger.info(f"Completed technical analysis for {symbol}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock indicators for {symbol}: {str(e)}")
            return None

    def _perform_technical_analysis(self, stock_data: Dict, previous_analysis: Dict = None) -> Dict:
        """Perform technical analysis on stock data.
        
        Args:
            stock_data (Dict): Dictionary containing stock data with indicators.
            previous_analysis (Dict, optional): Previous analysis data for comparison.
            
        Returns:
            Dict: Analysis results with scores and signals.
        """
        try:
            analysis = {
                'signal_type': 'NEUTRAL',
                'signal_strength': 0.0,
                'confidence_score': 0.0,
                'technical_score': 0.0,
                'trend_score': 0.0,
                'momentum_score': 0.0,
                'volume_score': 0.0,
                'volatility_score': 0.0,
                'support_level': None,
                'resistance_level': None,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward_ratio': None,
                'analysis_summary': [],
                'indicators_used': []
            }
            
            # Add symbol to analysis
            analysis['symbol'] = stock_data['symbol']
            
            # Copy price and indicator data from stock_data to analysis
            price_fields = ['close', 'open', 'high', 'low', 'volume', 'change', 'change_percent']
            indicator_fields = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sma_200', 'bb_upper', 'bb_lower']
            
            for field in price_fields + indicator_fields:
                if field in stock_data:
                    analysis[field] = stock_data[field]
            
            # Perform trend analysis
            self._analyze_trend(stock_data, analysis, previous_analysis)
            
            # Perform momentum analysis
            self._analyze_momentum(stock_data, analysis, previous_analysis)
            
            # Perform volume analysis
            self._analyze_volume(stock_data, analysis, previous_analysis)
            
            # Perform volatility analysis
            self._analyze_volatility(stock_data, analysis, previous_analysis)
            
            # Calculate final scores and determine signal
            self._calculate_final_scores(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing technical analysis: {e}")
            return None

    def _analyze_trend(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze trend indicators.
        
        Args:
            stock_data (Dict): Stock data with indicators.
            analysis (Dict): Current analysis dictionary to update.
            previous_analysis (Dict, optional): Previous analysis for comparison.
        """
        try:
            if all(x is not None for x in [stock_data['close'], stock_data['sma_20'], stock_data['sma_50'], stock_data['sma_200']]):
                analysis['indicators_used'].append('SMA')
                close = stock_data['close']
                sma20 = stock_data['sma_20']
                sma50 = stock_data['sma_50']
                sma200 = stock_data['sma_200']
                
                # Calculate price position relative to SMAs
                price_above_sma20 = (close - sma20) / sma20 * 100
                price_above_sma50 = (close - sma50) / sma50 * 100
                price_above_sma200 = (close - sma200) / sma200 * 100
                
                trend_strength = 0
                
                # Golden Cross (SMA20 crosses above SMA50)
                if sma20 > sma50 and previous_analysis and previous_analysis.get('sma_20', 0) <= previous_analysis.get('sma_50', 0):
                    trend_strength += 15
                    analysis['analysis_summary'].append("Golden Cross detected: SMA20 crossed above SMA50")
                
                # Death Cross (SMA20 crosses below SMA50)
                elif sma20 < sma50 and previous_analysis and previous_analysis.get('sma_20', 0) >= previous_analysis.get('sma_50', 0):
                    trend_strength -= 15
                    analysis['analysis_summary'].append("Death Cross detected: SMA20 crossed below SMA50")
                
                # Strong uptrend conditions
                if close > sma20 > sma50 > sma200:
                    if price_above_sma20 > 5:
                        trend_strength += 25
                        analysis['analysis_summary'].append(f"Strong uptrend: Price {price_above_sma20:.2f}% above SMA20")
                    else:
                        trend_strength += 15
                        analysis['analysis_summary'].append(f"Moderate uptrend: Price {price_above_sma20:.2f}% above SMA20")
                # Strong downtrend conditions
                elif close < sma20 < sma50 < sma200:
                    if price_above_sma20 < -5:
                        trend_strength -= 25
                        analysis['analysis_summary'].append(f"Strong downtrend: Price {abs(price_above_sma20):.2f}% below SMA20")
                    else:
                        trend_strength -= 15
                        analysis['analysis_summary'].append(f"Moderate downtrend: Price {abs(price_above_sma20):.2f}% below SMA20")
                
                analysis['trend_score'] = trend_strength
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")

    def _analyze_momentum(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze momentum indicators.
        
        Args:
            stock_data (Dict): Stock data with indicators.
            analysis (Dict): Current analysis dictionary to update.
            previous_analysis (Dict, optional): Previous analysis for comparison.
        """
        try:
            if all(x is not None for x in [stock_data.get('rsi'), stock_data.get('macd'), stock_data.get('macd_signal'), stock_data.get('ao')]):
                analysis['indicators_used'].extend(['RSI', 'MACD', 'AO'])
                rsi = stock_data['rsi']
                macd = stock_data['macd']
                macd_signal = stock_data['macd_signal']
                ao = stock_data['ao']
                
                momentum_strength = 0
                
                # RSI Analysis
                if rsi < 30:
                    momentum_strength += 15
                    analysis['analysis_summary'].append(f"Strong oversold: RSI at {rsi:.2f}")
                elif rsi < 40:
                    momentum_strength += 10
                    analysis['analysis_summary'].append(f"Moderately oversold: RSI at {rsi:.2f}")
                elif rsi > 70:
                    momentum_strength -= 15
                    analysis['analysis_summary'].append(f"Strong overbought: RSI at {rsi:.2f}")
                elif rsi > 60:
                    momentum_strength -= 10
                    analysis['analysis_summary'].append(f"Moderately overbought: RSI at {rsi:.2f}")
                
                # MACD Analysis
                macd_diff = macd - macd_signal
                macd_diff_percent = (macd_diff / abs(macd_signal)) * 100 if macd_signal != 0 else 0
                
                if previous_analysis:
                    prev_macd = previous_analysis.get('macd', 0)
                    prev_macd_signal = previous_analysis.get('macd_signal', 0)
                    
                    if macd > macd_signal and prev_macd <= prev_macd_signal:
                        momentum_strength += 10
                        analysis['analysis_summary'].append("Bullish MACD crossover detected")
                    elif macd < macd_signal and prev_macd >= prev_macd_signal:
                        momentum_strength -= 10
                        analysis['analysis_summary'].append("Bearish MACD crossover detected")
                
                # Awesome Oscillator Analysis
                ao_abs = abs(ao)
                ao_threshold = 50
                
                if previous_analysis:
                    prev_ao = previous_analysis.get('ao', 0)
                    if ao > 0 and prev_ao <= 0:
                        momentum_strength += 5
                        analysis['analysis_summary'].append("Bullish AO crossover detected")
                    elif ao < 0 and prev_ao >= 0:
                        momentum_strength -= 5
                        analysis['analysis_summary'].append("Bearish AO crossover detected")
                
                if ao > ao_threshold:
                    momentum_strength += 5
                    analysis['analysis_summary'].append(f"Strong bullish AO: {ao:.2f}")
                elif ao > 0:
                    momentum_strength += 2
                    analysis['analysis_summary'].append(f"Moderate bullish AO: {ao:.2f}")
                elif ao < -ao_threshold:
                    momentum_strength -= 5
                    analysis['analysis_summary'].append(f"Strong bearish AO: {ao:.2f}")
                elif ao < 0:
                    momentum_strength -= 2
                    analysis['analysis_summary'].append(f"Moderate bearish AO: {ao:.2f}")
                
                analysis['momentum_score'] = momentum_strength
                
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")

    def _analyze_volume(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze volume indicators.
        
        Args:
            stock_data (Dict): Stock data with indicators.
            analysis (Dict): Current analysis dictionary to update.
            previous_analysis (Dict, optional): Previous analysis for comparison.
        """
        try:
            if all(x is not None for x in [stock_data['volume'], stock_data['change']]):
                analysis['indicators_used'].append('Volume')
                volume = stock_data['volume']
                change = stock_data['change']
                change_percent = stock_data.get('change_percent', 0)
                
                volume_strength = 0
                
                # Volume analysis
                if volume > 2000000:  # High volume threshold
                    if change_percent > 5:
                        volume_strength = 20
                        analysis['analysis_summary'].append(f"Very high volume with strong price increase: {change_percent:.2f}%")
                    elif change_percent > 2:
                        volume_strength = 15
                        analysis['analysis_summary'].append(f"High volume with moderate price increase: {change_percent:.2f}%")
                    elif change_percent < -5:
                        volume_strength = -20
                        analysis['analysis_summary'].append(f"Very high volume with strong price decrease: {abs(change_percent):.2f}%")
                    elif change_percent < -2:
                        volume_strength = -15
                        analysis['analysis_summary'].append(f"High volume with moderate price decrease: {abs(change_percent):.2f}%")
                elif volume > 1000000:  # Moderate volume threshold
                    if change_percent > 2:
                        volume_strength = 10
                        analysis['analysis_summary'].append(f"Moderate volume with price increase: {change_percent:.2f}%")
                    elif change_percent < -2:
                        volume_strength = -10
                        analysis['analysis_summary'].append(f"Moderate volume with price decrease: {abs(change_percent):.2f}%")
                
                # Volume trend analysis
                if previous_analysis:
                    prev_volume = previous_analysis.get('volume', 0)
                    if volume > prev_volume * 1.5:  # 50% volume increase
                        volume_strength += 5
                        analysis['analysis_summary'].append("Significant volume increase detected")
                    elif volume < prev_volume * 0.5:  # 50% volume decrease
                        volume_strength -= 5
                        analysis['analysis_summary'].append("Significant volume decrease detected")
                
                analysis['volume_score'] = volume_strength
                
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")

    def _analyze_volatility(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze volatility indicators.
        
        Args:
            stock_data (Dict): Stock data with indicators.
            analysis (Dict): Current analysis dictionary to update.
            previous_analysis (Dict, optional): Previous analysis for comparison.
        """
        try:
            if all(x is not None for x in [stock_data['bb_upper'], stock_data['bb_lower'], stock_data['close']]):
                analysis['indicators_used'].append('Bollinger Bands')
                bb_upper = stock_data['bb_upper']
                bb_lower = stock_data['bb_lower']
                close = stock_data['close']
                
                bb_range = round(bb_upper - bb_lower, 2)
                volatility = round(bb_range / close * 100, 2)
                price_position = round((close - bb_lower) / bb_range * 100, 2)
                
                volatility_strength = 0
                
                # Volatility analysis
                if volatility > 15:
                    volatility_strength = -20
                    analysis['analysis_summary'].append(f"Very high volatility: BB range {volatility:.2f}%")
                elif volatility > 10:
                    volatility_strength = -15
                    analysis['analysis_summary'].append(f"High volatility: BB range {volatility:.2f}%")
                elif volatility < 5:
                    volatility_strength = 15
                    analysis['analysis_summary'].append(f"Low volatility: BB range {volatility:.2f}%")
                elif volatility < 8:
                    volatility_strength = 10
                    analysis['analysis_summary'].append(f"Moderate volatility: BB range {volatility:.2f}%")
                
                # Calculate support and resistance levels
                analysis['support_level'] = bb_lower
                analysis['resistance_level'] = bb_upper
                
                # Price position relative to BB
                if price_position > 80:
                    analysis['analysis_summary'].append(f"Price near upper BB: {price_position:.2f}% of range")
                    volatility_strength -= 5
                elif price_position < 20:
                    analysis['analysis_summary'].append(f"Price near lower BB: {price_position:.2f}% of range")
                    volatility_strength += 5
                
                analysis['volatility_score'] = volatility_strength
                
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")

    def _calculate_final_scores(self, analysis: Dict):
        """Calculate final scores and determine signal.
        
        Args:
            analysis (Dict): Current analysis dictionary to update with final scores.
        """
        try:
            # Calculate technical score
            analysis['technical_score'] = (
                analysis['trend_score'] +
                analysis['momentum_score'] +
                analysis['volume_score'] +
                analysis['volatility_score']
            )
            
            # Determine signal type and strength
            if analysis['technical_score'] >= 70:
                analysis['signal_type'] = 'STRONG_BUY'
                analysis['signal_strength'] = min(analysis['technical_score'] / 70, 1.0)
            elif analysis['technical_score'] >= 40:
                analysis['signal_type'] = 'BUY'
                analysis['signal_strength'] = min(analysis['technical_score'] / 50, 0.8)
            elif analysis['technical_score'] <= -70:
                analysis['signal_type'] = 'STRONG_SELL'
                analysis['signal_strength'] = min(abs(analysis['technical_score']) / 70, 1.0)
            elif analysis['technical_score'] <= -40:
                analysis['signal_type'] = 'SELL'
                analysis['signal_strength'] = min(abs(analysis['technical_score']) / 50, 0.8)
            else:
                analysis['signal_type'] = 'NEUTRAL'
                analysis['signal_strength'] = 0.5
            
            # Debug logging for required values
            logger.debug(f"Required values for {analysis.get('symbol', 'unknown')}:")
            logger.debug(f"Close price: {analysis.get('close')}")
            logger.debug(f"BB Upper: {analysis.get('bb_upper')}")
            logger.debug(f"BB Lower: {analysis.get('bb_lower')}")
            logger.debug(f"SMA20: {analysis.get('sma_20')}")
            logger.debug(f"Signal Type: {analysis.get('signal_type')}")
            logger.debug(f"Volatility Score: {analysis.get('volatility_score')}")
            
            # Calculate stop loss and take profit levels
            if analysis.get('close') is not None:
                current_price = analysis['close']
                logger.debug(f"Using current price: {current_price}")
                
                # Calculate stop loss based on volatility and support levels
                if analysis.get('bb_lower') is not None and analysis.get('bb_upper') is not None:
                    # Use Bollinger Bands for stop loss
                    stop_loss_long = analysis['bb_lower']
                    stop_loss_short = analysis['bb_upper']
                    logger.debug("Using Bollinger Bands for stop loss calculation")
                elif analysis.get('sma_20') is not None:
                    # Use SMA20 as fallback
                    stop_loss_long = analysis['sma_20'] * 0.95  # 5% below SMA20
                    stop_loss_short = analysis['sma_20'] * 1.05  # 5% above SMA20
                    logger.debug("Using SMA20 as fallback for stop loss calculation")
                else:
                    # Default to percentage-based stop loss
                    stop_loss_long = current_price * 0.95  # 5% below current price
                    stop_loss_short = current_price * 1.05  # 5% above current price
                    logger.debug("Using percentage-based stop loss calculation")
                
                # Calculate take profit based on risk-reward ratio and volatility
                if analysis.get('volatility_score') is not None:
                    # Adjust take profit based on volatility
                    volatility_factor = 1 + (abs(analysis['volatility_score']) / 100)
                    logger.debug(f"Using volatility factor: {volatility_factor}")
                else:
                    volatility_factor = 1.0
                    logger.debug("No volatility score available, using default factor")
                
                # Set take profit levels based on signal type
                if analysis['signal_type'] in ['STRONG_BUY', 'BUY']:
                    analysis['stop_loss'] = stop_loss_long
                    # Take profit at 2:1 risk-reward ratio minimum
                    analysis['take_profit'] = current_price + (2 * (current_price - stop_loss_long)) * volatility_factor
                    logger.debug("Calculated levels for BUY signal")
                elif analysis['signal_type'] in ['STRONG_SELL', 'SELL']:
                    analysis['stop_loss'] = stop_loss_short
                    # Take profit at 2:1 risk-reward ratio minimum
                    analysis['take_profit'] = current_price - (2 * (stop_loss_short - current_price)) * volatility_factor
                    logger.debug("Calculated levels for SELL signal")
                else:
                    # For neutral signals, set both levels but with wider ranges
                    analysis['stop_loss'] = current_price * 0.90  # 10% below
                    analysis['take_profit'] = current_price * 1.10  # 10% above
                    logger.debug("Calculated levels for NEUTRAL signal")
                
                # Calculate risk-reward ratio
                if analysis['signal_type'] in ['STRONG_BUY', 'BUY']:
                    risk = current_price - analysis['stop_loss']
                    reward = analysis['take_profit'] - current_price
                elif analysis['signal_type'] in ['STRONG_SELL', 'SELL']:
                    risk = analysis['stop_loss'] - current_price
                    reward = current_price - analysis['take_profit']
                else:
                    risk = current_price - analysis['stop_loss']
                    reward = analysis['take_profit'] - current_price
                
                if risk != 0:
                    analysis['risk_reward_ratio'] = round(reward / risk, 2)
                    logger.debug(f"Calculated risk-reward ratio: {analysis['risk_reward_ratio']:.2f}")
                else:
                    analysis['risk_reward_ratio'] = 0.00
                    logger.debug("Risk is zero, setting risk-reward ratio to 0.00")
                
                logger.info(f"Calculated trading levels for {analysis.get('symbol', 'unknown')}:")
                logger.info(f"Stop Loss: {analysis['stop_loss']:.2f}")
                logger.info(f"Take Profit: {analysis['take_profit']:.2f}")
                logger.info(f"Risk-Reward Ratio: {analysis['risk_reward_ratio']:.2f}")
                
                # Round calculated values to 2 decimal places
                analysis['stop_loss'] = round(analysis['stop_loss'], 2)
                analysis['take_profit'] = round(analysis['take_profit'], 2)
                analysis['technical_score'] = round(analysis['technical_score'], 2)
                analysis['trend_score'] = round(analysis['trend_score'], 2)
                analysis['momentum_score'] = round(analysis['momentum_score'], 2)
                analysis['volume_score'] = round(analysis['volume_score'], 2)
                analysis['volatility_score'] = round(analysis['volatility_score'], 2)
                analysis['signal_strength'] = round(analysis['signal_strength'], 2)
                analysis['confidence_score'] = round(analysis['confidence_score'], 2)
            else:
                logger.warning(f"Missing close price for {analysis.get('symbol', 'unknown')}, cannot calculate trading levels")
                analysis['stop_loss'] = None
                analysis['take_profit'] = None
                analysis['risk_reward_ratio'] = None
            
        except Exception as e:
            logger.error(f"Error calculating final scores: {e}")
            logger.error(f"Error details: {str(e)}")
            # Set default values in case of error
            analysis['stop_loss'] = None
            analysis['take_profit'] = None
            analysis['risk_reward_ratio'] = None
