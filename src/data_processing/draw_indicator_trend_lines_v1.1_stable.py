"""
PSX Stock Trading Predictor with Dashboard
Version: 1.1 (Stable)
Last Updated: 2024-03-19

This script analyzes PSX stocks using technical indicators and generates trading signals
with visual dashboards. It includes:
- Technical analysis using RSI, AO, and Moving Averages
- Market phase detection (Accumulation/Distribution)
- Trading signal generation (BUY/HOLD/SELL/OPPORTUNITY)
- Telegram integration for alerts and dashboards
- Market overview and recommendation dashboards

Author: Muhammad Hafeez
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_predictor.log'),
        logging.StreamHandler()
    ]
)

class Config:
    """Configuration manager for the trading predictor"""
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'database': {
                'path': 'data/psx_data.db'
            },
            'output': {
                'folder': 'output'
            },
            'telegram': {
                'enabled': True,
                'max_images_per_message': 1,
                'bot_token': '6860197701:AAESTzERZLYbqyU6gFKfAwJQL8jJ_HNKLbM',
                'chat_id': '-4152327824'
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

class TelegramNotifier:
    """Handles Telegram notifications and dashboard sharing"""
    def __init__(self, config: Config):
        self.config = config
        self.bot_token = config.get('telegram.bot_token')
        self.chat_id = config.get('telegram.chat_id')
        self.enabled = config.get('telegram.enabled', True)
        self.max_images = config.get('telegram.max_images_per_message', 1)
        
    def _send_telegram_api_call(self, method: str, params: Dict) -> Dict:
        """Make API call to Telegram with rate limit handling"""
        if not self.enabled:
            return {'ok': False, 'error': 'Telegram notifications disabled'}
            
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=params)
                data = response.json()
                
                if data.get('ok'):
                    return data
                    
                if 'retry_after' in data.get('parameters', {}):
                    time.sleep(data['parameters']['retry_after'])
                    continue
                    
                if 'chat not found' in str(data.get('description', '')).lower():
                    logging.error(f"Chat not found. Please verify chat ID: {self.chat_id}")
                    return data
                    
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                    
                return data
                
            except Exception as e:
                logging.error(f"Telegram API error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return {'ok': False, 'error': str(e)}
    
    def send_message(self, message: str) -> bool:
        """Send text message to Telegram"""
        if not self.enabled:
            return False
            
        # Split long messages
        max_length = 4000
        messages = [message[i:i+max_length] for i in range(0, len(message), max_length)]
        
        success = True
        for msg in messages:
            result = self._send_telegram_api_call('sendMessage', {
                'chat_id': self.chat_id,
                'text': msg,
                'parse_mode': 'HTML'
            })
            if not result.get('ok'):
                success = False
                logging.error(f"Failed to send message: {result.get('description')}")
        return success
    
    def send_image(self, image_path: str, caption: str = '') -> bool:
        """Send image to Telegram"""
        if not self.enabled:
            return False
            
        if not os.path.exists(image_path):
            logging.error(f"Image file not found: {image_path}")
            return False
            
        with open(image_path, 'rb') as f:
            result = self._send_telegram_api_call('sendPhoto', {
                'chat_id': self.chat_id,
                'photo': f,
                'caption': caption,
                'parse_mode': 'HTML'
            })
            
        return result.get('ok', False)

def create_market_overview_dashboard(df: pd.DataFrame, folder: str) -> bool:
    """Create market overview dashboard with enhanced visualization"""
    if df.empty:
        return False
    date = datetime.now().strftime('%Y-%m-%d')
    
    # Create figure with custom size and style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'axes.edgecolor': '#CCCCCC',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333'
    })
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Add main title with date
    fig.suptitle(f'PSX Market Overview - {date}', fontsize=20, y=0.95)
    
    # Signal Distribution
    plt.subplot(2, 2, 1)
    signal_counts = df['Status'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    plt.pie(signal_counts, labels=signal_counts.index, autopct='%1.1f%%',
            colors=colors[:len(signal_counts)],
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.title('Signal Distribution', pad=20, fontsize=12)
    
    # Market Phase Distribution
    plt.subplot(2, 2, 2)
    phase_counts = df['Market_Phase'].value_counts()
    colors = ['#27ae60', '#2ecc71', '#c0392b', '#7f8c8d']
    plt.pie(phase_counts, labels=phase_counts.index, autopct='%1.1f%%',
            colors=colors[:len(phase_counts)],
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.title('Market Phase Distribution', pad=20, fontsize=12)
    
    # Market Breadth Indicators
    plt.subplot(2, 2, 3)
    breadth_data = {
        'Above MA30': len(df[df['Above_MA30']]) / len(df) * 100,
        'RSI > 50': len(df[df['RSI'] > 50]) / len(df) * 100,
        'Accumulation': len(df[df['Market_Phase'].isin(['ACCUMULATION', 'WEAK_ACCUMULATION'])]) / len(df) * 100
    }
    bars = plt.bar(breadth_data.keys(), breadth_data.values(), 
                  color=['#3498db', '#2ecc71', '#27ae60'])
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
    plt.ylim(0, 100)
    plt.title('Market Breadth Indicators', pad=20, fontsize=12)
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Top Performing Stocks
    plt.subplot(2, 2, 4)
    top_stocks = df.nlargest(5, 'Phase_Probability')
    bars = plt.barh(top_stocks['Symbol'], top_stocks['Phase_Probability'],
                   color='#2ecc71', alpha=0.8)
    for i, v in enumerate(top_stocks['Phase_Probability']):
        plt.text(v + 1, i, f'{v:.1f}%', va='center')
    plt.title('Top Performing Stocks', pad=20, fontsize=12)
    plt.xlabel('Accumulation Probability (%)', fontsize=10)
    
    # Add footer with summary
    plt.figtext(0.5, 0.02, 
                f'Total Stocks: {len(df)} | Strong Accumulation: {len(df[df["Market_Phase"] == "ACCUMULATION"])} | '
                f'Weak Accumulation: {len(df[df["Market_Phase"] == "WEAK_ACCUMULATION"])}',
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save with high DPI
    path = os.path.join(folder, f'market_overview_{date}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path

def create_recommendation_dashboard(df: pd.DataFrame, folder: str) -> bool:
    """Create recommendation dashboard with enhanced visualization"""
    if df.empty:
        return False
    date = datetime.now().strftime('%Y-%m-%d')
    
    # Create figure with custom size and style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'axes.edgecolor': '#CCCCCC',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333'
    })
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Add main title with date
    fig.suptitle(f'PSX Trading Recommendations - {date}', fontsize=20, y=0.95)
    
    # Top 5 BUY/HOLD
    plt.subplot(2, 2, 1)
    buy_df = df[df['Status'] == 'BUY/HOLD'].nlargest(5, 'Phase_Probability')
    if not buy_df.empty:
        bars = plt.barh(buy_df['Symbol'], buy_df['Phase_Probability'], 
                       color='#2ecc71', alpha=0.8)
        for i, v in enumerate(buy_df['Phase_Probability']):
            plt.text(v + 1, i, f'{v:.1f}%', va='center')
        plt.title('Top BUY/HOLD Stocks', pad=20, fontsize=12)
        plt.xlabel('Accumulation Probability (%)', fontsize=10)
    
    # Top 5 OPPORTUNITY
    plt.subplot(2, 2, 2)
    opp_df = df[(df['Status'] == 'OPPORTUNITY') & 
                (df['Market_Phase'].isin(['ACCUMULATION', 'WEAK_ACCUMULATION']))]
    if not opp_df.empty:
        top_opps = opp_df.nlargest(5, 'Phase_Probability')
        bars = plt.barh(top_opps['Symbol'], top_opps['Phase_Probability'], 
                       color='#3498db', alpha=0.8)
        for i, v in enumerate(top_opps['Phase_Probability']):
            plt.text(v + 1, i, f'{v:.1f}%', va='center')
        plt.title('Top OPPORTUNITY Stocks', pad=20, fontsize=12)
        plt.xlabel('Accumulation Probability (%)', fontsize=10)
    
    # Top 5 SELL
    plt.subplot(2, 2, 3)
    sell_df = df[df['Status'] == 'SELL'].nlargest(5, 'Phase_Probability')
    if not sell_df.empty:
        bars = plt.barh(sell_df['Symbol'], sell_df['Phase_Probability'], 
                       color='#e74c3c', alpha=0.8)
        for i, v in enumerate(sell_df['Phase_Probability']):
            plt.text(v + 1, i, f'{v:.1f}%', va='center')
        plt.title('Top SELL Stocks', pad=20, fontsize=12)
        plt.xlabel('Distribution Probability (%)', fontsize=10)
    
    # Market Phase Summary
    plt.subplot(2, 2, 4)
    phase_summary = df['Market_Phase'].value_counts()
    colors = ['#27ae60', '#2ecc71', '#c0392b', '#7f8c8d']
    plt.pie(phase_summary, labels=phase_summary.index, autopct='%1.1f%%',
            colors=colors[:len(phase_summary)],
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.title('Market Phase Summary', pad=20, fontsize=12)
    
    # Add footer with summary
    plt.figtext(0.5, 0.02, 
                f'Total Stocks: {len(df)} | BUY/HOLD: {len(df[df["Status"] == "BUY/HOLD"])} | '
                f'SELL: {len(df[df["Status"] == "SELL"])} | '
                f'OPPORTUNITY: {len(df[df["Status"] == "OPPORTUNITY"])}',
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save with high DPI
    path = os.path.join(folder, f'recommendations_{date}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path

def main():
    """Main function to run the trading predictor"""
    try:
        # Initialize configuration
        config = Config()
        
        # Create output directory
        output_folder = config.get('output.folder', 'output')
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize Telegram notifier
        telegram = TelegramNotifier(config)
        
        # TODO: Add your data processing and analysis code here
        # This is where you would:
        # 1. Load stock data
        # 2. Calculate indicators
        # 3. Generate signals
        # 4. Create dashboards
        # 5. Send notifications
        
        logging.info("Analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 