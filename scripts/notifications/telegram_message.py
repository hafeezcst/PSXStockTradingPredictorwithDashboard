import requests
import time
import logging
from typing import Dict, Optional
from datetime import datetime
from ..src.psx_predictor.models.signals import (
    generate_buy_signal_description,
    generate_sell_signal_description,
    generate_neutral_signal_description
)

class TelegramMessenger:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for MarkdownV2 format."""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    def send_message(self, message: str) -> bool:
        """Send message to Telegram channel with automatic chunking for long messages."""
        if not self.bot_token or not self.chat_id:
            logging.error("Telegram bot token or chat ID not configured")
            return False

        escaped_message = self._escape_markdown(message)
        max_length = 4096
        messages = [escaped_message[i:i+max_length] for i in range(0, len(escaped_message), max_length)]
        
        success = True
        for chunk in messages:
            try:
                payload = {
                    'chat_id': self.chat_id,
                    'text': chunk,
                    'parse_mode': 'MarkdownV2',
                    'disable_web_page_preview': True
                }
                
                response = requests.post(f"{self.base_url}/sendMessage", json=payload)
                response.raise_for_status()
                
                if len(messages) > 1:
                    time.sleep(1)  # Avoid rate limiting
                    
            except requests.RequestException as e:
                logging.error(f"Error sending message to Telegram: {e}")
                if response:
                    logging.error(f"Response content: {response.text}")
                success = False
                
        if success:
            logging.info("Successfully sent message(s) to Telegram")
        
        return success

    def format_buy_signals(self, data_source_name: str, df, KMI30_symbols: set, dividend_info_func: callable) -> str:
        """Format buy signals for Telegram message."""
        analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"🟢 *{data_source_name} Buy Signals* 📈\n\n"
        
        df = df.sort_values(by='Holding_Days', ascending=True)
        
        for _, row in df.iterrows():
            symbol = row['Stock']
            KMI30_tag = " (KMI30)" if symbol in KMI30_symbols else ""
            
            message += self._format_common_signal_info(row, symbol, KMI30_tag, analysis_time)
            message += self._format_buy_specific_info(row)
            
            dividend_info = dividend_info_func(symbol)
            if dividend_info:
                message += f"\n{self._format_dividend_info(dividend_info)}\n"
            
            message += f"\n🤖 {generate_buy_signal_description(row)}\n\n"
        
        return message

    def format_sell_signals(self, data_source_name: str, df, KMI30_symbols: set, dividend_info_func: callable) -> str:
        """Format sell signals for Telegram message."""
        analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"🔴 *{data_source_name} Sell Signals* 📉\n\n"
        
        df = df.sort_values(by='% P/L', ascending=False)
        
        for _, row in df.iterrows():
            symbol = row['Stock']
            KMI30_tag = " (KMI30)" if symbol in KMI30_symbols else ""
            
            message += self._format_common_signal_info(row, symbol, KMI30_tag, analysis_time)
            message += self._format_sell_specific_info(row)
            
            dividend_info = dividend_info_func(symbol)
            if dividend_info:
                message += f"\n{self._format_dividend_info(dividend_info)}\n"
            
            message += f"\n🤖 {generate_sell_signal_description(row)}\n\n"
        
        return message

    def format_neutral_signals(self, data_source_name: str, df, KMI30_symbols: set, dividend_info_func: callable) -> str:
        """Format neutral signals for Telegram message."""
        analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"⚪️ *{data_source_name} Neutral Signals* 📊\n\n"
        
        df = df.sort_values(by='RSI_Weekly_Avg', ascending=False)
        
        for _, row in df.iterrows():
            symbol = row['Stock']
            KMI30_tag = " (KMI30)" if symbol in KMI30_symbols else ""
            
            message += self._format_common_signal_info(row, symbol, KMI30_tag, analysis_time)
            message += self._format_neutral_specific_info(row)
            
            dividend_info = dividend_info_func(symbol)
            if dividend_info:
                message += f"\n{self._format_dividend_info(dividend_info)}\n"
            
            message += f"\n🤖 {generate_neutral_signal_description(row)}\n\n"
        
        return message

    def _format_common_signal_info(self, row: Dict, symbol: str, KMI30_tag: str, analysis_time: str) -> str:
        """Format common signal information shared between all signal types."""
        return (
            f"🕒 Analysis Time: {analysis_time}\n"
            f"📅 DataBase Update Date: {row['Date']}\n\n"
            f"*{symbol}*{KMI30_tag}\n"
            f"💰 Current Price: {row['Close']:.2f}\n"
            f"📊 RSI: {row['RSI_Weekly_Avg']:.2f}\n"
            f"📈 AO: {row['AO_Weekly']:.2f}\n"
            f"💸 Latest Volume: {row.get('Volume', 0):,.0f}\n"
        )

    def _format_buy_specific_info(self, row: Dict) -> str:
        """Format buy-specific signal information."""
        signal_date = row.get('Signal_Date', 'N/A')
        signal_price = row.get('Signal_Close', 0)
        pl_text = ""
        
        if signal_price and signal_price > 0:
            pl = ((row['Close'] - signal_price) / signal_price * 100)
            pl_text = f"📊 P/L: {pl:+.2f}%\n"
            
        return (
            f"📅 Signal Date: {signal_date}\n"
            f"💵 Signal Price: {signal_price:.2f}\n"
            f"{pl_text}"
            f"⏳ Holding Days: {row.get('Holding_Days', 'N/A')}\n"
        )

    def _format_sell_specific_info(self, row: Dict) -> str:
        """Format sell-specific signal information."""
        return (
            f"📅 Signal Date: {row.get('Signal_Date', 'N/A')}\n"
            f"📊 P/L: {row.get('% P/L', 0):+.2f}%\n"
        )

    def _format_neutral_specific_info(self, row: Dict) -> str:
        """Format neutral-specific signal information."""
        return f"📈 Trend: {row['Trend_Direction']}\n"

    def _format_dividend_info(self, dividend_info: Dict) -> str:
        """Format dividend information."""
        if dividend_info['dividend_amount']:
            div_per_share = dividend_info['face_value'] * dividend_info['dividend_amount']
            div_yield = (div_per_share / dividend_info['last_close'] * 100) if dividend_info['last_close'] else 0
            return (
                f"📅 Book Closure: {dividend_info['bc_to']}\n"
                f"💰 Dividend: Rs. {div_per_share:.2f}/share ({dividend_info['payout_text']})\n"
                f"📈 Yield: {div_yield:.2f}%"
            )
        elif dividend_info['right_amount']:
            return (
                f"📅 Book Closure: {dividend_info['bc_to']}\n"
                f"🔄 Right Share: {dividend_info['payout_text']}"
            )
        return ""
