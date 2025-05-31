import os
import requests
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Telegram configuration from environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram_message(message):
    """Send a text message to Telegram"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logging.warning("Telegram configuration not found. Message not sent.")
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=data)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")
        return False

def send_telegram_message_with_image(image_path, caption=None):
    """Send an image with optional caption to Telegram"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logging.warning("Telegram configuration not found. Image not sent.")
            return False
            
        if not os.path.exists(image_path):
            logging.error(f"Image file not found: {image_path}")
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {
            'photo': open(image_path, 'rb')
        }
        data = {
            'chat_id': TELEGRAM_CHAT_ID
        }
        if caption:
            data['caption'] = caption
            data['parse_mode'] = 'Markdown'
            
        response = requests.post(url, files=files, data=data)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Error sending Telegram image: {e}")
        return False 