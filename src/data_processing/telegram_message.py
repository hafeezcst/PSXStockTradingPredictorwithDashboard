import requests
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_telegram_config():
    """Get Telegram configuration from environment variables"""
    try:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logging.error("Telegram configuration not found in environment variables")
            return None, None
            
        return bot_token, chat_id
    except Exception as e:
        logging.error(f"Error getting Telegram configuration: {e}")
        return None, None

def send_telegram_message(message):
    """Send a text message to Telegram"""
    try:
        bot_token, chat_id = get_telegram_config()
        if not bot_token or not chat_id:
            logging.error("Cannot send message: Telegram configuration missing")
            return False
            
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, data=data)
        response.raise_for_status()
        
        logging.info("Message sent successfully to Telegram")
        return True
        
    except Exception as e:
        logging.error(f"Error sending message to Telegram: {e}")
        return False

def send_telegram_message_with_image(image_path, caption=None):
    """Send an image with optional caption to Telegram"""
    try:
        bot_token, chat_id = get_telegram_config()
        if not bot_token or not chat_id:
            logging.error("Cannot send image: Telegram configuration missing")
            return False
            
        if not os.path.exists(image_path):
            logging.error(f"Image file not found: {image_path}")
            return False
            
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        
        with open(image_path, 'rb') as photo:
            files = {
                'photo': photo
            }
            data = {
                'chat_id': chat_id
            }
            if caption:
                data['caption'] = caption
                data['parse_mode'] = 'Markdown'
                
            response = requests.post(url, data=data, files=files)
            response.raise_for_status()
            
        logging.info(f"Image sent successfully to Telegram: {image_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error sending image to Telegram: {e}")
        return False 