import requests
import logging
import os
from datetime import datetime
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_telegram_config():
    """Get Telegram configuration from environment variables or config file"""
    try:
        # First try environment variables
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # If env vars are not available, try config file
        if not bot_token or not chat_id:
            try:
                import json
                config_path = Path(__file__).parent.parent / 'config' / 'telegram_config.json'
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if not bot_token:
                    bot_token = config.get('bot_token')
                if not chat_id:
                    chat_id = config.get('chat_id')
                    
            except Exception as config_error:
                logging.warning(f"Could not read config file: {config_error}")
        
        if not bot_token:
            logging.error("Bot token not found in environment variables or config file")
            return None, None
            
        if not chat_id:
            logging.error("Chat ID not found in environment variables or config file")
            logging.error("Please set TELEGRAM_CHAT_ID in your .env file or telegram_config.json")
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
        
        # Validate message
        if not message or not message.strip():
            logging.error("Cannot send empty message")
            return False
            
        # Check message length (Telegram limit is 4096 characters)
        if len(message) > 4096:
            logging.warning(f"Message too long ({len(message)} chars), truncating to 4096")
            message = message[:4093] + "..."
            
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Try with different parse modes if Markdown fails
        for parse_mode in ["Markdown", "HTML", None]:
            try:
                data = {
                    "chat_id": chat_id,
                    "text": message
                }
                if parse_mode:
                    data["parse_mode"] = parse_mode
                
                response = requests.post(url, data=data, timeout=30)
                response.raise_for_status()
                
                logging.info(f"Message sent successfully to Telegram using {parse_mode or 'no'} parse mode")
                return True
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    logging.warning(f"Parse mode {parse_mode} failed with 400 error, trying next...")
                    if parse_mode is None:  # Last attempt failed
                        # Try with plain text and character sanitization
                        safe_message = ''.join(c for c in message if ord(c) < 128 and c.isprintable())
                        if len(safe_message) > 4096:
                            safe_message = safe_message[:4093] + "..."
                        
                        final_data = {
                            "chat_id": chat_id,
                            "text": safe_message
                        }
                        
                        final_response = requests.post(url, data=final_data, timeout=30)
                        final_response.raise_for_status()
                        logging.info("Message sent successfully using sanitized plain text")
                        return True
                else:
                    raise e
                    
    except Exception as e:
        logging.error(f"Error sending message to Telegram: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logging.error(f"Response details: {e.response.text}")
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
        
        # Validate caption length (Telegram limit is 1024 characters for captions)
        if caption and len(caption) > 1024:
            logging.warning(f"Caption too long ({len(caption)} chars), truncating to 1024")
            caption = caption[:1021] + "..."
        
        with open(image_path, 'rb') as photo:
            files = {
                'photo': photo
            }
            data = {
                'chat_id': chat_id
            }
            if caption:
                # Try with different parse modes if Markdown fails
                for parse_mode in ["Markdown", "HTML", None]:
                    try:
                        test_data = data.copy()
                        test_data['caption'] = caption
                        if parse_mode:
                            test_data['parse_mode'] = parse_mode
                        
                        response = requests.post(url, data=test_data, files={'photo': photo}, timeout=60)
                        response.raise_for_status()
                        
                        logging.info(f"Image sent successfully to Telegram: {image_path}")
                        return True
                        
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 400 and parse_mode is not None:
                            logging.warning(f"Caption parse mode {parse_mode} failed, trying next...")
                            photo.seek(0)  # Reset file pointer
                            continue
                        else:
                            raise e
            else:
                # No caption, send image only
                response = requests.post(url, data=data, files=files, timeout=60)
                response.raise_for_status()
                
                logging.info(f"Image sent successfully to Telegram: {image_path}")
                return True
        
    except Exception as e:
        logging.error(f"Error sending image to Telegram: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logging.error(f"Response details: {e.response.text}")
        return False 