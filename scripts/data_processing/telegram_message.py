import os
import logging
import requests
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def send_telegram_message(message: str, chat_id: Optional[str] = None) -> bool:
    """
    Send a message to a Telegram chat using the bot API.
    
    Args:
        message (str): The message to send
        chat_id (str, optional): The chat ID to send the message to. If not provided,
                               will use the default chat ID from environment variables.
    
    Returns:
        bool: True if the message was sent successfully, False otherwise
    """
    try:
        # Get Telegram bot token from environment variables
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            logging.error("TELEGRAM_BOT_TOKEN not found in environment variables")
            return False
        
        # Use provided chat_id or get from environment variables
        if not chat_id:
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            if not chat_id:
                logging.error("TELEGRAM_CHAT_ID not found in environment variables")
                return False
        
        # Construct the API URL
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Prepare the payload
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        # Send the request
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        logging.info("Telegram message sent successfully")
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending Telegram message: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error sending Telegram message: {str(e)}")
        return False 