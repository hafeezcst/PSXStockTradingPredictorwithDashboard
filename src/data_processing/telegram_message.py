"""
Telegram messaging module for PSX application.

This module provides functions to send messages and images to a Telegram bot.
"""

import os
import requests
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(filename='telegram_message.log', level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def send_telegram_message(message: str) -> bool:
    """
    Send a text message to a Telegram bot.
    
    Args:
        message (str): Message text to send
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get bot token and chat ID from environment variables
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.warning("Telegram bot token or chat ID not set in environment variables")
            return False
        
        # Construct the API URL
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Prepare the payload
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        # Send the request
        response = requests.post(url, data=payload, timeout=10)
        
        # Check if the request was successful
        if response.status_code == 200:
            logger.info("Message sent successfully")
            return True
        else:
            logger.error(f"Failed to send message: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending Telegram message: {str(e)}")
        return False

def send_telegram_message_with_image(image_path: str, caption: str = "") -> bool:
    """
    Send an image with optional caption to a Telegram bot.
    
    Args:
        image_path (str): Path to the image file
        caption (str): Optional caption text
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get bot token and chat ID from environment variables
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.warning("Telegram bot token or chat ID not set in environment variables")
            return False
        
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            send_telegram_message(f"Error: Image file not found: {image_path}")
            return False
        
        # Construct the API URL
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        
        # Prepare the payload
        payload = {
            'chat_id': chat_id,
            'caption': caption,
            'parse_mode': 'HTML'
        }
        
        # Prepare the files
        files = {
            'photo': open(image_path, 'rb')
        }
        
        # Send the request
        response = requests.post(url, data=payload, files=files, timeout=30)
        
        # Close the file
        files['photo'].close()
        
        # Check if the request was successful
        if response.status_code == 200:
            logger.info(f"Image sent successfully: {image_path}")
            return True
        else:
            logger.error(f"Failed to send image: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending Telegram image: {str(e)}")
        return False

def send_telegram_message_with_image_and_message(image_path: str, message_text: str) -> bool:
    """
    Send both an image and a separate text message to Telegram.
    
    Args:
        image_path (str): Path to the image file
        message_text (str): Text message to send
        
    Returns:
        bool: True if both operations successful, False otherwise
    """
    image_sent = send_telegram_message_with_image(image_path, "")
    message_sent = send_telegram_message(message_text)
    
    return image_sent and message_sent
