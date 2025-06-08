"""
Telegram messaging utilities for PSX dashboard and analysis.
"""
import requests
import logging
import os
import time
from typing import Optional, List

def escape_markdown(text: str) -> str:
    """Escape special characters for Telegram Markdown."""
    for char in ['_', '*', '[', '`']:
        text = text.replace(char, f'\\{char}')
    return text

def send_telegram_message(message: str, config: dict, image_path: Optional[str] = None) -> bool:
    """Send a message or image to Telegram, handling errors and rate limits."""
    if not config.get('telegram', {}).get('enabled', False):
        logging.warning("Telegram messaging is disabled in configuration.")
        return False
    bot_token = config['telegram']['bot_token']
    chat_id = config['telegram']['chat_id']
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto" if image_path else f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {'chat_id': chat_id}
    if message:
        data['caption' if image_path else 'text'] = escape_markdown(message)
        data['parse_mode'] = 'Markdown'
    files = {'photo': open(image_path, 'rb')} if image_path else None
    for attempt in range(3):
        try:
            response = requests.post(url, data=data, files=files)
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 30))
                logging.warning(f"Rate limit hit, waiting {retry_after} seconds")
                time.sleep(retry_after)
            else:
                logging.error(f"Telegram error: {e}")
                break
        except Exception as e:
            logging.error(f"Telegram send failed: {e}")
            break
    return False

def send_telegram_message_with_images(image_paths: List[str], message: str, config: dict) -> None:
    """Send multiple images to Telegram, handling batching and rate limits."""
    max_images = config['telegram'].get('max_images_per_message', 10)
    for i in range(0, len(image_paths), max_images):
        batch = image_paths[i:i+max_images]
        for image_path in batch:
            send_telegram_message(message, config, image_path=image_path)
            time.sleep(1)  # Rate limit between images
        time.sleep(2)  # Delay between batches 