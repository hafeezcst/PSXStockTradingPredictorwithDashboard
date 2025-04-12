import logging
import telebot
import time

# Configure your Telegram bot credentials
TELEGRAM_TOKEN = '6860197701:AAESTzERZLYbqyU6gFKfAwJQL8jJ_HNKLbM'
CHAT_ID = '-4152327824'

# Initialize the bot
bot = telebot.TeleBot(TELEGRAM_TOKEN)

def send_telegram_message(message):
    """Send a text-only message to Telegram with rate limit handling"""
    max_retries = 3
    retry_delay = 5
    
    # Split long messages into chunks of max 3500 characters
    if len(message) > 3500:
        chunks = split_long_message(message)
        for i, chunk in enumerate(chunks):
            prefix = f"Part {i+1}/{len(chunks)}: " if len(chunks) > 1 else ""
            send_with_retry(prefix + chunk, max_retries, retry_delay)
    else:
        send_with_retry(message, max_retries, retry_delay)

def send_with_retry(message, max_retries, delay):
    """Send message with retry logic for rate limits"""
    for attempt in range(max_retries):
        try:
            bot.send_message(chat_id=CHAT_ID, text=message)
            logging.info(f"Telegram message sent successfully")
            return True
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:  # Rate limit error
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to send Telegram message: {e}")
                return False

def split_long_message(message):
    """Split a long message into smaller chunks that fit in Telegram"""
    # Find a good splitting point - headers and main content
    parts = message.split("\n\n", 1)
    header = parts[0]
    
    if len(parts) == 1:  # No clear separator found
        return [message[:3500], message[3500:]] if len(message) > 3500 else [message]
    
    # Process the signals table
    table_content = parts[1]
    table_rows = table_content.split("\n")
    
    # Keep header rows
    chunks = []
    current_chunk = header + "\n\n" + table_rows[0] + "\n" + table_rows[1] + "\n"
    
    # Process data rows
    for row in table_rows[2:]:
        if len(current_chunk) + len(row) + 1 > 3500:  # +1 for newline
            chunks.append(current_chunk)
            current_chunk = header + "\n\n" + table_rows[0] + "\n" + table_rows[1] + "\n" + row + "\n"
        else:
            current_chunk += row + "\n"
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def send_telegram_message_with_image(image_path, message):
    """Send a message with an image to Telegram"""
    try:
        # Your existing code...
        bot.send_photo(
            chat_id=CHAT_ID,
            photo=open(image_path, 'rb'),
            caption=message
        )
        return True
    except Exception as e:
        logging.error(f"Error sending Telegram message with image: {e}")
        # Define a base delay value for exponential backoff (in seconds)
        delay = 1  # Start with 1 second delay
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logging.info(f"Retrying in {wait_time} seconds (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
                
                # Retry sending the message
                bot.send_photo(
                    chat_id=CHAT_ID,
                    photo=open(image_path, 'rb'),
                    caption=message
                )
                return True
            except Exception as retry_e:
                logging.error(f"Retry attempt {attempt+1} failed: {retry_e}")
        
        return False

def send_telegram_message_with_image_and_message(image_path, message_text):
    """Send an image with a separate text message to Telegram"""
    # First send the image
    send_telegram_message_with_image(image_path, "")
    
    # Then send the text message separately
    send_telegram_message(message_text)
