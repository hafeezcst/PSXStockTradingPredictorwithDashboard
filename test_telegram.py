import requests
import json

def test_telegram_config():
    bot_token = "6860197701:AAESTzERZLYbqyU6gFKfAwJQL8jJ_HNKLbM"
    chat_id = "-4152327824"
    
    # Test 1: Check bot token
    response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe")
    print(f"Bot token valid: {response.status_code == 200}")
    if response.status_code == 200:
        print(f"Bot info: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Check chat access
    response = requests.get(f"https://api.telegram.org/bot{bot_token}/getChat?chat_id={chat_id}")
    print(f"\nChat access valid: {response.status_code == 200}")
    if response.status_code == 200:
        print(f"Chat info: {json.dumps(response.json(), indent=2)}")
    
    # Test 3: Send test message
    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": "Test message from script"}
    )
    print(f"\nMessage sent: {response.status_code == 200}")
    if response.status_code != 200:
        print(f"Error: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    test_telegram_config() 