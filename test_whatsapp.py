import requests
import json
import time

def test_whatsapp_integration():
    url = "http://localhost:8001/chat"
    
    payload = {
        "message": "Hi, I would like to order some food. This is a test for WhatsApp integration.",
        "business_id": "test_biz",
        "user_id": "test_user_123",
        "live": True
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("="*60)
    print("Testing WhatsApp integration with testing number: +2347046686066")
    print(f"Sending test chat request to {url} with live=True...")
    print("="*60)
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        print("\n✅ API Request Successful!")
        print(f"AI Response: {result.get('response')}")
        print("\n⏳ A test WhatsApp message should now be queued to +2347046686066.")
        print("Note: Make sure your WhatsApp Web is logged in if pywhatkit is used.")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error calling API: {e}")

if __name__ == "__main__":
    test_whatsapp_integration()
