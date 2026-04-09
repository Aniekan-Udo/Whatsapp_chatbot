import os
import httpx
from monitoring import logger

WAHA_URL = os.getenv("WAHA_URL", "http://localhost:3000")

async def enqueue_whatsapp_message(phone_or_chat_id: str, message: str):
    """
    Send a message via WAHA API.
    Accepts either:
      - A phone number with country code, e.g. '+2349159633734'
      - A pre-formed WAHA chat_id, e.g. '128205595897860@c.us' (LID)
    """
    # If it's already a chat_id (contains @), use it directly
    if "@" in phone_or_chat_id:
        chat_id = phone_or_chat_id
    else:
        clean_no = phone_or_chat_id.lstrip("+")
        chat_id = f"{clean_no}@c.us"

    payload = {
        "chatId": chat_id,
        "text": message,
        "session": "default"
    }

    logger.info("sending_waha_message", chat_id=chat_id)

    try:
        async with httpx.AsyncClient() as client:
            headers = {"X-Api-Key": "123123"}
            response = await client.post(f"{WAHA_URL}/api/sendText", json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
            logger.info("waha_message_sent", chat_id=chat_id)
            # WAHA sometimes returns an empty body on success — handle gracefully
            try:
                return response.json()
            except Exception:
                return {"status": "sent", "raw": response.text}
    except Exception as e:
        logger.error("waha_send_failed", chat_id=chat_id, error=str(e))
        raise

