from langchain_groq import ChatGroq
from trustcall import create_extractor
from config import settings
from pydantic_models import Profile


def get_chat_model():
    """Get the configured chat model."""
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.API_KEY
    )

def get_profile_extractor():
    """Get the profile extractor."""
    model = get_chat_model()
    return create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )