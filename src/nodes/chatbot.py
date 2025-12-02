from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from llm.chat_model import get_chat_model
from pydantic_models import CustomerAction
from prompt import MSG_PROMPT
from db_connection import setup_database, saver, store

model = get_chat_model()

async def chatbot(state: MessagesState, config: RunnableConfig):
    """Load memory from the store and use it to personalize the chatbot's response."""
    
    # Access store from config - LangGraph provides this
    global store
    if store is None or saver is None:
        await setup_database()

    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)
    key = "user_memory"

    # Use store directly - it's already in the correct context
    existing_memory = await store.aget(namespace, key)

    if existing_memory and isinstance(existing_memory.value, dict):
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    system_msg = MSG_PROMPT.format(user_profile=existing_memory_content)

    response = await model.bind_tools([CustomerAction]).ainvoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    return {"messages": [response]}