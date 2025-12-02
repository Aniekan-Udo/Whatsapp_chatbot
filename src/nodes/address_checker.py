from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from db_connection import setup_database, store
import asyncio

async def check_address_and_finalize(state: MessagesState, config: RunnableConfig):
    """Check if user has address, ask if not, or prepare for escalation."""
    global store
    if store is None:
        await setup_database()

    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)
    key = "user_memory"
    
    existing_memory = await store.aget(namespace, key)

    has_address = False
    if existing_memory:
        profile_data = existing_memory.value
        has_address = profile_data.get('address') is not None

    tool_calls = state['messages'][-1].tool_calls

    if not has_address:
        return {
            "messages": [{
                "role": "tool",
                "content": (
                    "User is ready to order but no delivery address on file. "
                    "Please ask for their delivery address."
                ),
                "tool_call_id": tool_calls[0]['id']
            }]
        }
    else:
        return {
            "messages": [{
                "role": "tool",
                "content": (
                    "User has address on file and is ready to complete order. "
                    "Escalating to operator."
                ),
                "tool_call_id": tool_calls[0]['id']
            }]
        }