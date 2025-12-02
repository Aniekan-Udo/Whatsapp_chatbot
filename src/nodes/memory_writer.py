import uuid
import asyncio
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, merge_message_runs
from llm.chat_model import get_profile_extractor
from db_connection import setup_database, store
from prompt import TRUSTCALL_INSTRUCTION

profile_extractor = get_profile_extractor()

async def write_memory(state: MessagesState, config: RunnableConfig):
    """Extract and save profile information using an asynchronous Postgres store."""
    
    global store
    if store is None:
        await setup_database()

    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)
    
    existing_items = await store.asearch(namespace) 
    
    tool_name = "Profile"
    existing_memories = (
        [(item.key, tool_name, item.value) for item in existing_items]
        if existing_items else None
    )
    
    conversation_history = [
        msg for msg in state["messages"]
        if not (hasattr(msg, 'type') and msg.type == 'tool')
    ]
    
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        memory=existing_memories
    )
    updated_messages = list(merge_message_runs(
        [SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + conversation_history
    ))
    
    result = await profile_extractor.ainvoke({ 
        "messages": updated_messages,
        "existing": existing_memories
    })
    
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        await store.aput(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )
    
    tool_calls = state['messages'][-1].tool_calls
    
    return {
        "messages": [{
            "role": "tool",
            "content": "Profile information saved successfully",
            "tool_call_id": tool_calls[0]['id']
        }]
    }