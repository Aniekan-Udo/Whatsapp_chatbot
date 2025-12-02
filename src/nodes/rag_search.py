import logging
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from rag.knowledge_base import kb

logger = logging.getLogger(__name__)

async def rag_search(state: MessagesState):
    """Perform asynchronous RAG search on knowledge base."""
    
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    args = tool_call.get('args', {})
    
    search_query = args.get('search_query', '')
    
    if not search_query:
        return {
            "messages": [{
                "role": "tool",
                "content": "No search query provided.",
                "tool_call_id": tool_call['id']
            }]
        }
    
    try:
        
        relevant_docs = await kb.asearch(search_query)
        
        if not relevant_docs:
            return {
                "messages": [{
                    "role": "tool",
                    "content": "I couldn't find any information about that in the knowledge base.",
                    "tool_call_id": tool_call['id']
                }]
            }
        
        # Extract page_content from documents and join them
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        return {
            "messages": [{
                "role": "tool",
                "content": context,
                "tool_call_id": tool_call['id']
            }]
        }
    
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return {
            "messages": [{
                "role": "tool",
                "content": "Sorry, I encountered an error searching the knowledge base.",
                "tool_call_id": tool_call['id']
            }]
        }