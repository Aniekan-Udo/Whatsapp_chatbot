from langgraph.graph import StateGraph, START, END, MessagesState
from nodes.chatbot import chatbot
from nodes.memory_writer import write_memory
from nodes.rag_search import rag_search
from nodes.address_checker import check_address_and_finalize
from graph.routing import route_customer_action
from db_connection import setup_database,saver
from rag.knowledge_base import kb
import logging

logger = logging.getLogger(__name__)  # ADD THIS

async def build_graph():
    """Build and compile the chatbot graph - async for LangGraph API."""
    
    # Initialize knowledge base (ADD THIS BLOCK)
    try:
        logger.info("Pre-initializing database...")
        await setup_database()
        logger.info("✅ database initialized successfully")

        logger.info("Pre-initializing knowledge base...")
        await kb.ainitialize()
        logger.info("✅ Knowledge base pre-initialized successfully")

    except Exception as e:
        logger.error(f"Failed to pre-initialize knowledge base: {e}")
        # Don't raise - let it initialize lazily on first search if this fails
    
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("chatbot", chatbot)
    builder.add_node("write_memory", write_memory)
    builder.add_node("rag_search", rag_search)
    builder.add_node("check_address_and_finalize", check_address_and_finalize)

    # Add edges
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges(
        "chatbot",
        route_customer_action,
        {
            "write_memory": "write_memory",
            "rag_search": "rag_search",
            "check_address_and_finalize": "check_address_and_finalize",
            "__end__": END
        }
    )

    # Loop back to chatbot
    builder.add_edge("write_memory", "chatbot")
    builder.add_edge("rag_search", "chatbot")
    builder.add_edge("check_address_and_finalize", "chatbot")

    graph = builder.compile(checkpointer=saver)

    return graph