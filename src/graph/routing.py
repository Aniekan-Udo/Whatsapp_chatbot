from langgraph.graph import MessagesState

def route_customer_action(state: MessagesState) -> str:
    """Route to the appropriate action node based on priority."""
    last_message = state["messages"][-1]
    
    if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
        return "__end__"
    
    tool_call = last_message.tool_calls[0]
    args = tool_call.get('args', {})
    
    if args.get('ready_to_order'):
        return "check_address_and_finalize"
    
    if args.get('update_profile'):
        return "write_memory"
    
    if args.get('search_menu'):
        return "rag_search"
    
    return "__end__"