#!/usr/bin/env python3
"""Quick run script for the chatbot."""

from src.main import initialize, run_conversation

if __name__ == "__main__":
    graph, observer = initialize()
    
    try:
        run_conversation(graph)
    except KeyboardInterrupt:
        print("\nShutting down...")
        observer.stop()
        observer.join()