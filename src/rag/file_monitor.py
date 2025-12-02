import os
import logging
import asyncio # New import for running async method
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import settings
from rag.knowledge_base import kb

logger = logging.getLogger(__name__)

class KBUpdateHandler(FileSystemEventHandler):
    """Handle file system events for knowledge base updates."""
    
    def on_modified(self, event):
        # We only care about the source document file itself
        if event.src_path == os.path.abspath(settings.kb_doc_path):
            logger.info(f"📁 Detected change in file: {event.src_path}")
            try:
                asyncio.run(kb.arefresh())
            except Exception as e:
                # The traceback should be logged if the async operation fails
                logger.error(f"Error refreshing knowledge base asynchronously: {e}", exc_info=True)

def start_file_monitoring():
    """Start watching file for changes."""
    event_handler = KBUpdateHandler()
    observer = Observer()
    
    # Get the directory of the knowledge base document
    kb_dir = os.path.dirname(os.path.abspath(settings.kb_doc_path))
    observer.schedule(event_handler, kb_dir, recursive=False)
    
    observer.start()
    logger.info(f"File monitoring started on directory: {kb_dir}")
    # Note: The calling script must keep the main thread alive for the observer to continue running.
    return observer
