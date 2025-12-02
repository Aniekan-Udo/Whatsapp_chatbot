import logging
from config import settings
from graph.builder import build_graph

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# For LangGraph API: export the factory function directly
# LangGraph will call this function when it needs to create a graph instance
graph = build_graph