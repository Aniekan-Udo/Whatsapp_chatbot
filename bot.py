import sys
import asyncio
import asyncpg

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Also set it on the current running loop if one exists
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running yet, create one with the right policy
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

import uuid
from typing import List, Optional, Literal, Dict, Any
import asyncio
import pandas as pd
import httpx
import os
import json
import hashlib
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt

from langchain_community.document_loaders import CSVLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

from trustcall import create_extractor
from langchain_groq import ChatGroq

from typing import List, Union, Optional
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

import os
import logging
import random
from typing import List, Union, Optional
from pydantic import BaseModel, Field
from typing import Any, Callable, Tuple
from functools import wraps
from openai import OpenAI, APIError, RateLimitError, Timeout
# Move these imports BEFORE get_pool() function
from tenacity import (
    retry, retry_if_exception_type, 
    stop_after_delay, stop_after_attempt,
    wait_combine, wait_exponential, wait_random
)
import requests
import aiohttp, urllib3,openai,langchain_core

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please check your .env file")

import uuid
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()

try:
    model = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY)
except Exception as e:
    raise ValueError(f"Failed to initialize ChatGroq model. Check your API key: {e}")



MSG_PROMPT= """
You are a whatsapp assistant, your duty is to assist business owners attend to customers.

Your primary responsibilities:
- Provide excellent customer service with a warm, conversational tone
- Help customers find information about menu items, pricing, and availability
- Remember customer preferences and delivery addresses for personalized service
- Guide customers through the complete order process including address collection
- Handle common questions efficiently while knowing when to escalate complex issues
- Add items to cart and always ask customer if they want to add to their order.
- When customers are ready to pay or complete their order, check for their address and confirm if they want pick-up or delivery.
- If user wants delivery, go through prior conversations to check for address, if no address provided during conversation, ask user for delivery address 

Customer Profile:
<user_profile>
{user_profile}
</user_profile>
"""

TRUSTCALL_INSTRUCTION = """You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, address, location, cart)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
   - Delivery address
   - Ordered items
3. Merge any new information with existing memory
4. Format the memory as a clear, structured profile
5. If new information conflicts with existing memory, keep the most recent version

CRITICAL: 
- Use null (not the string "None") for fields with no information
- Only include factual information directly stated by the user
- Do not make assumptions or inferences
- Leave fields as null if not mentioned in the conversation

Based on the chat history below, please update the user information:"""

class Profile(BaseModel):
    """User profile information for personalizing customer service.
    
    IMPORTANT: Use null for unknown fields, not the string "None".
    Only populate fields with actual information from the conversation.
    """
    
    name: Optional[str] = Field(
        default=None, 
        description="Customer's name. Use null if not mentioned."
    )
    location: Optional[str] = Field(
        default=None, 
        description="Customer's general location or city. Use null if not mentioned."
    )
    address: Optional[str] = Field(
        default=None, 
        description="Customer's full delivery address. Use null if not mentioned."
    )
    cart: Optional[List[str]] = Field(
        default=None, 
        description="List of items in customer's order. Use null if no items ordered yet."
    )
    
    human_active: Optional[bool] = Field(
        default=None, 
        description="Whether a human agent is currently handling this customer. Use null if unknown."
    )


class BusinessDocument(BaseModel):
    """Business document configuration - Only for business owners/admins."""
    
    business_id: str = Field(description="Unique business identifier (admin only)")
    document_path: str = Field(description="Local or S3 path to document (admin only)")
    document_name: Optional[str] = Field(
        default=None, 
        description="Friendly name for the document"
    )
    document_type: Optional[str] = Field(
        default=None, 
        description="Document type (csv, pdf, docx, etc.)"
    )
    status: str = Field(
        default='active', 
        description="Document status (active/inactive)"
    )
    metadata: Optional[dict] = Field(
        default=None, 
        description="Additional metadata (S3 URI, version, etc.)"
    )


# -----------------------------
# Profile extractor
# -----------------------------
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)



class CartItem(BaseModel):
    """Cart item with quantity support"""
    item: str = Field(description="Name of the product/item")
    quantity: int = Field(
        default=1, 
        ge=1, 
        le=99, 
        description="Quantity to add (1-99, default: 1)"
    )


class CustomerAction(BaseModel):
    """Handle customer requests which may include multiple actions.
    
    This tool should be called whenever the customer:
    - Shares personal information (name, address, preferences)
    - Asks about menu items or availability
    - Wants to add/remove items from cart
    - Wants to place an order
    """
    
    update_profile: bool = Field(
        default=False,
        description="True if message contains profile information to save (name, location, address, preferences) - NOT for cart items"
    )
    
    search_menu: bool = Field(
        default=False,
        description="True if customer is asking about menu items, prices, or availability without wanting to order yet"
    )
    
    search_query: Optional[str] = Field(
        default=None,
        description="The specific menu item, category, or question they're asking about"
    )
    
    add_to_cart: bool = Field(
        default=False,
        description="True if customer wants to add item(s) to their cart or order something"
    )
    

    cart_items: Optional[List[CartItem]] = Field(
        default=None,
        description=(
            "List of items to add to cart with quantities. "
            "Each item should have 'item' (name) and 'quantity' (number). "
            "Example: [{'item': 'Burger', 'quantity': 2}, {'item': 'Fries', 'quantity': 1}]"
        )
    )
    
    remove_from_cart: bool = Field(
        default=False,
        description="True if customer wants to remove item(s) from cart"
    )
    
    # Use List[CartItem] instead of Union
    items_to_remove: Optional[List[CartItem]] = Field(
        default=None,
        description=(
            "Items to remove from cart with quantities. "
            "Each item should have 'item' (name) and 'quantity' (number to remove). "
            "Example: [{'item': 'Burger', 'quantity': 1}]"
        )
    )
    
    view_cart: bool = Field(
        default=False,
        description="True if customer wants to see what's in their cart"
    )
    
    ready_to_order: bool = Field(
        default=False,
        description="True if customer is ready to place/finalize their order"
    )

    ready_to_pay: bool = Field(
        default=False,
        description="True if customer is ready to pay for their order"
    )


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MAX_MEMORY_MESSAGES = 100
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


WHATSAPP_CHATBOT_EXCEPTIONS = (
    # === NETWORK / HTTP ERRORS ===
    ConnectionError,
    requests.exceptions.RequestException,  # WhatsApp API calls
    aiohttp.ClientError,                   # Async HTTP (if using aiohttp)
    urllib3.exceptions.NewConnectionError,
    httpx.ConnectError, httpx.TimeoutException,  # If using httpx
    
    # === DATABASE ERRORS ===
    asyncpg.PostgresError,               # All Postgres errors
    asyncpg.InterfaceError,
    asyncpg.InternalServerError,
    
    
    # === VECTOR STORE / RAG ERRORS ===
    TimeoutError,                        # Embedding timeouts
    ValueError,                          # Invalid embeddings/vectors
    IndexError,                        # Vector index issues
    json.JSONDecodeError,


    # ==== API ERROR ===
    APIError,
    RateLimitError,
    
    
    # === EMBEDDING MODEL ERRORS ===
    openai.APIError, openai.RateLimitError, openai.APIConnectionError,  
    
    # === SYSTEM / FILE ERRORS ===
    OSError,
    FileNotFoundError,
    PermissionError,
    
    
)

# Use DIRECT connection for LangGraph (checkpointer/store)
POSTGRES_URI = os.getenv("POSTGRES_URI")
# Use POOLER connection for PGVector
POSTGRES_URI_POOLER = os.getenv("POSTGRES_URI_POOLER")

if not POSTGRES_URI:
    raise ValueError("POSTGRES_URI not set")
if not POSTGRES_URI_POOLER:
    raise ValueError("POSTGRES_URI_POOLER not set")

# FIND THIS SECTION IN bot.py (around line 320-350) AND REPLACE IT:

@retry(
    retry=retry_if_exception_type(WHATSAPP_CHATBOT_EXCEPTIONS),
    stop=(stop_after_delay(30) | stop_after_attempt(5)),
    wait=wait_combine(
        wait_exponential(multiplier=1, max=10),
        wait_random(min=0, max=2)
    )
)
async def get_pool() -> AsyncConnectionPool:
    """Get or create the database connection pool with retry logic."""
    global _pool
    
    if _pool is None:
        logger.info("📦 Creating database connection pool...")
        
        clean_uri = POSTGRES_URI.split('?')[0]
        
        try:
            _pool = AsyncConnectionPool(
                conninfo=clean_uri,
                min_size=1,
                max_size=10,
                open=False,
                timeout=30,
                max_waiting=5,
                max_lifetime=1800,  # ✅ 30 minutes - connections recycled before timeout
                max_idle=300,       # ✅ 5 minutes - close idle connections
                # 🔥 ADD THESE NEW PARAMETERS:
                reconnect_timeout=60.0,  # Retry reconnection for 60 seconds
                num_workers=3,           # Use connection pool workers
            )
            
            await _pool.open(wait=True, timeout=30)
            
            # Test connection
            async with _pool.connection() as conn:
                result = await conn.execute("SELECT 1")
                await result.fetchone()
            
            logger.info("✅ Database pool connected!")
            
        except Exception as e:
            logger.error(f"❌ Failed to create pool: {e}")
            _pool = None
            raise Exception(f"Database connection failed: {e}")
    
    # 🔥 ADD THIS: Check if pool is healthy, recreate if needed
    try:
        if _pool and _pool.closed:
            logger.warning("Pool was closed, recreating...")
            _pool = None
            return await get_pool()  # Recursive call to recreate
    except Exception as e:
        logger.warning(f"Pool health check failed: {e}")
        _pool = None
        return await get_pool()
    
    return _pool


async def ensure_pool_health():
    """Ensure the connection pool is healthy before use."""
    global _pool
    
    if _pool is None:
        return await get_pool()
    
    try:
        # Quick health check
        async with _pool.connection() as conn:
            await conn.execute("SELECT 1")
        return _pool
    except Exception as e:
        logger.warning(f"Pool unhealthy, recreating: {e}")
        try:
            await _pool.close()
        except:
            pass
        _pool = None
        return await get_pool()


_init_locks: Dict[str, asyncio.Lock]={}
_locks_lock=asyncio.Lock()

async def get_init_lock(business_id):
    """Get or create a lock for a particular business ID.
    
    Prevents thundering herd scenario during RAG initialization.
    """
    async with _locks_lock:
        if business_id not in _init_locks:
            _init_locks[business_id] = asyncio.Lock()

            # Simple cleanup: remove oldest locks if cache grows too large
            if len(_init_locks) > 1000:
                # Remove first 100 locks
                to_remove = list(_init_locks.keys())[:100]
                for key in to_remove:
                    del _init_locks[key]

        return _init_locks[business_id]
    
store = None
saver = None


# -----------------------------
# Connect to DB
# -----------------------------

_pool: AsyncConnectionPool = None

async def drop_old_tables_once():
    """Drop old tables - run once then comment out"""
    import asyncpg
    
    print("Dropping old tables...")
    clean_uri = POSTGRES_URI.split('?')[0]
    conn = await asyncpg.connect(clean_uri, timeout=30)
    
    try:
        await conn.execute("DROP TABLE IF EXISTS store CASCADE;")
        await conn.execute("DROP TABLE IF EXISTS checkpoints CASCADE;")
        await conn.execute("DROP TABLE IF EXISTS writes CASCADE;")
        print("Old tables dropped")
    finally:
        await conn.close()

async def create_tables_manually():
    """Create tables using the connection pool instead of direct connection"""
    print("Creating tables manually via connection pool...")
    
    pool = await ensure_pool_health()
    
    async with pool.connection() as conn:
        # Create store table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS store (
                prefix TEXT NOT NULL,
                key TEXT NOT NULL,
                value JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                ttl_minutes INTEGER,
                expires_at TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (prefix, key)
            );
        """)
        
        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS store_prefix_key_idx ON store(prefix, key);")
        await conn.execute("CREATE INDEX IF NOT EXISTS store_prefix_idx ON store(prefix);")
        await conn.execute("CREATE INDEX IF NOT EXISTS store_expires_at_idx ON store(expires_at) WHERE expires_at IS NOT NULL;")
        print("Store table created")
        
        # Create checkpoints table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS checkpoints_thread_id_checkpoint_ns_idx 
            ON checkpoints(thread_id, checkpoint_ns);
        """)
        print("Checkpoints table created")
        
        # Create writes table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value JSONB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS writes_thread_id_checkpoint_ns_checkpoint_id_idx 
            ON writes(thread_id, checkpoint_ns, checkpoint_id);
        """)
        print("Writes table created")



import asyncpg
# Global session maker
async_session_maker = None

@retry(
    retry=retry_if_exception_type(WHATSAPP_CHATBOT_EXCEPTIONS),
    stop=(stop_after_delay(15) | stop_after_attempt(3)),  # ✅ Reduced from 30s/5 attempts
    wait=wait_combine(
        wait_exponential(multiplier=1, max=5),  # ✅ Reduced from max=10
        wait_random(min=0, max=1)  # ✅ Reduced from max=2
    )
)
async def setup_database():
    """Setup database tables and return store and saver instances."""
    global store, saver
    
    logger.info("=== SETUP_DATABASE CALLED ===")
    
    try:
        pool = await ensure_pool_health()
        
        # Create required tables
        async with pool.connection() as conn:
            # Create user_profiles table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    business_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    name TEXT,
                    location TEXT,
                    address TEXT,
                    cart JSONB,
                    human_active BOOLEAN,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (business_id, user_id)
                );
            """)
            logger.info("✓ user_profiles table created/verified")
            
            # Create business_documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS business_documents (
                    business_id TEXT PRIMARY KEY,
                    document_path TEXT NOT NULL,
                    document_name TEXT,
                    document_type TEXT,
                    status TEXT DEFAULT 'active',
                    metadata JSONB,
                    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("✓ business_documents table created/verified")
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_profiles_business 
                ON user_profiles(business_id);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_business_documents_status 
                ON business_documents(status) WHERE status = 'active';
            """)
            logger.info("✓ Indexes created/verified")
        
        # Create store and saver
        store = AsyncPostgresStore(pool)
        saver = AsyncPostgresSaver(pool)
        
        # Setup store and saver (creates their tables)
        await store.setup()
        await saver.setup()
        
        logger.info("=== DATABASE READY ===")
        return store, saver
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}", exc_info=True)
        raise

async def register_business_document(
    business_id: str, 
    document_path: str, 
    document_name: str = None, 
    document_type: str = None, 
    metadata: dict = None
):
    """Register a document for a business."""
    
    # Validate file exists (skip for S3 URIs)
    if not document_path.startswith('s3://') and not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    # Auto-detect
    if not document_type:
        document_type = os.path.splitext(document_path)[1].replace('.', '').lower()
    if not document_name:
        document_name = os.path.basename(document_path)
    
    # Create Pydantic model for validation
    doc = BusinessDocument(
        business_id=business_id,
        document_path=document_path,
        document_name=document_name,
        document_type=document_type,
        metadata=metadata
    )
    
    pool = await ensure_pool_health()
    
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO business_documents (business_id, document_path, document_name, document_type, metadata, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (business_id) DO UPDATE SET
                document_path = EXCLUDED.document_path,
                document_name = EXCLUDED.document_name,
                document_type = EXCLUDED.document_type,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            (
                doc.business_id,
                doc.document_path,
                doc.document_name,
                doc.document_type,
                json.dumps(doc.metadata) if doc.metadata else None
            )
        )
    
    print(f"✓ Document registered for business {business_id}: {document_name}")


async def get_business_document(business_id: str) -> Optional[str]:
    """Get the document path for a business."""
    pool = await ensure_pool_health()
    
    async with pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT document_path, document_name, status 
            FROM business_documents 
            WHERE business_id = %s AND status = 'active'
            """,
            (business_id,)
        )
        row = await result.fetchone()
        
        if row:
            doc_path, doc_name, status = row
            print(f"✓ Found document for business {business_id}: {doc_name}")
            
            # Validate file still exists (skip for S3 URIs)
            if not doc_path.startswith('s3://') and not os.path.exists(doc_path):
                print(f"WARNING: Document path no longer exists: {doc_path}")
                return None
            
            return doc_path
        
        return None


async def list_business_documents() -> list[BusinessDocument]:
    """List all registered business documents."""
    pool = await ensure_pool_health()
    
    async with pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT business_id, document_path, document_name, document_type, status, metadata
            FROM business_documents
            ORDER BY uploaded_at DESC
            """
        )
        rows = await result.fetchall()
        
        return [
            BusinessDocument(
                business_id=row[0],
                document_path=row[1],
                document_name=row[2],
                document_type=row[3],
                status=row[4],
                metadata=json.loads(row[5]) if row[5] else None
            )
            for row in rows
        ]


async def deactivate_business_document(business_id: str):
    """Deactivate a business document (soft delete)."""
    pool = await ensure_pool_health()
    
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE business_documents
            SET status = 'inactive', updated_at = NOW()
            WHERE business_id = %s
            """,
            (business_id,)
        )
    
    print(f"Document deactivated for business {business_id}")

@retry(
    retry=retry_if_exception_type(WHATSAPP_CHATBOT_EXCEPTIONS),
    stop=(stop_after_delay(30) | stop_after_attempt(5)),
    wait=wait_combine(
        wait_exponential(multiplier=1, max=10),
        wait_random(min=0, max=2)
    )
)
async def chatbot(state: MessagesState, config: RunnableConfig):
    """Load memory from the store and use it to personalize the chatbot's response."""
    global store
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]

    # Consistent namespace order
    namespace = ("profile", business_id, user_id)
    key = "user_memory"
    existing_memory = await store.aget(namespace, key)


    # Extract the actual memory content if it exists
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = MSG_PROMPT.format(user_profile=existing_memory_content)
    
    # FIXED: Pass both system message AND conversation history
    response = await model.bind_tools([CustomerAction]).ainvoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )
    
    return {"messages": [response]}

@retry(
    retry=retry_if_exception_type(WHATSAPP_CHATBOT_EXCEPTIONS),
    stop=(stop_after_delay(30) | stop_after_attempt(5)),
    wait=wait_combine(
        wait_exponential(multiplier=1, max=10),
        wait_random(min=0, max=2)
    )
)
async def write_memory(state: MessagesState, config: RunnableConfig):
    """Extract and save profile information."""
    global store
    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]
    
    namespace = ("profile", business_id, user_id)
    
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
    
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(memory=existing_memories)
    updated_messages = list(merge_message_runs(
        [SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + conversation_history
    ))
    
    result = await profile_extractor.ainvoke({
        "messages": updated_messages,
        "existing": existing_memories
    })
    
    profile_data: Profile = result['responses'][0]
    
    print("=" * 50)
    print("DEBUG: Extracted profile data:")
    print(f"Business ID: {business_id}")
    print(f"User ID: {user_id}")
    print(f"Name: {profile_data.name}")
    print(f"Location: {profile_data.location}")
    print(f"Address: {profile_data.address}")
    print(f"Cart: {profile_data.cart}")
    print(f"Human Active: {profile_data.human_active}")
    print("=" * 50)
    
    # Save to LangGraph store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        await store.aput(
            namespace,
            "user_memory",
            r.model_dump(mode="json"),
        )
    
    # Convert cart to JSON string before inserting
    pool = await ensure_pool_health()
    async with pool.connection() as conn:
        # Convert cart to JSON - handle both list and dict formats
        cart_json = json.dumps(profile_data.cart) if profile_data.cart is not None else None
        
        await conn.execute(
            '''
            INSERT INTO user_profiles (business_id, user_id, name, cart, address, location, human_active, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (business_id, user_id) DO UPDATE SET
                name = COALESCE(EXCLUDED.name, user_profiles.name),
                cart = COALESCE(EXCLUDED.cart, user_profiles.cart),
                address = COALESCE(EXCLUDED.address, user_profiles.address),
                location = COALESCE(EXCLUDED.location, user_profiles.location),
                human_active = COALESCE(EXCLUDED.human_active, user_profiles.human_active),
                updated_at = NOW()
            ''',
            (
                business_id,
                user_id, 
                profile_data.name, 
                cart_json,  
                profile_data.address,
                profile_data.location,
                profile_data.human_active
            )
        )
    
    # Return tool message
    tool_calls = state['messages'][-1].tool_calls
    return {
        "messages": [{
            "role": "tool",
            "content": "Profile information saved successfully",
            "tool_call_id": tool_calls[0]['id']
        }]
    }

@retry(
    retry=retry_if_exception_type(WHATSAPP_CHATBOT_EXCEPTIONS),
    stop=(stop_after_delay(30) | stop_after_attempt(5)),
    wait=wait_combine(
        wait_exponential(multiplier=1, max=10),
        wait_random(min=0, max=2)
    )
)
async def check_address_and_finalize(state: MessagesState, config: RunnableConfig):
    """Check if user has address, ask if not, or prepare for escalation."""
    global store
    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]
    
    # FIXED: Consistent namespace
    namespace = ("profile", business_id, user_id)
    
    existing_memory = await store.aget(namespace, "user_memory")
    
    # Check for address in profile
    has_address = False
    if existing_memory:
        profile_data = existing_memory.value
        has_address = profile_data.get('address') is not None
    
    if not has_address:
        # Ask for address
        tool_calls = state['messages'][-1].tool_calls
        return {
            "messages": [{
                "role": "tool",
                "content": "User is ready to order but no delivery address on file. Please ask for their delivery address.",
                "tool_call_id": tool_calls[0]['id']
            }]
        }
    else:
        # Has address - ready to escalate
        tool_calls = state['messages'][-1].tool_calls
        return {
            "messages": [{
                "role": "tool",
                "content": "User has address on file and is ready to complete order. Escalating to operator.",
                "tool_call_id": tool_calls[0]['id']
            }]
        }
    
@retry(
    retry=retry_if_exception_type(WHATSAPP_CHATBOT_EXCEPTIONS),
    stop=(stop_after_delay(30) | stop_after_attempt(5)),
    wait=wait_combine(
        wait_exponential(multiplier=1, max=10),
        wait_random(min=0, max=2)
    )
)
async def add_to_cart(state: MessagesState, config: RunnableConfig):
    """Add items to the user's cart with quantity support."""
    global store
    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]
    
    # Configuration
    MAX_CART_SIZE = 50
    MAX_ITEM_QUANTITY = 99
    
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    args = tool_call.get('args', {})
    
    cart_items = args.get('cart_items', [])
    
    if not isinstance(cart_items, list):
        return {
            "messages": [{
                "role": "tool",
                "content": "Invalid cart items format. Expected a list.",
                "tool_call_id": tool_call['id']
            }]
        }
    
    if not cart_items:
        return {
            "messages": [{
                "role": "tool",
                "content": "No items specified to add to cart.",
                "tool_call_id": tool_call['id']
            }]
        }
    
    try:
        pool = await ensure_pool_health()
        
        async with pool.connection() as conn:
            # Get current cart
            result = await conn.execute(
                "SELECT cart FROM user_profiles WHERE business_id = %s AND user_id = %s",
                (business_id, user_id)
            )
            
            row = await result.fetchone()
            
            # Parse cart from JSON if it exists
            if row and row[0]:
                import json
                current_cart = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            else:
                current_cart = {}
            
            # Convert old list format to dict format if needed
            if isinstance(current_cart, list):
                current_cart = {item: 1 for item in current_cart}
            
            # Process new items
            added_items = []
            updated_items = []
            quantity_limited_items = []
            
            for item in cart_items:
                # Now we only expect dict format
                if isinstance(item, dict):
                    item_name = item.get('item')
                    quantity = item.get('quantity', 1)
                else:
                    # Fallback for safety
                    continue
                
                if not item_name:
                    continue
                
                # Calculate new quantity
                current_qty = current_cart.get(item_name, 0)
                new_qty = current_qty + quantity
                
                # Apply quantity limit
                if new_qty > MAX_ITEM_QUANTITY:
                    new_qty = MAX_ITEM_QUANTITY
                    quantity_limited_items.append(item_name)
                
                # Update cart
                if current_qty == 0:
                    added_items.append(f"{item_name} (x{new_qty})")
                else:
                    updated_items.append(f"{item_name} ({current_qty} → {new_qty})")
                
                current_cart[item_name] = new_qty
            
            # Check unique items limit
            if len(current_cart) > MAX_CART_SIZE:
                return {
                    "messages": [{
                        "role": "tool",
                        "content": f"Cannot add items. Cart limit is {MAX_CART_SIZE} unique items.",
                        "tool_call_id": tool_call['id']
                    }]
                }
            
            # Calculate total items
            total_items = sum(current_cart.values())
            
            # Convert to JSON before inserting
            import json
            cart_json = json.dumps(current_cart)
            
            # Update database
            await conn.execute(
                '''
                INSERT INTO user_profiles (business_id, user_id, cart, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (business_id, user_id) DO UPDATE SET
                    cart = EXCLUDED.cart,
                    updated_at = NOW()
                ''',
                (business_id, user_id, cart_json)
            )
            
            # Update store
            try:
                namespace = ("profile", business_id, user_id)
                existing_memory = await store.aget(namespace, "user_memory")
                
                if existing_memory:
                    memory_data = existing_memory.value
                    memory_data['cart'] = current_cart
                else:
                    memory_data = {'cart': current_cart}
                
                await store.aput(namespace, "user_memory", memory_data)
            except Exception as store_error:
                logger.warning(f"Failed to update store: {store_error}")
        
        # Format success message
        message_parts = []
        
        if added_items:
            message_parts.append(f"Added: {', '.join(added_items)}")
        
        if updated_items:
            message_parts.append(f"Updated: {', '.join(updated_items)}")
        
        message = ". ".join(message_parts) if message_parts else "Cart updated"
        message += f". Cart now has {len(current_cart)} unique item(s) ({total_items} total items)."
        
        if quantity_limited_items:
            message += f" Note: {', '.join(quantity_limited_items)} reached maximum quantity of {MAX_ITEM_QUANTITY}."
        
        return {
            "messages": [{
                "role": "tool",
                "content": message,
                "tool_call_id": tool_call['id']
            }]
        }
    
    except Exception as e:
        logger.error(f"Failed to add to cart: {e}")
        return {
            "messages": [{
                "role": "tool",
                "content": "Failed to add items to cart. Please try again.",
                "tool_call_id": tool_call['id']
            }]
        }

@retry(
    retry=retry_if_exception_type(WHATSAPP_CHATBOT_EXCEPTIONS),
    stop=(stop_after_delay(30) | stop_after_attempt(5)),
    wait=wait_combine(
        wait_exponential(multiplier=1, max=10),
        wait_random(min=0, max=2)
    )
)
async def remove_cart_item(state: MessagesState, config: RunnableConfig):
    """Remove items from the user's cart with quantity support."""
    global store

    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]

    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    args = tool_call.get("args", {})

    items_to_remove = args.get("items_to_remove", [])

    
    if not isinstance(items_to_remove, list):
        return {
            "messages": [{
                "role": "tool",
                "content": "Invalid items format. Expected a list.",
                "tool_call_id": tool_call["id"]
            }]
        }

    if not items_to_remove:
        return {
            "messages": [{
                "role": "tool",
                "content": "No items specified to remove from cart.",
                "tool_call_id": tool_call["id"]
            }]
        }

    try:
        pool = await ensure_pool_health()

        async with pool.connection() as conn:
            # ---- Load cart ----
            result = await conn.execute(
                "SELECT cart FROM user_profiles WHERE business_id = %s AND user_id = %s",
                (business_id, user_id)
            )
            row = await result.fetchone()

            raw_cart = row[0] if row else None

            # Normalize cart format → always a dict
            if isinstance(raw_cart, str):
                current_cart = json.loads(raw_cart)
            elif isinstance(raw_cart, list):
                current_cart = {item: 1 for item in raw_cart}
            else:
                current_cart = raw_cart or {}

            if not current_cart:
                return {
                    "messages": [{
                        "role": "tool",
                        "content": "Your cart is empty. Nothing to remove.",
                        "tool_call_id": tool_call["id"]
                    }]
                }

            removed_items = []
            updated_items = []
            not_found_items = []

            # ---- Process removals ----
            for item in items_to_remove:
                if not isinstance(item, dict):
                    continue

                item_name = item.get("item")
                quantity_to_remove = item.get("quantity", 1)

                if not item_name:
                    continue

                if item_name not in current_cart:
                    not_found_items.append(item_name)
                    continue

                current_qty = current_cart[item_name]

                if quantity_to_remove >= current_qty:
                    removed_items.append(f"{item_name} (all {current_qty} removed)")
                    del current_cart[item_name]
                else:
                    new_qty = current_qty - quantity_to_remove
                    current_cart[item_name] = new_qty
                    updated_items.append(f"{item_name} ({current_qty} → {new_qty})")

            # ---- No changes? ----
            if not removed_items and not updated_items:
                available_items = [f"{item} (x{qty})" for item, qty in current_cart.items()]
                return {
                    "messages": [{
                        "role": "tool",
                        "content": (
                            f"Could not find items to remove: {', '.join(not_found_items)}. "
                            f"Your cart contains: {', '.join(available_items)}."
                        ),
                        "tool_call_id": tool_call["id"]
                    }]
                }

            # ---- Save updated cart ----
            await conn.execute(
                """
                UPDATE user_profiles
                SET cart = %s, updated_at = NOW()
                WHERE business_id = %s AND user_id = %s
                """,
                (json.dumps(current_cart), business_id, user_id)
            )

            # ---- Update memory store (optional) ----
            try:
                namespace = ("profile", business_id, user_id)
                existing_memory = await store.aget(namespace, "user_memory")
                memory_data = existing_memory.value if existing_memory else {}
                memory_data["cart"] = current_cart
                await store.aput(namespace, "user_memory", memory_data)
            except Exception as store_error:
                logger.warning(f"Failed to update store, but database was updated: {store_error}")

        # ---- Build success message ----
        parts = []
        if removed_items:
            parts.append(f"Removed completely: {', '.join(removed_items)}")
        if updated_items:
            parts.append(f"Reduced quantity: {', '.join(updated_items)}")

        message = ". ".join(parts) if parts else "Cart updated"

        if current_cart:
            remaining_items = [f"{item} (x{qty})" for item, qty in current_cart.items()]
            total_remaining = sum(current_cart.values())
            message += (
                f". Cart now has {len(current_cart)} unique item(s) ({total_remaining} total): "
                f"{', '.join(remaining_items)}."
            )
        else:
            message += ". Your cart is now empty."

        if not_found_items:
            message += f" Note: Could not find: {', '.join(not_found_items)}."

        return {
            "messages": [{
                "role": "tool",
                "content": message,
                "tool_call_id": tool_call["id"]
            }]
        }

    except Exception as e:
        logger.error(f"Failed to remove from cart: {e}")
        return {
            "messages": [{
                "role": "tool",
                "content": "Failed to remove items from cart. Please try again.",
                "tool_call_id": tool_call["id"]
            }]
        }



## RAG

# Setup logging
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# KB1 Setup - MAKE THIS CONFIGURABLE OR USE ENVIRONMENT VARIABLE
KB1_DOC_PATH = os.getenv("KB1_DOC_PATH", "synthetic_restaurant_menu_10000.csv")

# 
_vector_store_cache = {}
_retriever_cache = {}
_file_hash_cache = {}
_embeddings_cache = None  
_documents_loaded = {} 


async def get_file_hash(path):
    """Compute a hash of the file contents to detect changes."""
    hasher = hashlib.sha256()
    loop = asyncio.get_event_loop()
    
    def _read_file():
        with open(path, "rb") as f:
            return hasher.update(f.read())
    
    await loop.run_in_executor(None, _read_file)
    return hasher.hexdigest()

@retry(
    retry=retry_if_exception_type(WHATSAPP_CHATBOT_EXCEPTIONS),
    stop=(stop_after_delay(30) | stop_after_attempt(5)),
    wait=wait_combine(
        wait_exponential(multiplier=1, max=10),
        wait_random(min=0, max=2)
    )
)
async def initialize_rag(business_id: str = None, doc_path: str = None, doc_paths: List = None):
    """Initialize RAG - accepts single doc_path or multiple doc_paths"""
    global _vector_store_cache, _retriever_cache, _file_hash_cache, _embeddings_cache, _documents_loaded
    
    # Validate business_id
    if not business_id:
        raise ValueError("business_id is required for RAG initialization")
    
    # Check if already initialized for THIS business
    if business_id in _retriever_cache and _documents_loaded.get(business_id, False):
        print(f"RAG already initialized for business {business_id}")
        return _retriever_cache[business_id]
    
    
    # Handle both parameter names
    if doc_path and not doc_paths:
        doc_paths = [doc_path]
    
    if not doc_paths:
        doc_paths = await get_business_document(business_id)
        if not doc_paths:
            raise ValueError(f"No document for business '{business_id}'")
        doc_paths = [doc_paths]
    
    try:
        # Initialize embeddings once and store in global cache (shared across businesses)
        if _embeddings_cache is None:
            print("Initializing embeddings model...")
            _embeddings_cache = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )
        
        loop = asyncio.get_event_loop()
        
        collection_name = f"business_{business_id}_menu"
        print(f"Connecting to Supabase pgvector for collection: {collection_name}")
        
        # Initialize PGVector connection to Supabase (use pooler)
        vector_store = PGVector(
            embeddings=_embeddings_cache,
            collection_name=collection_name,
            connection=POSTGRES_URI_POOLER,
            use_jsonb=True,
            async_mode=True,
            pre_delete_collection=False,
            create_extension=False, 
        )
        
        # Check if documents exist in THIS specific collection using database query
        try:
            print(f"Checking database for documents in collection: {collection_name}...")
            pool = await ensure_pool_health()
            
            async with pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT COUNT(*) 
                    FROM langchain_pg_embedding 
                    WHERE collection_id = (
                        SELECT uuid 
                        FROM langchain_pg_collection 
                        WHERE name = %s
                    )
                    """,
                    (collection_name,)
                )
                row = await result.fetchone()
                doc_count = row[0] if row else 0
                
                print(f"Found {doc_count} documents in collection {collection_name}")
                
                if doc_count > 0:
                    # Check if any file has changed
                    doc_paths_list = doc_paths if isinstance(doc_paths, list) else [doc_paths]
                    current_hashes = await asyncio.gather(*[get_file_hash(path) for path in doc_paths_list])
                    
                    if business_id in _file_hash_cache and _file_hash_cache[business_id] != current_hashes:
                        print(f"Document files have changed for business {business_id}, will refresh...")
                        await refresh_rag(business_id=business_id, doc_paths=doc_paths)
                        return _retriever_cache[business_id]
                    
                    # Documents exist and files haven't changed
                    _file_hash_cache[business_id] = current_hashes
                    _documents_loaded[business_id] = True
                    _vector_store_cache[business_id] = vector_store
                    _retriever_cache[business_id] = vector_store.as_retriever(search_kwargs={"k": 5})
                    print(f"RAG initialized from existing documents for business {business_id}")
                    return _retriever_cache[business_id]
                else:
                    print(f"No documents found in collection {collection_name}, will load from files")
                    
        except Exception as e:
            print(f"Error checking collection for business {business_id}: {e}")
            print("Will proceed to load documents from files...")
        
        # Load and process documents
        print(f"Building knowledge base from files for business {business_id}...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        
        # Configuration
        DOC_LOAD_BATCH_SIZE = 5      
        EMBEDDING_BATCH_SIZE = 256    
        VECTOR_INSERT_BATCH_SIZE = 100
                
        # Ensure doc_paths is a list
        doc_paths_list = doc_paths if isinstance(doc_paths, list) else [doc_paths]
        all_documents = []

        # Process documents in batches to avoid memory issues
        print(f"Processing {len(doc_paths_list)} documents in batches of {DOC_LOAD_BATCH_SIZE}...")
        
        for batch_idx in range(0, len(doc_paths_list), DOC_LOAD_BATCH_SIZE):
            batch_paths = doc_paths_list[batch_idx:batch_idx + DOC_LOAD_BATCH_SIZE]
            print(f"\nProcessing batch {batch_idx//DOC_LOAD_BATCH_SIZE + 1}/{(len(doc_paths_list) + DOC_LOAD_BATCH_SIZE - 1)//DOC_LOAD_BATCH_SIZE}")
            
            # Load documents in this batch
            loaded_docs = []
            for doc_path in batch_paths:
                if not doc_path or not os.path.exists(doc_path):
                    raise ValueError(f"Document path does not exist: {doc_path}")
                    
                print(f"  Loading {os.path.basename(doc_path)}...")
                if doc_path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(doc_path)
                elif doc_path.lower().endswith(".docx"):
                    loader = Docx2txtLoader(doc_path)
                elif doc_path.lower().endswith(".csv"):
                    loader = CSVLoader(file_path=doc_path, encoding="utf-8-sig")
                else:
                    raise ValueError(f"Unsupported file format: {doc_path}")
                
                if hasattr(loader, 'aload'):
                    docs = await loader.aload()
                else:
                    docs = await loop.run_in_executor(None, loader.load)
                
                loaded_docs.append(docs)
                print(f"    Loaded {len(docs)} pages/rows")

            # Split documents in this batch in parallel
            print(f"  Splitting {len(loaded_docs)} documents in parallel...")
            chunk_results = await asyncio.gather(*[
                loop.run_in_executor(None, splitter.split_documents, docs)
                for docs in loaded_docs
            ])

            # Flatten chunks from this batch
            batch_chunks = []
            for chunks in chunk_results:
                batch_chunks.extend(chunks)
            
            all_documents.extend(batch_chunks)
            print(f"  Created {len(batch_chunks)} chunks from this batch (Total: {len(all_documents)})")

        if not all_documents:
            raise ValueError("No documents were loaded")

        print(f"\nTotal chunks created: {len(all_documents)}")

        # Store file hashes per business
        _file_hash_cache[business_id] = await asyncio.gather(*[get_file_hash(path) for path in doc_paths_list])
        
        # ============================================================
        # BATCH EMBEDDING GENERATION (Safe and Fast)
        # ============================================================
        print(f"\nGenerating embeddings for {len(all_documents)} chunks...")
        print(f"Using batch size of {EMBEDDING_BATCH_SIZE} for optimal performance")
        
        texts = [doc.page_content for doc in all_documents]
        metadatas = [doc.metadata for doc in all_documents]
        all_embeddings = []
        
        # Generate embeddings in batches using the model's native batching
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
            
            # Use embed_documents (plural) - this is thread-safe and internally optimized
            batch_embeddings = await loop.run_in_executor(
                None,
                _embeddings_cache.embed_documents,  # Native batch method
                batch_texts
            )
            
            all_embeddings.extend(batch_embeddings)
            
            progress = min(i + EMBEDDING_BATCH_SIZE, len(texts))
            print(f"  Embeddings: {progress}/{len(texts)} ({progress*100//len(texts)}%)")
        
        print(f"✓ Generated {len(all_embeddings)} embeddings")
        
        # ============================================================
        # ADD PRE-EMBEDDED DOCUMENTS TO VECTOR STORE
        # ============================================================
        print(f"\nAdding {len(all_embeddings)} pre-embedded documents to Supabase...")
        print(f"Using insert batch size of {VECTOR_INSERT_BATCH_SIZE}")
        
        # Add documents with pre-computed embeddings in batches
        for i in range(0, len(all_embeddings), VECTOR_INSERT_BATCH_SIZE):
            batch_size = min(VECTOR_INSERT_BATCH_SIZE, len(all_embeddings) - i)
            
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = all_embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            # Create tuples of (text, embedding) as required by add_embeddings
            text_embedding_pairs = list(zip(batch_texts, batch_embeddings))
            
            # Add to vector store with pre-computed embeddings
            if hasattr(vector_store, 'aadd_embeddings'):
                # await vector_store.aadd_embeddings(
                #     text_embeddings=text_embedding_pairs,
                #     metadatas=batch_metadatas
                # )
                await vector_store.aadd_embeddings(
                    texts=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
            elif hasattr(vector_store, 'add_embeddings'):
                await loop.run_in_executor(
                    None,
                    vector_store.add_embeddings,
                    text_embedding_pairs,
                    batch_metadatas
                )
            else:
                # Fallback: reconstruct documents with embeddings
                # This shouldn't re-compute embeddings since we're providing them
                batch_docs = [
                    type(all_documents[i + j])(
                        page_content=batch_texts[j],
                        metadata=batch_metadatas[j]
                    )
                    for j in range(len(batch_texts))
                ]
                
                if hasattr(vector_store, 'aadd_documents'):
                    await vector_store.aadd_documents(batch_docs)
                else:
                    await loop.run_in_executor(None, vector_store.add_documents, batch_docs)
            
            progress = min(i + batch_size, len(all_embeddings))
            print(f"  Progress: {progress}/{len(all_embeddings)} ({progress*100//len(all_embeddings)}%)")
        
        _documents_loaded[business_id] = True
        _vector_store_cache[business_id] = vector_store
        _retriever_cache[business_id] = vector_store.as_retriever(search_kwargs={"k": 5})
        print(f"\nSetup complete - {len(all_documents)} documents indexed in Supabase for business {business_id}!")
        
        return _retriever_cache[business_id]
      
    except Exception as e:
        logger.error(f"RAG initialization failed for business {business_id}: {e}")
        raise


async def refresh_rag(business_id=None, doc_path=None):
    """Refresh RAG when file changes detected - clears and reloads documents for a specific business."""
    global _vector_store_cache, _retriever_cache, _file_hash_cache, _embeddings_cache, _documents_loaded
    
    if not business_id:
        raise ValueError("business_id is required for RAG refresh")
    
    print(f"Refreshing knowledge base for business {business_id}...")
    
    loop = asyncio.get_event_loop()
    
    # Delete existing collection in Supabase for this business
    if business_id in _vector_store_cache:
        try:
            print(f"Clearing existing documents from Supabase for business {business_id}...")
            await loop.run_in_executor(
                None,
                lambda: _vector_store_cache[business_id].delete_collection()
            )
            print("Collection cleared")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    # Clear cache for this business only
    _vector_store_cache.pop(business_id, None)
    _retriever_cache.pop(business_id, None)
    _file_hash_cache.pop(business_id, None)
    _documents_loaded.pop(business_id, None)
    # Keep embeddings cache to avoid reloading model
    
    # Reinitialize for this business
    retriever = await initialize_rag(business_id=business_id, doc_path=doc_path)
    print(f"Knowledge base refreshed for business {business_id}!")
    
    return retriever


# File change handler
class KBUpdateHandler(FileSystemEventHandler):
    def __init__(self, business_id):
        self.business_id = business_id
        super().__init__()
    
    def on_modified(self, event):
        if event.src_path == os.path.abspath(KB1_DOC_PATH):
            print(f"Detected change in file: {event.src_path} for business {self.business_id}")
            try:
                asyncio.create_task(refresh_rag(business_id=self.business_id))
            except Exception as e:
                logger.error(f"Error refreshing: {e}")


def start_file_monitoring(business_id):
    """Start watching file for changes for a specific business."""
    event_handler = KBUpdateHandler(business_id)
    observer = Observer()
    
    kb_dir = os.path.dirname(os.path.abspath(KB1_DOC_PATH))
    observer.schedule(event_handler, kb_dir, recursive=False)
    
    observer.start()
    print(f"File monitoring started for business {business_id}...")
    return observer


@retry(
    retry=retry_if_exception_type(WHATSAPP_CHATBOT_EXCEPTIONS),
    stop=(stop_after_delay(40) | stop_after_attempt(5)),
    wait=wait_combine(
        wait_exponential(multiplier=1, max=10),
        wait_random(min=0, max=2)
    )
)

async def rag_search(state: MessagesState, config: RunnableConfig):
    """Perform RAG search on knowledge base."""
    business_id = config["configurable"]["business_id"]
    
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
        
        init_lock = await get_init_lock(business_id)
        async with init_lock:
            if business_id not in _retriever_cache:
                await initialize_rag(business_id=business_id)
        
        retriever = _retriever_cache.get(business_id)
        
        if not retriever:
            return {
                "messages": [{
                    "role": "tool",
                    "content": "Knowledge base not initialized. Please contact support.",
                    "tool_call_id": tool_call['id']
                }]
            }
        
        # This part might benefit from retry (connection issues during search)
        loop = asyncio.get_event_loop()
        if hasattr(retriever, 'ainvoke'):
            relevant_docs=await retriever.ainvoke(search_query)
        else:
            # Fallback logic
            relevant_docs = await loop.run_in_executor(
                None, 
                lambda: retriever.invoke(search_query)
            )
        
        if not relevant_docs:
            return {
                "messages": [{
                    "role": "tool",
                    "content": "I couldn't find any information about that in the knowledge base.",
                    "tool_call_id": tool_call['id']
                }]
            }
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        return {
            "messages": [{
                "role": "tool",
                "content": context,
                "tool_call_id": tool_call['id']
            }]
        }
    
    except (APIError, RateLimitError, Timeout) as e:
        
        logger.error(f"RAG initialization failed after retries for business {business_id}: {e}")
        return {
            "messages": [{
                "role": "tool",
                "content": "Sorry, knowledge base initialization failed. Please try again later.",
                "tool_call_id": tool_call['id']
            }]
        }
    except Exception as e:
        # Other errors get logged and returned (or let retry wrapper handle if it's ConnectionError/OSError)
        logger.error(f"RAG search failed for business {business_id}: {e}", exc_info=True)
        return {
            "messages": [{
                "role": "tool",
                "content": "Sorry, I encountered an error searching the knowledge base. Please try again.",
                "tool_call_id": tool_call['id']
            }]
        }


def route_customer_action(state: MessagesState) -> str:
    """Route to the appropriate action node based on priority."""
    last_message = state["messages"][-1]
    
    if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
        return "__end__"
    
    tool_call = last_message.tool_calls[0]
    args = tool_call.get('args', {})
    
    # Simple priority: Process ONE action at a time
    # Chatbot will handle the rest autonomously
    
    if args.get('ready_to_order'):
        return "check_address_and_finalize"
    
    if args.get('update_profile'):
        return "write_memory"
    
    if args.get('search_menu'):
        return "rag_search"
    
    if args.get('add_to_cart'):
        return "add_to_cart"
    
    if args.get('items_to_remove'):
        return "remove_cart_item"
    
    return "__end__"


# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_node("write_memory", write_memory)
builder.add_node("rag_search", rag_search)
builder.add_node("check_address_and_finalize", check_address_and_finalize)
builder.add_node("add_to_cart", add_to_cart)
builder.add_node("remove_cart_item", remove_cart_item)

builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    route_customer_action,
    {
        "write_memory": "write_memory",
        "rag_search": "rag_search",
        "check_address_and_finalize": "check_address_and_finalize",
        "add_to_cart": "add_to_cart",
        "remove_cart_item": "remove_cart_item",
        "__end__": END
    }
)

# All actions loop back to chatbot
builder.add_edge("write_memory", "chatbot")
builder.add_edge("rag_search", "chatbot")
builder.add_edge("check_address_and_finalize", "chatbot")
builder.add_edge("add_to_cart", "chatbot")
builder.add_edge("remove_cart_item", "chatbot")


async def initialize_graph():
    """Initialize the graph with database connection."""
    global store, saver
    
    logger.info("Initializing graph...")
    
    # FIXED: Correct variable order matching setup_database return
    store, saver = await setup_database()
    
    logger.info("Compiling graph...")
    compiled_graph = builder.compile(checkpointer=saver, store=store)
    
    logger.info("✅ Graph initialized successfully!")
    return compiled_graph