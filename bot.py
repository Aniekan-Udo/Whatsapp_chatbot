import sys
import asyncio

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

from typing import List, Union, Optional


from typing import List, Union, Optional
from pydantic import BaseModel, Field

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
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Use DIRECT connection for LangGraph (checkpointer/store)
POSTGRES_URI = os.getenv("POSTGRES_URI")
# Use POOLER connection for PGVector
POSTGRES_URI_POOLER = os.getenv("POSTGRES_URI_POOLER")

if not POSTGRES_URI:
    raise ValueError("POSTGRES_URI not set")
if not POSTGRES_URI_POOLER:
    raise ValueError("POSTGRES_URI_POOLER not set")


# After all imports and before any functions
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
    
    pool = await get_pool()
    
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




# Global session maker
async_session_maker = None
# Replace your get_pool() and setup_database() functions with these fixed versions:

async def get_pool() -> AsyncConnectionPool:
    """Get or create the database connection pool with better error handling."""
    global _pool
    
    if _pool is None:
        print("\n📦 Creating database connection pool...")
        logger.info("Creating new database connection pool...")
        
        # Clean the URI - remove any query parameters that might cause issues
        clean_uri = POSTGRES_URI.split('?')[0]
        
        try:
            _pool = AsyncConnectionPool(
                conninfo=clean_uri,
                min_size=1,
                max_size=10,
                open=False,
                timeout=30,
                max_waiting=5,
                max_lifetime=1800,
                max_idle=300,
            )
            
            print("🔌 Opening connection pool...")
            logger.info("Opening connection pool...")
            await _pool.open(wait=True, timeout=30)
            
            print("🧪 Testing database connection...")
            logger.info("Testing database connection...")
            async with _pool.connection() as conn:
                result = await conn.execute("SELECT 1")
                await result.fetchone()
            
            print("✅ Database pool connected and verified!\n")
            logger.info("✅ Database pool connected and verified!")
            
        except Exception as e:
            print(f"❌ Failed to create/open pool: {e}\n")
            logger.error(f"❌ Failed to create/open pool: {e}")
            logger.error(f"Connection string (censored): {clean_uri[:20]}...")
            _pool = None
            raise Exception(f"Database connection failed: {e}")
    
    return _pool


async def setup_database():
    """Setup database tables and return store and saver instances."""
    global store, saver
    
    print("\n" + "="*60)
    print("🗄️  DATABASE SETUP")
    print("="*60)
    logger.info("=== SETUP_DATABASE CALLED ===")
    
    try:
        pool = await get_pool()
        
        # Create all necessary tables
        async with pool.connection() as conn:
            print("\n📋 Creating database tables...")
            logger.info("Creating database tables...")
            
            tables_created = []
            
            # 1. Store table
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
            await conn.execute("CREATE INDEX IF NOT EXISTS store_prefix_key_idx ON store(prefix, key);")
            tables_created.append("store")
            print("  ✓ Store table")
            
            # 2. Checkpoints table
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
            tables_created.append("checkpoints")
            print("  ✓ Checkpoints table")
            
            # 3. Writes table
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
            tables_created.append("writes")
            print("  ✓ Writes table")
            
            # 4. User profiles table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    business_id VARCHAR(255) NOT NULL,  
                    user_id VARCHAR(255) NOT NULL, 
                    name VARCHAR(255),
                    cart JSONB,
                    address TEXT,
                    location VARCHAR(255),
                    human_active BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (business_id, user_id)  
                )
            """)
            tables_created.append("user_profiles")
            print("  ✓ User profiles table")
            
            # 5. Business documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS business_documents (
                    business_id VARCHAR(255) PRIMARY KEY,
                    document_path TEXT NOT NULL,
                    document_name VARCHAR(500),
                    document_type VARCHAR(50),
                    uploaded_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    status VARCHAR(50) DEFAULT 'active',
                    metadata JSONB
                )
            """)
            tables_created.append("business_documents")
            print("  ✓ Business documents table")
            
            print(f"\n✅ All {len(tables_created)} tables created/verified successfully!")
            logger.info(f"✓ All {len(tables_created)} tables created successfully!")
        
        # Verify critical tables exist
        async with pool.connection() as conn:
            print("\n🔍 Verifying tables...")
            logger.info("Verifying tables...")
            
            tables_to_check = ['store', 'checkpoints', 'writes', 'user_profiles', 'business_documents']
            verified_tables = []
            
            for table_name in tables_to_check:
                result = await conn.execute(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                    (table_name,)
                )
                exists = (await result.fetchone())[0]
                status = '✓' if exists else '✗'
                print(f"  {status} {table_name}")
                
                if exists:
                    verified_tables.append(table_name)
                else:
                    raise Exception(f"Table '{table_name}' was not created!")
            
            print(f"\n✅ All {len(verified_tables)} tables verified!")
        
        # Create store and saver instances
        print("\n🔧 Creating LangGraph store and saver...")
        logger.info("Creating LangGraph store and saver...")
        store = AsyncPostgresStore(pool)
        saver = AsyncPostgresSaver(pool)
        print("✅ Store and saver created!")
        
        print("\n" + "="*60)
        print("✅ DATABASE READY")
        print("="*60 + "\n")
        logger.info("=== DATABASE READY ===")
        
        return store, saver
        
    except Exception as e:
        print(f"\n❌ Database setup failed: {e}\n")
        logger.error(f"❌ Database setup failed")

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
    
    pool = await get_pool()
    
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
    pool = await get_pool()
    
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
    pool = await get_pool()
    
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
    pool = await get_pool()
    
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
    pool = await get_pool()
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
        pool = await get_pool()
        
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
        pool = await get_pool()

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


async def initialize_rag(business_id=None, doc_path=None):
    """Initialize RAG system with Supabase pgvector. Call once at startup per business."""
    global _vector_store_cache, _retriever_cache, _file_hash_cache, _embeddings_cache, _documents_loaded
    
    # CRITICAL: Validate business_id
    if not business_id:
        raise ValueError("business_id is required for RAG initialization")
    
    # Check if already initialized for THIS business
    if business_id in _retriever_cache and _documents_loaded.get(business_id, False):
        print(f"RAG already initialized for business {business_id}")
        return _retriever_cache[business_id]
    
    # Get document path from database if not explicitly provided
    if not doc_path:
        doc_path = await get_business_document(business_id)
        
        if not doc_path:
            raise ValueError(
                f"No document registered for business '{business_id}'. "
                f"Please register a document first using register_business_document()."
            )
    
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
        )
        
        # Check if documents exist in THIS specific collection using database query
        try:
            print(f"Checking database for documents in collection: {collection_name}...")
            pool = await get_pool()
            
            async with pool.connection() as conn:
                # Query the actual PGVector tables to check if THIS collection has documents
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
                    # Check if file has changed
                    current_hash = await get_file_hash(doc_path)
                    
                    if business_id in _file_hash_cache and _file_hash_cache[business_id] != current_hash:
                        print(f"Document file has changed for business {business_id}, will refresh...")
                        await refresh_rag(business_id=business_id, doc_path=doc_path)
                        return _retriever_cache[business_id]
                    
                    # Documents exist and file hasn't changed
                    _file_hash_cache[business_id] = current_hash
                    _documents_loaded[business_id] = True
                    _vector_store_cache[business_id] = vector_store
                    _retriever_cache[business_id] = vector_store.as_retriever(search_kwargs={"k": 5})
                    print(f"RAG initialized from existing documents for business {business_id}")
                    return _retriever_cache[business_id]
                else:
                    print(f"No documents found in collection {collection_name}, will load from file: {doc_path}")
                    
        except Exception as e:
            print(f"Error checking collection for business {business_id}: {e}")
            print("Will proceed to load documents from file...")
        
        # Load and process documents
        print(f"Building knowledge base from file for business {business_id}...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        all_documents = []
        
        # Load from document
        if doc_path and os.path.exists(doc_path):
            print(f"Processing {doc_path}")
            if doc_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(doc_path)
            elif doc_path.lower().endswith(".docx"):
                loader = Docx2txtLoader(doc_path)
            elif doc_path.lower().endswith(".csv"):
                loader = CSVLoader(file_path=doc_path, encoding="utf-8-sig")
            else:
                raise ValueError(f"Unsupported file format: {doc_path}")
            
            docs = await loop.run_in_executor(None, loader.load)
            doc_chunks = await loop.run_in_executor(None, splitter.split_documents, docs)
            all_documents.extend(doc_chunks)
            print(f"Loaded {len(doc_chunks)} chunks from {doc_path}")
        else:
            raise ValueError(f"Document path does not exist: {doc_path}")

        if not all_documents:
            raise ValueError("No documents were loaded")

        # Store file hash per business
        _file_hash_cache[business_id] = await get_file_hash(doc_path)
        
        print(f"Adding {len(all_documents)} document chunks to Supabase for business {business_id}...")
        print("This may take a few minutes for large datasets...")
        
        # Add documents to PGVector in batches for better performance
        batch_size = 100
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i+batch_size]
            await loop.run_in_executor(
                None,
                lambda b=batch: vector_store.add_documents(b)
            )
            print(f"  Progress: {min(i+batch_size, len(all_documents))}/{len(all_documents)} documents added")
        
        _documents_loaded[business_id] = True
        _vector_store_cache[business_id] = vector_store
        _retriever_cache[business_id] = vector_store.as_retriever(search_kwargs={"k": 5})
        print(f"Setup complete - {len(all_documents)} documents indexed in Supabase for business {business_id}!")
        
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


async def rag_search(state: MessagesState, config: RunnableConfig):
    """Perform RAG search on knowledge base."""
    global _retriever_cache, _embeddings_cache
    business_id = config["configurable"]["business_id"]

    # Initialize RAG for this specific business
    retriever = await initialize_rag(business_id=business_id)
    
    # Check if RAG is initialized
    if not retriever:
        return {
            "messages": [{
                "role": "tool",
                "content": "Knowledge base not initialized. Please contact support.",
                "tool_call_id": state["messages"][-1].tool_calls[0]['id']
            }]
        }
    
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
        # 1. Retrieve relevant documents from vector store using ainvoke if available
        loop = asyncio.get_event_loop()
        
        # Use the retriever's invoke method in executor to avoid blocking
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
        
        # 2. Format context from retrieved docs
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 3. Return the context (or you can generate a response using LLM)
        return {
            "messages": [{
                "role": "tool",
                "content": context,
                "tool_call_id": tool_call['id']
            }]
        }
    
    except Exception as e:
        logger.error(f"RAG search failed for business {business_id}: {e}")
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

# Replace your initialize_graph() function with this:

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