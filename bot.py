import sys
import asyncio
import os
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



# OPIK IMPORTS
from opik.integrations.langchain import OpikTracer
from opik import track
from opik import opik_context

import psycopg.errors as psycopg_errors
import uuid
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import time
from datetime import datetime
from datetime import timedelta

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

from langchain_community.document_loaders import CSVLoader, WebBaseLoader, TextLoader

from pydantic import BaseModel, Field
from pydantic import field_validator

from trustcall import create_extractor
from langchain_groq import ChatGroq

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, JSON, DateTime, Boolean, func, select

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)

from cashews import cache

from sqlalchemy import text

from tenacity import (
    retry, retry_if_exception_type, 
    stop_after_delay, stop_after_attempt,
    wait_combine, wait_exponential, wait_random
)

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables")

import opik

opik.configure(
    api_key=os.getenv("OPIK_API_KEY")
)

POSTGRES_URI = os.getenv("POSTGRES_URI")
POSTGRES_URI_POOLER = os.getenv("POSTGRES_URI_POOLER")
if not POSTGRES_URI or not POSTGRES_URI_POOLER:
    raise ValueError("POSTGRES_URI and POSTGRES_URI_POOLER must be set")

print(f"POSTGRES_URI: {POSTGRES_URI.split('@')[0]}@...")
print(f"POSTGRES_URI_POOLER: {POSTGRES_URI_POOLER.split('@')[0]}@...")

from monitoring import logger

logger.info("opik_configured", workspace=os.getenv("OPIK_WORKSPACE", "default"))

try:
    model = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), api_key=API_KEY)
except Exception as e:
    raise ValueError(f"Failed to initialize ChatGroq model: {e}")

async_uri = POSTGRES_URI.replace('postgresql://', 'postgresql+asyncpg://')

if 'sslmode' not in async_uri:
    separator = '&' if '?' in async_uri else '?'
    async_uri += f'{separator}sslmode=require'

engine = create_async_engine(
    async_uri,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=10,
    echo=False,
    connect_args={
        "prepare_threshold": 0,
        "autocommit": True,
    }
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

print(f"✅ SQLAlchemy engine created with prepare_threshold=0")

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = 'user_profiles'
    
    business_id = Column(String, primary_key=True)
    user_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    location = Column(String, nullable=True)
    address = Column(String, nullable=True)
    cart = Column(JSON, nullable=True)
    human_active = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

class BusinessDocument(Base):
    __tablename__ = 'business_documents'
    
    business_id = Column(String, primary_key=True)
    document_path = Column(String, nullable=False)
    document_name = Column(String)
    document_type = Column(String)
    status = Column(String, default='active')
    doc_metadata = Column('metadata', JSON)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

class Profile(BaseModel):
    name: Optional[str] = Field(default=None, description="Customer's name")
    location: Optional[str] = Field(default=None, description="Customer's location")
    address: Optional[str] = Field(default=None, description="Customer's delivery address")
    cart: Optional[List[str]] = Field(default=None, description="Items in cart")
    human_active: Optional[bool] = Field(default=None, description="Human agent active")

class CartItem(BaseModel):
    item: str = Field(description="Name of the product/item")
    quantity: int = Field(default=1, ge=1, le=99, description="Quantity (1-99)")

class CustomerAction(BaseModel):
    update_profile: bool = Field(default=False, description="Save profile info")
    search_menu: bool = Field(default=False, description="Search menu items")
    search_query: Optional[str] = Field(default=None, description="Search query")
    add_to_cart: bool = Field(default=False, description="Add items to cart")
    cart_items: Optional[List[CartItem]] = Field(default=None, description="Items to add")
    remove_from_cart: bool = Field(default=False, description="Remove items")
    items_to_remove: Optional[List[CartItem]] = Field(default=None, description="Items to remove")
    view_cart: bool = Field(default=False, description="View cart")
    ready_to_order: bool = Field(default=False, description="Ready to order")
    ready_to_pay: bool = Field(default=False, description="Ready to pay")

    @field_validator('search_query')
    @classmethod
    def validate_search_query(cls, v):
        if v and len(v) > 500:
            raise ValueError("Search query too long (max 500 chars)")
        return v
    
    @field_validator('cart_items')
    @classmethod
    def validate_cart_items(cls, v):
        if v and len(v) > 50:
            raise ValueError("Too many items in single request (max 50)")
        return v

MSG_PROMPT = """
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
2. Identify new information about the user
3. Merge any new information with existing memory
4. Format the memory as a clear, structured profile

CRITICAL: 
- Use null (not the string "None") for fields with no information
- Only include factual information directly stated by the user

Based on the chat history below, please update the user information:"""

_profile_extractor = None
_llama_index_setup = False

def get_profile_extractor():
    global _profile_extractor
    
    if _profile_extractor is None:
        logger.info("loading_trustcall_extractor")
        from trustcall import create_extractor
        
        _profile_extractor = create_extractor(
            model,
            tools=[Profile],
            tool_choice="Profile",
        )
        logger.info("trustcall_extractor_loaded")
    
    return _profile_extractor


import threading

_llama_index_setup = False
_setup_lock = threading.Lock()

def setup_llama_index():
    global _llama_index_setup
    
    # Fast path: already initialized (no lock needed)
    if _llama_index_setup:
        return
    
    # Slow path: might need initialization
    with _setup_lock:
        # Double-check inside lock (another thread might have initialized)
        if _llama_index_setup:
            return
        
        logger.info("setting_up_llama_index")
        
        from llama_index.core import Settings
        from llama_index.embeddings.cohere import CohereEmbedding
        
        Settings.embed_model = CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name=os.getenv("COHERE_EMBED_MODEL", "embed-english-light-v3.0")
        )
        Settings.llm = None
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        _llama_index_setup = True
        logger.info("llama_index_ready")

WHATSAPP_CHATBOT_EXCEPTIONS = (
    ConnectionError,
    psycopg_errors.Error,
    TimeoutError,
    ValueError,
    IndexError,
    json.JSONDecodeError,
    OSError,
    FileNotFoundError,
)

_init_locks: Dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()

async def get_init_lock(business_id):
    async with _locks_lock:
        if business_id not in _init_locks:
            _init_locks[business_id] = asyncio.Lock()
            
            if len(_init_locks) > 1000:
                to_remove = list(_init_locks.keys())[:100]
                for key in to_remove:
                    del _init_locks[key]
        
        return _init_locks[business_id]

store = None
saver = None
_pool: AsyncConnectionPool = None


async def get_pool() -> AsyncConnectionPool:
    global _pool
    
    if _pool is None or _pool.closed:
        logger.info("pool_creation_started")
        
        clean_uri = POSTGRES_URI
        if 'prepare_threshold' in clean_uri:
            clean_uri = clean_uri.replace('&prepare_threshold=0', '').replace('?prepare_threshold=0', '')
        
        logger.info("creating_pool", uri_preview=clean_uri.split('@')[0])
        
        _pool = AsyncConnectionPool(
            conninfo=clean_uri,
            min_size=2,
            max_size=10,
            max_waiting=20,
            timeout=20,
            max_lifetime=600,
            max_idle=120,
            reconnect_timeout=5.0,
            num_workers=2,
            kwargs={
                "prepare_threshold": 0
            }
        )
        
        await asyncio.wait_for(_pool.open(wait=True), timeout=20.0)
        
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
                logger.info("pool_test_successful", result=result[0])
        
        logger.info("pool_created_successfully")
    
    return _pool

import signal


async def shutdown(signal_received):
    logger.info("shutdown_initiated", signal=signal_received)
    
    global _pool
    if _pool:
        await _pool.close()
    
    await engine.dispose()
    
    logger.info("shutdown_complete")

for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda s, f: asyncio.create_task(shutdown(s)))

async def setup_database():
    global store, saver
    
    logger.info("database_setup_started")
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("sqlalchemy_tables_created")
        
        pool = await get_pool()
        
        async with pool.connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS store (
                    prefix TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (prefix, key)
                );
            """)
            
            await conn.execute("CREATE INDEX IF NOT EXISTS store_prefix_idx ON store(prefix);")
            
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
        
        logger.info("langgraph_tables_created")
        
        store = AsyncPostgresStore(pool)
        saver = AsyncPostgresSaver(pool)
        
        await store.setup()
        await saver.setup()
        
        logger.info("database_setup_completed")
        return store, saver
        
    except Exception as e:
        logger.error("database_setup_failed", error=str(e))
        raise


async def register_business_document(
    business_id: str,
    document_path: str,
    document_name: str = None,
    document_type: str = None,
    metadata: dict = None
):
    if not document_path.startswith('s3://') and not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    if not document_type:
        document_type = os.path.splitext(document_path)[1].replace('.', '').lower()
    if not document_name:
        document_name = os.path.basename(document_path)
    
    async with async_session_factory() as session:
        doc = BusinessDocument(
            business_id=business_id,
            document_path=document_path,
            document_name=document_name,
            document_type=document_type,
            metadata=metadata,
            status='active'
        )
        
        await session.merge(doc)
        await session.commit()
    
    logger.info("document_registered", business_id=business_id, document_name=document_name)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((
        psycopg_errors.Error,
        ConnectionError,
    )),
    reraise=True
)
async def get_business_document(business_id: str) -> Optional[str]:
    async with async_session_factory() as session:
        result = await session.execute(
            select(BusinessDocument).where(
                BusinessDocument.business_id == business_id,
                BusinessDocument.status == 'active'
            )
        )
        doc = result.scalar_one_or_none()
        
        if doc:
            if not doc.document_path.startswith('s3://') and not os.path.exists(doc.document_path):
                logger.warning("document_path_missing", business_id=business_id, path=doc.document_path)
                return None
            
            logger.info("document_found", business_id=business_id, document_name=doc.document_name)
            return doc.document_path
        
        return None


async def create_metadata_indexes(business_id: str):
    table_name = f"data_business_{business_id}_menu"
    
    try:
        async with engine.begin() as conn:
            table_exists = await conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{table_name}'
                )
            """))
            exists = (await table_exists.fetchone())[0]
            
            if not exists:
                logger.warning("table_not_found",
                             business_id=business_id,
                             table_name=table_name)
                return
            
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_category 
                ON {table_name} 
                USING gin ((metadata_->>'category'))
            """))
            
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_available 
                ON {table_name} 
                USING btree ((metadata_->>'available'))
            """))
            
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_price 
                ON {table_name} 
                USING btree (((metadata_->>'price')::numeric))
            """))
        
        logger.info("metadata_indexes_created",
                    business_id=business_id,
                    table_name=table_name)
                    
    except Exception as e:
        logger.error("metadata_index_creation_failed",
                    business_id=business_id,
                    error=str(e))

cache.setup("mem://")
logger.info("cache_initialized", backend="memory")

@cache(ttl=timedelta(minutes=30), key="business_id:{business_id}")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        psycopg_errors.Error,
        ConnectionError,
        TimeoutError,
        FileNotFoundError,
    )),
    reraise=True
)
async def initialize_rag(
    business_id: str = None,
    doc_path: str = None,
    doc_paths: List[str] = None
) -> Dict[str, Any]:
    setup_llama_index()
    
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.vector_stores.postgres import PGVectorStore
    
    if not business_id:
        raise ValueError("business_id is required")
    
    collection_name = f"business_{business_id}_menu"
    
    
    if doc_path and not doc_paths:
        doc_paths = [doc_path]
    
    if not doc_paths:
        doc_paths = await get_business_document(business_id)
        if not doc_paths:
            raise ValueError(f"No documents found for business '{business_id}'")
    
    if isinstance(doc_paths, str):
        doc_paths = [doc_paths]
    
    missing_files = []
    for path in doc_paths:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        raise FileNotFoundError(
            f"Documents not found: {', '.join(missing_files)}"
        )
    
    logger.info("loading_documents",
               business_id=business_id,
               file_count=len(doc_paths))
    
    documents = SimpleDirectoryReader(input_files=doc_paths).load_data()
    
    if not documents:
        raise ValueError(f"No documents could be loaded from {doc_paths}")
    
    logger.info("documents_loaded",
               business_id=business_id,
               document_count=len(documents),
               total_chars=sum(len(doc.text) for doc in documents))
    
    try:
        vector_store = PGVectorStore.from_params(
            connection_string=POSTGRES_URI,
            table_name=collection_name,
            embed_dim=384,
            hybrid_search=True,
            perform_setup=True,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )
        
        logger.info("vector_store_created",
                   business_id=business_id,
                   table_name=collection_name)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        logger.info("creating_embeddings", business_id=business_id)
        
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        logger.info("embeddings_created", business_id=business_id)
        
    except Exception as e:
        logger.error("vector_store_creation_failed",
                    business_id=business_id,
                    error=str(e),
                    error_type=type(e).__name__)
        raise
    
    try:
        await create_metadata_indexes(business_id)
    except Exception as idx_error:
        logger.warning("metadata_index_creation_skipped",
                     business_id=business_id,
                     error=str(idx_error))
    
    logger.info("rag_initialized_successfully",
               business_id=business_id,
               collection_name=collection_name)
    
    return {
        "collection_name": collection_name,
        "status": "created",
        "business_id": business_id,
        "document_count": len(documents)
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )),
    reraise=True
)
@cache(ttl="2m")

async def call_groq_api(
    messages: list,
    tools: list = None,
    business_id: str = None,
    user_id: str = None
):
    try:
        logger.info("groq_api_call_started")
        
        opik_config = {}
        if business_id or user_id:
            tracer = OpikTracer(
                tags=["production", "groq"] +
                     ([f"business:{business_id}"] if business_id else []),
                metadata={
                    "business_id": business_id,
                    "user_id": user_id,
                    "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    "has_tools": tools is not None
                }
            )
            opik_config = {"callbacks": [tracer]}
        
        if tools:
            response = await model.bind_tools(tools).ainvoke(messages, config=opik_config)
        else:
            response = await model.ainvoke(messages, config=opik_config)
        
        logger.info("groq_api_call_success")
        return response
        
    except Exception as e:
        logger.error("groq_api_call_failed",
                    error=str(e),
                    error_type=type(e).__name__)
        raise


async def chatbot(state: MessagesState, config: RunnableConfig):
    global store
    
    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]
    
    namespace = ("profile", business_id, user_id)
    key = "user_memory"
    existing_memory = await store.aget(namespace, key)
    
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."
    
    system_msg = MSG_PROMPT.format(user_profile=existing_memory_content)
    
    try:
        response = await call_groq_api(
            [SystemMessage(content=system_msg)] + state["messages"],
            tools=[CustomerAction],
            business_id=business_id,
            user_id=user_id
        )
        return {"messages": [response]}
        
    except Exception as e:
        logger.error("chatbot_api_failure",
                    business_id=business_id,
                    error=str(e))
        
        return {
            "messages": [AIMessage(
                content="I'm having trouble connecting right now. Please try again in a moment or contact support if this continues."
            )]
        }


async def write_memory(state: MessagesState, config: RunnableConfig):
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
    
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        memory=existing_memories
    )
    updated_messages = list(merge_message_runs(
        [SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + conversation_history
    ))
    
    try:
        extractor = get_profile_extractor()
        
        tracer = OpikTracer(
            tags=["profile_extraction", f"business:{business_id}"],
            metadata={"business_id": business_id, "user_id": user_id}
        )
        
        result = await extractor.ainvoke(
            {"messages": updated_messages, "existing": existing_memories},
            config={"callbacks": [tracer]}
        )
        
        profile_data: Profile = result['responses'][0]
        
        logger.info("profile_extracted",
                   business_id=business_id,
                   user_id=user_id,
                   has_name=profile_data.name is not None)
        
    except Exception as e:
        logger.error("profile_extraction_failed",
                    business_id=business_id,
                    error=str(e))
        
        tool_calls = state['messages'][-1].tool_calls
        return {
            "messages": [{
                "role": "tool",
                "content": "Profile update temporarily unavailable.",
                "tool_call_id": tool_calls[0]['id']
            }]
        }
    
    for r in result["responses"]:
        await store.aput(namespace, "user_memory", r.model_dump(mode="json"))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((
            psycopg_errors.Error,
            ConnectionError,
        )),
        reraise=True
    )
    async def save_profile_to_db():
        async with async_session_factory() as session:
            profile = await session.get(UserProfile, (business_id, user_id))
            
            if not profile:
                profile = UserProfile(business_id=business_id, user_id=user_id)
                session.add(profile)
            
            if profile_data.name is not None:
                profile.name = profile_data.name
            if profile_data.location is not None:
                profile.location = profile_data.location
            if profile_data.address is not None:
                profile.address = profile_data.address
            if profile_data.cart is not None:
                profile.cart = profile_data.cart
            if profile_data.human_active is not None:
                profile.human_active = profile_data.human_active
            
            profile.updated_at = func.now()
            
            await session.commit()
    
    try:
        await save_profile_to_db()
        logger.info("profile_saved", business_id=business_id, user_id=user_id)
    except Exception as db_error:
        logger.error("profile_db_save_failed",
                    business_id=business_id,
                    error=str(db_error))
    
    tool_calls = state['messages'][-1].tool_calls
    return {
        "messages": [{
            "role": "tool",
            "content": "Profile information saved successfully",
            "tool_call_id": tool_calls[0]['id']
        }]
    }


async def check_address_and_finalize(state: MessagesState, config: RunnableConfig):
    global store
    
    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]
    
    namespace = ("profile", business_id, user_id)
    existing_memory = await store.aget(namespace, "user_memory")
    
    has_address = False
    if existing_memory:
        profile_data = existing_memory.value
        has_address = profile_data.get('address') is not None
    
    tool_calls = state['messages'][-1].tool_calls
    
    if not has_address:
        logger.info("address_missing", business_id=business_id, user_id=user_id)
        return {
            "messages": [{
                "role": "tool",
                "content": "User is ready to order but no delivery address on file. Please ask for their delivery address.",
                "tool_call_id": tool_calls[0]['id']
            }]
        }
    else:
        logger.info("order_ready_to_escalate", business_id=business_id, user_id=user_id)
        return {
            "messages": [{
                "role": "tool",
                "content": "User has address on file and is ready to complete order. Escalating to operator.",
                "tool_call_id": tool_calls[0]['id']
            }]
        }


async def add_to_cart(state: MessagesState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]
    
    MAX_CART_SIZE = 50
    MAX_ITEM_QUANTITY = 99
    
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    args = tool_call.get('args', {})
    
    cart_items = args.get('cart_items', [])
    
    if not isinstance(cart_items, list) or not cart_items:
        return {
            "messages": [{
                "role": "tool",
                "content": "No items specified to add to cart.",
                "tool_call_id": tool_call['id']
            }]
        }
    
    try:
        async with async_session_factory() as session:
            profile = await session.get(UserProfile, (business_id, user_id))
            
            if not profile:
                profile = UserProfile(business_id=business_id, user_id=user_id, cart={})
                session.add(profile)
            
            current_cart = profile.cart or {}
            
            if isinstance(current_cart, list):
                current_cart = {item: 1 for item in current_cart}
            
            added_items = []
            updated_items = []
            quantity_limited_items = []
            
            for item in cart_items:
                if isinstance(item, dict):
                    item_name = item.get('item')
                    quantity = item.get('quantity', 1)
                else:
                    continue
                
                if not item_name:
                    continue
                
                current_qty = current_cart.get(item_name, 0)
                new_qty = current_qty + quantity
                
                if new_qty > MAX_ITEM_QUANTITY:
                    new_qty = MAX_ITEM_QUANTITY
                    quantity_limited_items.append(item_name)
                
                if current_qty == 0:
                    added_items.append(f"{item_name} (x{new_qty})")
                else:
                    updated_items.append(f"{item_name} ({current_qty} → {new_qty})")
                
                current_cart[item_name] = new_qty
            
            if len(current_cart) > MAX_CART_SIZE:
                return {
                    "messages": [{
                        "role": "tool",
                        "content": f"Cannot add items. Cart limit is {MAX_CART_SIZE} unique items.",
                        "tool_call_id": tool_call['id']
                    }]
                }
            
            total_items = sum(current_cart.values())
            
            profile.cart = current_cart
            profile.updated_at = func.now()
            
            await session.commit()
            
            logger.info("cart_updated", business_id=business_id, user_id=user_id, total_items=total_items)
        
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
        logger.error("add_to_cart_failed", business_id=business_id, user_id=user_id, error=str(e))
        return {
            "messages": [{
                "role": "tool",
                "content": "Failed to add items to cart. Please try again.",
                "tool_call_id": tool_call['id']
            }]
        }


async def remove_cart_item(state: MessagesState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]
    
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    args = tool_call.get("args", {})
    
    items_to_remove = args.get("items_to_remove", [])
    
    if not isinstance(items_to_remove, list) or not items_to_remove:
        return {
            "messages": [{
                "role": "tool",
                "content": "No items specified to remove from cart.",
                "tool_call_id": tool_call["id"]
            }]
        }
    
    try:
        async with async_session_factory() as session:
            profile = await session.get(UserProfile, (business_id, user_id))
            
            if not profile or not profile.cart:
                return {
                    "messages": [{
                        "role": "tool",
                        "content": "Your cart is empty. Nothing to remove.",
                        "tool_call_id": tool_call["id"]
                    }]
                }
            
            current_cart = profile.cart
            if isinstance(current_cart, list):
                current_cart = {item: 1 for item in current_cart}
            
            removed_items = []
            updated_items = []
            not_found_items = []
            
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
            
            if not removed_items and not updated_items:
                available_items = [f"{item} (x{qty})" for item, qty in current_cart.items()]
                return {
                    "messages": [{
                        "role": "tool",
                        "content": f"Could not find items to remove: {', '.join(not_found_items)}. Your cart contains: {', '.join(available_items)}.",
                        "tool_call_id": tool_call["id"]
                    }]
                }
            
            profile.cart = current_cart
            profile.updated_at = func.now()
            
            await session.commit()
            
            logger.info("cart_items_removed", business_id=business_id, user_id=user_id, removed=len(removed_items))
        
        parts = []
        if removed_items:
            parts.append(f"Removed completely: {', '.join(removed_items)}")
        if updated_items:
            parts.append(f"Reduced quantity: {', '.join(updated_items)}")
        
        message = ". ".join(parts) if parts else "Cart updated"
        
        if current_cart:
            remaining_items = [f"{item} (x{qty})" for item, qty in current_cart.items()]
            total_remaining = sum(current_cart.values())
            message += f". Cart now has {len(current_cart)} unique item(s) ({total_remaining} total): {', '.join(remaining_items)}."
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
        logger.error("remove_cart_item_failed", business_id=business_id, user_id=user_id, error=str(e))
        return {
            "messages": [{
                "role": "tool",
                "content": "Failed to remove items from cart. Please try again.",
                "tool_call_id": tool_call["id"]
            }]
        }


async def vector_store_exists(collection_name: str) -> bool:
    """Check if vector store table exists in PostgreSQL"""
    try:
        import asyncpg
        conn = await asyncpg.connect(POSTGRES_URI)
        result = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
            collection_name
        )
        await conn.close()
        return result
    except Exception:
        return False
    
    
async def rag_search(state: MessagesState, config: RunnableConfig):
    setup_llama_index()
    
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.postgres import PGVectorStore
    
    business_id = config["configurable"].get("business_id")
    
    if not business_id:
        logger.error("rag_search_missing_business_id")
        return {
            "messages": [{
                "role": "tool",
                "content": "Configuration error: business_id not provided.",
                "tool_call_id": state["messages"][-1].tool_calls[0]['id']
            }]
        }
    get_init_lock(business_id)

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
    
    logger.info("rag_search_started",
               business_id=business_id,
               query=search_query[:100])
    
    try:
        collection_name = f"business_{business_id}_menu"
    
        # Lazy initialization - only if missing
        if not await vector_store_exists(collection_name):
            try:
                rag_config = await asyncio.wait_for(
                    initialize_rag(business_id=business_id),
                    timeout=60.0
                )
                logger.info("rag_config_retrieved",
                           business_id=business_id,
                           status=rag_config.get("status"))
            except asyncio.TimeoutError:
                logger.error("rag_init_timeout", business_id=business_id)
                try:
                    opik_context.update_current_trace(
                        metadata={
                            "results_count": 0,
                            "retrieval_success": False,
                            "error": "timeout"
                        }
                    )
                except Exception:
                    pass
                
                return {
                    "messages": [{
                        "role": "tool",
                        "content": "The knowledge base is still loading. Please try again in a moment.",
                        "tool_call_id": tool_call['id']
                    }]
                }
            except Exception as init_error:
                logger.error("rag_init_failed",
                            business_id=business_id,
                            error=str(init_error),
                            error_type=type(init_error).__name__)
                return {
                    "messages": [{
                        "role": "tool",
                        "content": "I'm having trouble accessing the knowledge base right now. Please try again or contact support if the issue persists.",
                        "tool_call_id": tool_call['id']
                    }]
                }
        
        # Always create vector store connection and search (outside lazy init block)
        try:
            vector_store = PGVectorStore.from_params(
                connection_string=POSTGRES_URI_POOLER,
                table_name=collection_name,
                embed_dim=384,
                hybrid_search=True,
            )
            
            index = VectorStoreIndex.from_vector_store(vector_store)
            retriever = index.as_retriever(
                similarity_top_k=5,
                vector_store_query_mode="default"
            )
            
            logger.info("retriever_created", business_id=business_id)
            
        except Exception as retriever_error:
            logger.error("retriever_creation_failed",
                        business_id=business_id,
                        error=str(retriever_error))
            return {
                "messages": [{
                    "role": "tool",
                    "content": "Error creating search interface. Please try again.",
                    "tool_call_id": tool_call['id']
                }]
            }
        
        # Perform the search
        try:
            nodes = await asyncio.wait_for(
                retriever.aretrieve(search_query),
                timeout=10.0
            )
            
        except asyncio.TimeoutError:
            logger.error("rag_search_timeout",
                        business_id=business_id,
                        query=search_query[:100])
            try:
                opik_context.update_current_trace(
                    metadata={
                        "results_count": 0,
                        "retrieval_success": False,
                        "error": "timeout"
                    }
                )
            except Exception:
                pass
            
            return {
                "messages": [{
                    "role": "tool",
                    "content": "The search is taking longer than expected. Please try again.",
                    "tool_call_id": tool_call['id']
                }]
            }
        
        logger.info("rag_search_completed",
                   business_id=business_id,
                   results_found=len(nodes))
        
        # Log search metrics
        try:
            avg_score = sum(n.score for n in nodes) / len(nodes) if nodes else 0
            opik_context.update_current_trace(
                tags=["rag_search", f"business:{business_id}"],
                metadata={
                    "business_id": business_id,
                    "query": search_query,
                    "results_count": len(nodes),
                    "avg_relevance_score": avg_score,
                    "top_score": nodes[0].score if nodes else 0,
                    "retrieval_success": len(nodes) > 0
                }
            )
        except Exception:
            pass
        
        # Handle no results
        if not nodes:
            logger.info("rag_search_no_results",
                       business_id=business_id,
                       query=search_query[:100])
            return {
                "messages": [{
                    "role": "tool",
                    "content": f"No items found for '{search_query}'. Try a different search term.",
                    "tool_call_id": tool_call['id']
                }]
            }
        
        # Format results
        context_parts = []
        for i, node in enumerate(nodes, 1):
            text = node.node.text
            score = node.score if hasattr(node, 'score') else None
            
            if score:
                context_parts.append(f"[Score: {score:.2f}]\n{text}")
            else:
                context_parts.append(f"[Result {i}]\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info("rag_search_success",
                   business_id=business_id,
                   results=len(nodes),
                   top_score=nodes[0].score if nodes and hasattr(nodes[0], 'score') else None)
        
        return {
            "messages": [{
                "role": "tool",
                "content": context,
                "tool_call_id": tool_call['id']
            }]
        }
    
    except Exception as e:
        try:
            opik_context.update_current_trace(
                metadata={
                    "results_count": 0,
                    "retrieval_success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
        except Exception:
            pass
        
        logger.error("rag_search_unexpected_error",
                    business_id=business_id,
                    error=str(e),
                    error_type=type(e).__name__)
        return {
            "messages": [{
                "role": "tool",
                "content": "Sorry, I encountered an unexpected error. Please try again.",
                "tool_call_id": tool_call['id']
            }]
        }

def route_customer_action(state: MessagesState) -> str:
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
    
    if args.get('add_to_cart'):
        return "add_to_cart"
    
    if args.get('items_to_remove'):
        return "remove_cart_item"
    
    return "__end__"

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

builder.add_edge("write_memory", "chatbot")
builder.add_edge("rag_search", "chatbot")
builder.add_edge("check_address_and_finalize", "chatbot")
builder.add_edge("add_to_cart", "chatbot")
builder.add_edge("remove_cart_item", "chatbot")


async def initialize_graph():
    global store, saver
    
    logger.info("graph_initialization_started")
    
    store, saver = await setup_database()
    
    compiled_graph = builder.compile(checkpointer=saver, store=store)
    
    logger.info("graph_initialized")
    return compiled_graph