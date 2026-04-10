import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import re
import threading
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import psycopg.errors as psycopg_errors
import pandas as pd
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, StateGraph, START, END
#from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_community.document_loaders import CSVLoader, WebBaseLoader, TextLoader

from pydantic import BaseModel, Field, field_validator

from trustcall import create_extractor
from langchain_groq import ChatGroq

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, JSON, DateTime, Boolean, func, select, text

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)

from cashews import cache
from tenacity import (
    retry, retry_if_exception_type,
    stop_after_attempt,
    wait_exponential
)

from opik.integrations.langchain import OpikTracer
from opik import track, opik_context
import opik

from dotenv import load_dotenv
load_dotenv()

# ============================================
# OPIK CONFIGURATION
# ============================================

# ============================================
# ENVIRONMENT VARIABLES
# ============================================

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables")

POSTGRES_URI = os.getenv("POSTGRES_URI")
POSTGRES_URI_POOLER = os.getenv("POSTGRES_URI_POOLER")
if not POSTGRES_URI or not POSTGRES_URI_POOLER:
    raise ValueError("POSTGRES_URI and POSTGRES_URI_POOLER must be set")

REDIS_URI = os.getenv("REDIS_URI") or os.getenv("CACHE_URL")
if not REDIS_URI:
    raise ValueError("REDIS_URI or CACHE_URL must be set")

print(f"POSTGRES_URI: {POSTGRES_URI.split('@')[0]}@...")
print(f"POSTGRES_URI_POOLER: {POSTGRES_URI_POOLER.split('@')[0]}@...")

from monitoring import logger

logger.info("opik_configured", workspace=os.getenv("OPIK_WORKSPACE", "default"))

# ============================================
# MODEL INITIALIZATION
# ============================================

try:
    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=API_KEY
    )
except Exception as e:
    raise ValueError(f"Failed to initialize ChatGroq model: {e}")

# ============================================
# SQLALCHEMY SETUP - asyncpg (works on Windows)
# ============================================

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

async_uri = POSTGRES_URI.replace('postgresql://', 'postgresql+asyncpg://')

_parsed = urlparse(async_uri)
_query_params = parse_qs(_parsed.query, keep_blank_values=True)
_query_params.pop('sslmode', None)
_query_params.pop('connect_timeout', None)
_new_query = urlencode(_query_params, doseq=True)
async_uri = urlunparse(_parsed._replace(query=_new_query))

engine = create_async_engine(
    async_uri,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=10,
    echo=False,
    connect_args={
        "ssl": True,
        "server_settings": {"jit": "off"},
        "statement_cache_size": 0
    }
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

print("SQLAlchemy engine created with statement_cache_size=0 (Windows compatible)")

Base = declarative_base()

# ============================================
# DATABASE MODELS
# ============================================

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
    business_name = Column(String, nullable=True, default='Our Restaurant')  # ADD
    business_context = Column(String, nullable=True)                          # ADD
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())



# ============================================
# PYDANTIC MODELS
# ============================================

class Profile(BaseModel):
    name: Optional[str] = Field(default=None, description="Customer's name")
    location: Optional[str] = Field(default=None, description="Customer's location")
    address: Optional[str] = Field(default=None, description="Customer's delivery address")
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


# ============================================
# PROMPTS
# ============================================

MSG_PROMPT = """
You are a warm, helpful AI assistant for {business_name}, a restaurant. Your job is to assist customers with their dining experience.

{business_context}

Your responsibilities:
- Greet customers warmly and make them feel welcome
- Help customers explore the menu, including dishes, ingredients, prices, and specials
- Answer questions about allergens, dietary requirements (vegetarian, vegan, gluten-free, halal, etc.)
- Take note of customer preferences and remember them for future visits
- Help customers with reservations, opening hours, location, and general enquiries
- Escalate to a human staff member when the customer has a complaint, special request, or asks to speak to someone
- Always end every turn with a clear, friendly, conversational reply — never leave a turn with only a tool call and no message

Customer Profile:
<user_profile>
{user_profile}
</user_profile>

CRITICAL GUIDELINES:
1. ALWAYS provide a final natural language response after every tool call.
2. After searching the menu or knowledge base, summarise the findings clearly and appetisingly for the customer.
3. After updating a profile, confirm naturally without being robotic (e.g. "Got it, I'll remember you prefer no onions!").
4. If you cannot find an answer in the knowledge base, say so honestly and offer to escalate to a staff member.
5. Do NOT fabricate menu items, prices, or information — only use what the knowledge base or conversation provides.
6. NEVER show raw tool calls, function names, or JSON in your responses. Tool calls are invisible backend actions — the customer must never see them.
7. Keep responses concise and friendly — customers are often messaging on WhatsApp, so avoid long walls of text.
8. Use natural, conversational language. A little warmth and personality goes a long way in hospitality.
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


# ============================================
# CACHE SETUP
# ============================================

cache.setup("mem://")
logger.info("cache_initialized", backend="memory")
# ============================================
# LAZY INITIALIZATION
# ============================================

_profile_extractor = None
_llama_index_setup = False
_setup_lock = threading.Lock()
_rag_init_locks: Dict[str, asyncio.Lock] = {}


def get_profile_extractor():
    global _profile_extractor

    if _profile_extractor is None:
        logger.info("loading_trustcall_extractor")
        _profile_extractor = create_extractor(
            model,
            tools=[Profile],
            tool_choice="Profile",
        )
        logger.info("trustcall_extractor_loaded")

    return _profile_extractor


async def setup_llama_index():
    global _llama_index_setup

    if _llama_index_setup:
        return

    # Use a lock to prevent multiple threads/tasks from initializing simultaneously
    with _setup_lock:
        if _llama_index_setup: # Re-check after acquiring lock
            return

        logger.info("setting_up_llama_index")

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


async def get_init_lock(business_id: str) -> asyncio.Lock:
    if business_id not in _rag_init_locks:
        _rag_init_locks[business_id] = asyncio.Lock()
    return _rag_init_locks[business_id]


# ============================================
# GLOBAL STATE
# ============================================

store = None
saver = None
_saver_ctx = None  # Keeps the AsyncSqliteSaver context manager alive


# ============================================
# DATABASE SETUP - No psycopg pool, no ProactorEventLoop issues
# ============================================

async def setup_database():
    global store, saver, _saver_ctx

    logger.info("database_setup_started")

    try:
        # SQLAlchemy to Neon - handles user profiles, documents, RAG
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("sqlalchemy_tables_created")

        
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

       
        _saver_ctx = AsyncPostgresSaver.from_conn_string(POSTGRES_URI)
        saver = await _saver_ctx.__aenter__()
        await saver.setup()


        from langgraph.store.postgres.aio import AsyncPostgresStore
        from psycopg_pool import AsyncConnectionPool
        
        _store_pool = AsyncConnectionPool(
            conninfo=POSTGRES_URI,
            max_size=5,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
            }
        )
        await _store_pool.open()
        store = AsyncPostgresStore(_store_pool)
        await store.setup()

        logger.info("database_setup_completed")
        return store, saver

    except Exception as e:
        logger.error("database_setup_failed", error=str(e))
        raise


# ============================================
# DOCUMENT MANAGEMENT
# ============================================

async def register_business_config(
    business_id: str,
    business_name: str,
    business_context: str = "",
) -> None:
    async with async_session_factory() as session:
        result = await session.execute(
            select(BusinessDocument).where(
                BusinessDocument.business_id == business_id,
                BusinessDocument.status == 'active'
            )
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise ValueError(f"No document found for business_id: {business_id}. Upload a document first.")
        doc.business_name = business_name
        doc.business_context = business_context
        doc.updated_at = func.now()
        await session.commit()
    logger.info("business_config_registered", business_id=business_id)


async def get_business_config(business_id: str) -> Dict[str, Any]:
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(BusinessDocument).where(
                    BusinessDocument.business_id == business_id,
                    BusinessDocument.status == 'active'
                )
            )
            doc = result.scalar_one_or_none()
            if doc:
                return {
                    "business_name": doc.business_name or "Our Restaurant",
                    "business_context": doc.business_context or "",
                }
    except Exception as e:
        logger.warning("business_config_fetch_failed", business_id=business_id, error=str(e))

    return {
        "business_name": "Our Restaurant",
        "business_context": "",
    }




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


# ============================================
# RAG INITIALIZATION
# ============================================

@cache(ttl=timedelta(minutes=30), key="business_id:{business_id}")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
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
    await setup_llama_index()

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

    missing_files = [p for p in doc_paths if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(f"Documents not found: {', '.join(missing_files)}")

    logger.info("loading_documents", business_id=business_id, file_count=len(doc_paths))

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

        logger.info("vector_store_created", business_id=business_id, table_name=collection_name)

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


# ============================================
# VECTOR STORE HELPER
# ============================================

async def vector_store_exists(collection_name: str) -> bool:
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :name)"
            ), {"name": collection_name})
            return result.scalar()
    except Exception:
        return False


# ============================================
# GROQ API CALL
# ============================================

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

    # ============================================
# GRAPH NODES
# ============================================

async def chatbot(state: MessagesState, config: RunnableConfig):
    global store

    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]

    namespace = ("profile", business_id, user_id)
    existing_memory = await store.aget(namespace, "user_memory")
    existing_memory_content = (
        existing_memory.value.get('memory')
        if existing_memory
        else "No existing memory found."
    )

    biz_config = await get_business_config(business_id)

    system_msg = MSG_PROMPT.format(
        business_name=biz_config["business_name"],
        business_context=biz_config["business_context"],
        user_profile=existing_memory_content
    )

    try:
        response = await call_groq_api(
            [SystemMessage(content=system_msg)] + state["messages"],
            tools=[CustomerAction],
            business_id=business_id,
            user_id=user_id
        )
        return {"messages": [response]}

    except Exception as e:
        logger.error("chatbot_api_failure", business_id=business_id, error=str(e))
        return {
            "messages": [AIMessage(
                content="I'm having trouble connecting right now. Please try again in a moment."
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

            logger.info("cart_updated", business_id=business_id,
                        user_id=user_id, total_items=total_items)

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
        logger.error("add_to_cart_failed", business_id=business_id,
                     user_id=user_id, error=str(e))
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

            logger.info("cart_items_removed", business_id=business_id,
                        user_id=user_id, removed=len(removed_items))

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
        logger.error("remove_cart_item_failed", business_id=business_id,
                     user_id=user_id, error=str(e))
        return {
            "messages": [{
                "role": "tool",
                "content": "Failed to remove items from cart. Please try again.",
                "tool_call_id": tool_call["id"]
            }]
        }


async def view_cart(state: MessagesState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    business_id = config["configurable"]["business_id"]

    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]

    try:
        async with async_session_factory() as session:
            profile = await session.get(UserProfile, (business_id, user_id))

            if not profile or not profile.cart:
                return {
                    "messages": [{
                        "role": "tool",
                        "content": "Your cart is currently empty.",
                        "tool_call_id": tool_call["id"]
                    }]
                }

            current_cart = profile.cart
            if isinstance(current_cart, list): # Legacy format
                current_cart = {item: 1 for item in current_cart}

            if not current_cart:
                return {
                    "messages": [{
                        "role": "tool",
                        "content": "Your cart is currently empty.",
                        "tool_call_id": tool_call["id"]
                    }]
                }

            items = [f"{item} (x{qty})" for item, qty in current_cart.items()]
            total_items = sum(current_cart.values())

            message = f"Your current cart has {len(current_cart)} unique item(s) ({total_items} total):\n" + "\n".join(items)

            return {
                "messages": [{
                    "role": "tool",
                    "content": message,
                    "tool_call_id": tool_call["id"]
                }]
            }
    except Exception as e:
        logger.error("view_cart_failed", business_id=business_id, user_id=user_id, error=str(e))
        return {
            "messages": [{
                "role": "tool",
                "content": "Failed to retrieve your cart. Please try again.",
                "tool_call_id": tool_call["id"]
            }]
        }


async def rag_search(state: MessagesState, config: RunnableConfig):
    await setup_llama_index()

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

        lock = await get_init_lock(business_id)
        async with lock:
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
                    logger.error("rag_init_failed", business_id=business_id,
                                 error=str(init_error))
                    return {
                        "messages": [{
                            "role": "tool",
                            "content": "I'm having trouble accessing the knowledge base. Please try again or contact support.",
                            "tool_call_id": tool_call['id']
                        }]
                    }

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
            logger.error("retriever_creation_failed", business_id=business_id,
                         error=str(retriever_error))
            return {
                "messages": [{
                    "role": "tool",
                    "content": "Error creating search interface. Please try again.",
                    "tool_call_id": tool_call['id']
                }]
            }

        try:
            nodes = await asyncio.wait_for(
                retriever.aretrieve(search_query),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error("rag_search_timeout", business_id=business_id,
                         query=search_query[:100])
            return {
                "messages": [{
                    "role": "tool",
                    "content": "Search is taking longer than expected. Please try again.",
                    "tool_call_id": tool_call['id']
                }]
            }

        logger.info("rag_search_completed", business_id=business_id,
                    results_found=len(nodes))

        if not nodes:
            return {
                "messages": [{
                    "role": "tool",
                    "content": f"No items found for '{search_query}'. Try a different search term.",
                    "tool_call_id": tool_call['id']
                }]
            }

        context_parts = []
        for i, node in enumerate(nodes, 1):
            text = node.node.text
            score = node.score if hasattr(node, 'score') else None
            if score:
                context_parts.append(f"[Score: {score:.2f}]\n{text}")
            else:
                context_parts.append(f"[Result {i}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        return {
            "messages": [{
                "role": "tool",
                "content": context,
                "tool_call_id": tool_call['id']
            }]
        }

    except Exception as e:
        logger.error("rag_search_unexpected_error", business_id=business_id,
                     error=str(e), error_type=type(e).__name__)
        return {
            "messages": [{
                "role": "tool",
                "content": "Sorry, I encountered an unexpected error. Please try again.",
                "tool_call_id": tool_call['id']
            }]
        }


# ============================================
# ROUTING
# ============================================

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
    if args.get('view_cart'):
        return "view_cart"

    return "__end__"


# ============================================
# GRAPH BUILDER
# ============================================

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_node("write_memory", write_memory)
builder.add_node("rag_search", rag_search)
builder.add_node("check_address_and_finalize", check_address_and_finalize)
builder.add_node("add_to_cart", add_to_cart)
builder.add_node("remove_cart_item", remove_cart_item)
builder.add_node("view_cart", view_cart)

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
        "view_cart": "view_cart",
        "__end__": END
    }
)

builder.add_edge("write_memory", "chatbot")
builder.add_edge("rag_search", "chatbot")
builder.add_edge("check_address_and_finalize", "chatbot")
builder.add_edge("add_to_cart", "chatbot")
builder.add_edge("remove_cart_item", "chatbot")
builder.add_edge("view_cart", "chatbot")


# ============================================
# GRAPH INITIALIZATION
# ============================================

async def initialize_graph():
    global store, saver

    logger.info("graph_initialization_started")

    store, saver = await setup_database()

    compiled_graph = builder.compile(checkpointer=saver, store=store)

    logger.info("graph_initialized")
    return compiled_graph
