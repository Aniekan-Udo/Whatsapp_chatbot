import asyncio
import sys

# Fix for Windows + psycopg async - MUST BE FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import re
import time
import uuid
import aiofiles
from typing import Optional, List, Dict, Any, Set
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

# FastAPI
from fastapi import FastAPI, HTTPException, Request, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain
from langchain_core.messages import HumanMessage

# SQLAlchemy
from sqlalchemy import select, distinct

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Import from bot.py - AFTER event loop policy is set
from bot import (
    initialize_graph,
    register_business_document,
    get_business_document,
    initialize_rag,
    logger,
    store,
    saver,
    async_session_factory,
    BusinessDocument,
    setup_database,
    engine
)

from monitoring import setup_monitoring

# ============================================
# CONFIGURATION - MEMORY OPTIMIZED
# ============================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".csv", ".txt", ".pdf", ".json"}

# ============================================
# PYDANTIC MODELS
# ============================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", max_length=2000)
    business_id: str = Field(..., description="Business identifier")
    user_id: str = Field(..., description="User identifier")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation")

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    business_id: str
    user_id: str

class DocumentResponse(BaseModel):
    business_id: str
    document_path: Optional[str]
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    database: str
    graph: str

# ============================================
# UTILITY FUNCTIONS
# ============================================

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    filename = Path(filename).name
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace('/', '').replace('\\', '')
    return filename[:200]

# ============================================
# GLOBAL STATE - LAZY LOADED
# ============================================

background_tasks: Set[asyncio.Task] = set()
_graph_initialized = False  # Track graph state

# ============================================
# MEMORY-OPTIMIZED BACKGROUND INITIALIZATION
# ============================================

async def background_init():
    """Lightweight background init - no heavy RAG loading."""
    try:
        await asyncio.sleep(1)  # Reduced from 10s to 1s
        
        logger.info("background_init_started")
        
        # Minimal DB check only
        async with async_session_factory() as session:
            result = await session.execute(
                select(distinct(BusinessDocument.business_id)).where(
                    BusinessDocument.status == 'active'
                ).limit(5)  # Limit to 5 businesses max
            )
            businesses = [row[0] for row in result.fetchall()]
        
        logger.info("background_init_businesses_found", count=len(businesses))
        logger.info("background_init_completed")
        
    except Exception as e:
        logger.error("background_init_failure", error=str(e))

# ============================================
# LIGHTWEIGHT LIFESPAN - NO HEAVY STARTUP
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Minimal lifespan - defer heavy work to endpoints."""
    try:
        logger.info("minimal_startup_initiated")
        
        # Light monitoring only
        setup_monitoring(app)
        logger.info("monitoring_initialized")
        
        # Light DB setup only
        await setup_database()
        logger.info("database_initialized")
        
        # NO graph init here - lazy load later
        app.state.graph = None
        app.state.ready = False  # Start as not ready
        app.state.background_tasks = background_tasks
        
        # Light background task only
        task = asyncio.create_task(background_init())
        background_tasks.add(task)
        task.add_done_callback(lambda t: background_tasks.discard(t))
        logger.info("light_background_scheduled")
        
        yield
        
    except Exception as e:
        logger.error("startup_failed", error=str(e), exc_info=True)
        raise
    finally:
        logger.info("shutdown_initiated")
        app.state.ready = False
        
        # Cleanup tasks
        for task in list(background_tasks):
            if not task.done():
                task.cancel()
        
        if background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*background_tasks, return_exceptions=True),
                    timeout=3.0  # Reduced timeout
                )
            except asyncio.TimeoutError:
                logger.warning("tasks_shutdown_timeout")
        
        await engine.dispose()
        logger.info("engine_closed")

# ============================================
# FASTAPI APP - MEMORY OPTIMIZED
# ============================================

app = FastAPI(
    title="WhatsApp Chatbot API",
    description="AI-powered WhatsApp chatbot with RAG",
    version="1.0.0",
    lifespan=lifespan  # Keep lifespan but lightweight
)

# CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins == "*":
    logger.warning("cors_all_origins_enabled")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins.split(",") if allowed_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================
# MIDDLEWARE
# ============================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    from structlog import contextvars
    
    request_id = str(uuid.uuid4())
    contextvars.bind_contextvars(request_id=request_id)
    
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        contextvars.unbind_contextvars("request_id")

# ============================================
# HEALTH CHECK - NON-BLOCKING
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Always returns 200 - non-blocking for Render."""
    return {
        "status": "healthy" if app.state.ready else "starting",  # Always 200
        "database": "connected" if (store and saver) else "disconnected",
        "graph": "lazy" if not _graph_initialized else "initialized"
    }

@app.get("/readiness", tags=["Health"])
async def readiness_check():
    """Detailed check - can return 503."""
    try:
        async with async_session_factory() as session:
            await session.execute(select(1))
        db_status = "ready"
    except Exception as e:
        db_status = f"error: {str(e)[:100]}"
    
    is_ready = app.state.ready and db_status == "ready" and _graph_initialized
    return {
        "status": "ready" if is_ready else "not_ready",
        "database": db_status,
        "graph": "initialized" if _graph_initialized else "not_initialized"
    }

# ============================================
# LAZY GRAPH INITIALIZATION
# ============================================

async def ensure_graph_initialized():
    """Lazy load graph on first request."""
    global _graph_initialized
    if not app.state.graph and not _graph_initialized:
        try:
            logger.info("lazy_graph_init_start")
            app.state.graph = await initialize_graph()
            app.state.ready = True
            _graph_initialized = True
            logger.info("lazy_graph_init_complete")
        except Exception as e:
            logger.error("lazy_graph_init_failed", error=str(e))
            app.state.graph = None

# ============================================
# CHAT ENDPOINTS - LAZY LOADED
# ============================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("60/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """Send a message to the chatbot - lazy loads graph."""
    # Lazy init graph
    await ensure_graph_initialized()
    
    if not app.state.graph:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service initializing. Please try again."
        )
    
    try:
        logger.info(
            "chat_request_received",
            business_id=chat_request.business_id,
            user_id_hash=hash(chat_request.user_id) % 10000
        )
        
        thread_id = chat_request.thread_id or f"{chat_request.business_id}_{chat_request.user_id}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "business_id": chat_request.business_id,
                "user_id": chat_request.user_id
            }
        }
        
        input_message = {"messages": [HumanMessage(content=chat_request.message)]}
        result = await asyncio.wait_for(
            app.state.graph.ainvoke(input_message, config),
            timeout=30.0
        )
        
        ai_response = result["messages"][-1].content
        
        logger.info(
            "chat_response_generated",
            business_id=chat_request.business_id,
            response_length=len(ai_response)
        )
        
        return ChatResponse(
            response=ai_response,
            thread_id=thread_id,
            business_id=chat_request.business_id,
            user_id=chat_request.user_id
        )
        
    except asyncio.TimeoutError:
        logger.error("chat_request_timeout", business_id=chat_request.business_id)
        raise HTTPException(status_code=504, detail="Request timed out.")
    except Exception as e:
        logger.error("chat_request_failed", business_id=chat_request.business_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to process message")

# ============================================
# DOCUMENT MANAGEMENT (UNCHANGED - LIGHTWEIGHT)
# ============================================

@app.post("/documents/upload", response_model=DocumentResponse, tags=["Documents"])
async def upload_document(
    business_id: str = Form(...),
    file: UploadFile = File(...),
    document_name: Optional[str] = Form(None)
):
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max size: 50MB")
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    file_path = None
    try:
        business_dir = UPLOAD_DIR / business_id
        business_dir.mkdir(exist_ok=True)
        
        safe_name = sanitize_filename(file.filename)
        safe_filename = f"{int(time.time())}_{safe_name}"
        file_path = business_dir / safe_filename
        
        logger.info("file_upload_started", business_id=business_id, file_size=file_size)
        
        content = await file.read()
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(content)
        
        doc_name = document_name or safe_name
        await register_business_document(
            business_id=business_id,
            document_path=str(file_path),
            document_name=doc_name,
            document_type=file_ext.replace(".", ""),
            metadata={
                "original_filename": file.filename,
                "sanitized_filename": safe_filename,
                "file_size": file_size,
                "uploaded_at": datetime.now().isoformat()
            }
        )
        
        logger.info("file_uploaded_successfully", business_id=business_id)
        
        return DocumentResponse(
            business_id=business_id,
            document_path=str(file_path),
            status="uploaded",
            message="File uploaded successfully."
        )
        
    except Exception as e:
        if file_path and await asyncio.to_thread(file_path.exists):
            await asyncio.to_thread(file_path.unlink)
        logger.error("file_upload_failed", business_id=business_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to upload file")
    finally:
        await file.close()

# [Keep all other endpoints unchanged - they're lightweight]
# ... (get_document, delete_document, admin endpoints, error handlers, root)

@app.get("/documents/{business_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(business_id: str):
    try:
        document_path = await get_business_document(business_id)
        if not document_path:
            raise HTTPException(status_code=404, detail=f"No document found for business_id: {business_id}")
        return DocumentResponse(business_id=business_id, document_path=document_path, status="active", message="Document found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_document_failed", business_id=business_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve document")

@app.delete("/documents/{business_id}", tags=["Documents"])
async def delete_document(business_id: str, delete_file: bool = False):
    # [Keep existing implementation - lightweight]
    try:
        doc_path = await get_business_document(business_id)
        if not doc_path:
            raise HTTPException(status_code=404, detail=f"No document found for business_id: {business_id}")
        
        async with async_session_factory() as session:
            result = await session.execute(
                select(BusinessDocument).where(
                    BusinessDocument.business_id == business_id,
                    BusinessDocument.status == 'active'
                )
            )
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = 'deleted'
                await session.commit()
                logger.info("document_marked_deleted", business_id=business_id)
        
        return {"status": "deleted", "business_id": business_id, "message": "Document marked as deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("document_deletion_failed", business_id=business_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "WhatsApp Chatbot API",
        "version": "1.0.0",
        "status": "starting",  # Always show running
        "endpoints": {
            "health": "/health",
            "readiness": "/readiness",
            "chat": "/chat",
            "upload": "/documents/upload"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info("starting_uvicorn_server", host="0.0.0.0", port=port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
