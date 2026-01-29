import asyncio
import sys

# Fix for Windows + psycopg async - ONLY apply on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import re
import time
import uuid
import aiofiles
from typing import Optional, Set
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


# ============================================
# CONFIGURATION
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
# GLOBAL STATE
# ============================================

background_tasks: Set[asyncio.Task] = set()

# Lazy initialization flags
_db_initialized = False
_graph_instance = None
_initialization_lock = asyncio.Lock()

# ============================================
# LAZY INITIALIZATION FUNCTIONS
# ============================================

async def ensure_database_initialized():
    """Initialize database only when first needed"""
    global _db_initialized
    
    if _db_initialized:
        return
    
    async with _initialization_lock:
        # Double-check after acquiring lock
        if _db_initialized:
            return
        
        logger.info("lazy_database_initialization_started")
        await setup_database()
        _db_initialized = True
        logger.info("lazy_database_initialization_completed")

async def get_or_create_graph():
    """Get graph instance, initializing only if needed"""
    global _graph_instance
    
    if _graph_instance is not None:
        return _graph_instance
    
    async with _initialization_lock:
        # Double-check after acquiring lock
        if _graph_instance is not None:
            return _graph_instance
        
        logger.info("lazy_graph_initialization_started")
        await ensure_database_initialized()
        _graph_instance = await initialize_graph()
        logger.info("lazy_graph_initialization_completed")
        return _graph_instance

# ============================================
# LIFESPAN MANAGEMENT (ULTRA-FAST STARTUP)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ultra-fast startup - NO blocking initialization"""
    try:
        logger.info("application_startup_initiated", 
                   platform=sys.platform)
        
        # ONLY set ready flag - NOTHING ELSE
        app.state.ready = True
        app.state.background_tasks = background_tasks
        
        logger.info("application_startup_completed_fast_path", 
                   message="Server ready. Database and graph initialize on first use.")
        
        # DON'T schedule any background tasks here - they block startup
        # Let initialization happen on first actual request
        
        yield
        
    except Exception as e:
        logger.error("application_startup_failed", error=str(e), exc_info=True)
        raise
    finally:
        # Shutdown
        logger.info("application_shutdown_initiated")
        app.state.ready = False
        
        # Cancel background tasks
        for task in list(background_tasks):
            if not task.done():
                task.cancel()
        
        if background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*background_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("background_tasks_shutdown_timeout")
        
        # Close engine if initialized
        if _db_initialized:
            try:
                await engine.dispose()
                logger.info("sqlalchemy_engine_closed")
            except Exception as e:
                logger.warning("engine_disposal_warning", error=str(e))
        
        logger.info("application_shutdown_completed")

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="WhatsApp Chatbot API",
    description="AI-powered WhatsApp chatbot with RAG",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/ping")
@app.head("/ping")  # Support HEAD requests from health checkers
async def ping():
    """Instant response - no dependencies"""
    return {"status": "ok"}

    
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
    """Add request ID to logs and response"""
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
# HEALTH CHECKS - MUST BE FIRST (BEFORE OTHER ROUTES)
# ============================================

@app.get("/ping")
async def ping():
    """Ultra-fast ping endpoint - instant response, no dependencies"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check - minimal overhead"""
    health = {
        "status": "healthy" if app.state.ready else "starting",
        "database": "initialized" if _db_initialized else "lazy",
        "graph": "initialized" if _graph_instance else "lazy"
    }
    
    if not app.state.ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health
        )
    
    return health

@app.get("/readiness")
async def readiness_check():
    """Detailed readiness check - triggers lazy initialization if needed"""
    try:
        # This will trigger lazy initialization
        await ensure_database_initialized()
        
        # Quick DB test
        async with async_session_factory() as session:
            await session.execute(select(1))
        
        db_status = "ready"
    except Exception as e:
        db_status = f"error: {str(e)[:100]}"
    
    return {
        "status": "ready" if app.state.ready and db_status == "ready" else "not_ready",
        "database": db_status,
        "graph": "initialized" if _graph_instance else "will_initialize_on_first_use"
    }

# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "WhatsApp Chatbot API",
        "version": "1.0.0",
        "status": "running" if app.state.ready else "starting",
        "message": "Fast startup enabled - components initialize on first use",
        "endpoints": {
            "ping": "/ping (instant response)",
            "health": "/health (basic check)",
            "readiness": "/readiness (full check)",
            "docs": "/docs",
            "chat": "/chat",
            "upload": "/documents/upload"
        }
    }

# ============================================
# CHAT ENDPOINTS
# ============================================

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("60/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """Send a message to the chatbot"""
    if not app.state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready. Please try again in a moment."
        )
    
    try:
        logger.info(
            "chat_request_received",
            business_id=chat_request.business_id,
            user_id_hash=hash(chat_request.user_id) % 10000
        )
        
        # Lazy initialize graph on first chat request
        graph = await get_or_create_graph()
        
        # Generate thread_id if not provided
        thread_id = chat_request.thread_id or f"{chat_request.business_id}_{chat_request.user_id}"
        
        # Configure graph
        config = {
            "configurable": {
                "thread_id": thread_id,
                "business_id": chat_request.business_id,
                "user_id": chat_request.user_id
            }
        }
        
        # Invoke graph with timeout
        input_message = {"messages": [HumanMessage(content=chat_request.message)]}
        result = await asyncio.wait_for(
            graph.ainvoke(input_message, config),
            timeout=30.0
        )
        
        # Extract AI response
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
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. Please try again."
        )
    except Exception as e:
        logger.error(
            "chat_request_failed",
            business_id=chat_request.business_id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )

# ============================================
# DOCUMENT MANAGEMENT
# ============================================

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    business_id: str = Form(...),
    file: UploadFile = File(...),
    document_name: Optional[str] = Form(None)
):
    """Upload a document file and register it for RAG"""
    
    # Ensure database is initialized
    await ensure_database_initialized()
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Max size: 50MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty"
        )
    
    file_path = None
    try:
        # Create business directory
        business_dir = UPLOAD_DIR / business_id
        business_dir.mkdir(exist_ok=True)
        
        # Sanitize filename
        safe_name = sanitize_filename(file.filename)
        safe_filename = f"{int(time.time())}_{safe_name}"
        file_path = business_dir / safe_filename
        
        logger.info(
            "file_upload_started",
            business_id=business_id,
            original_filename=file.filename,
            sanitized_filename=safe_filename,
            file_size=file_size
        )
        
        # Save file
        content = await file.read()
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(content)
        
        logger.info("file_saved_to_disk", business_id=business_id, path=str(file_path))
        
        # Register in database
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
        
        logger.info("document_registered_successfully", business_id=business_id)
        
        # Return success (200, not 202)
        return DocumentResponse(
            business_id=business_id,
            document_path=str(file_path),
            status="uploaded",
            message="File uploaded successfully. RAG will initialize on first search."
        )
        
    except Exception as e:
        # Cleanup file on error
        if file_path and await asyncio.to_thread(file_path.exists):
            try:
                await asyncio.to_thread(file_path.unlink)
                logger.info("cleanup_deleted_file", path=str(file_path))
            except Exception as cleanup_error:
                logger.warning("cleanup_failed", error=str(cleanup_error))
        
        logger.error("file_upload_failed", 
                    business_id=business_id, 
                    error=str(e),
                    error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )
    finally:
        await file.close()

@app.get("/documents/{business_id}", response_model=DocumentResponse)
async def get_document(business_id: str):
    """Get registered document for a business"""
    try:
        await ensure_database_initialized()
        document_path = await get_business_document(business_id)
        
        if not document_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document found for business_id: {business_id}"
            )
        
        return DocumentResponse(
            business_id=business_id,
            document_path=document_path,
            status="active",
            message="Document found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_document_failed", 
                    business_id=business_id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )

@app.delete("/documents/{business_id}")
async def delete_document(business_id: str, delete_file: bool = False):
    """Delete/deactivate a business document"""
    try:
        await ensure_database_initialized()
        
        doc_path = await get_business_document(business_id)
        
        if not doc_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document found for business_id: {business_id}"
            )
        
        # Mark as inactive
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
        
        # Optionally delete physical file
        if delete_file:
            try:
                if await asyncio.to_thread(os.path.exists, doc_path):
                    await asyncio.to_thread(os.remove, doc_path)
                    message = "Document deleted from database and file system"
                    logger.info("document_file_deleted", business_id=business_id)
                else:
                    message = "Document marked as deleted (file not found)"
            except Exception as file_error:
                logger.error("document_file_deletion_failed", error=str(file_error))
                message = "Document marked as deleted (file deletion failed)"
        else:
            message = "Document marked as deleted in database"
        
        # Clear RAG cache
        try:
            from cashews import cache
            await cache.delete(f"rag:{business_id}")
            logger.info("rag_cache_cleared", business_id=business_id)
        except Exception as cache_error:
            logger.warning("cache_clear_failed", error=str(cache_error))
        
        return {
            "status": "deleted",
            "business_id": business_id,
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("document_deletion_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )

# ============================================
# ADMIN ENDPOINTS
# ============================================

@app.post("/admin/initialize-rag/{business_id}")
async def admin_initialize_rag(business_id: str, force: bool = False):
    """Manually trigger RAG initialization for a business"""
    try:
        await ensure_database_initialized()
        
        logger.info("admin_rag_init_requested", business_id=business_id, force=force)
        
        # Get document path
        doc_path = await get_business_document(business_id)
        if not doc_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document found for business_id: {business_id}"
            )
        
        # Initialize RAG
        result = await initialize_rag(
            business_id=business_id,
            doc_path=doc_path,
            force_reinit=force
        )
        
        return {
            "status": "success",
            "business_id": business_id,
            "result": result,
            "message": "RAG initialized successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin_rag_init_failed", 
                    business_id=business_id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize RAG: {str(e)}"
        )

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    
    # Print port binding info for deployment platforms
    print("=" * 60)
    print(f"🚀 BINDING TO PORT: {port}")
    print(f"🌐 HOST: 0.0.0.0")
    print(f"📍 Health check: http://0.0.0.0:{port}/ping")
    print("=" * 60)
    
    logger.info("starting_uvicorn_server", 
               host="0.0.0.0", 
               port=port)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=65
    )