import asyncio
import sys

# Fix for Windows + psycopg async - ONLY apply on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# On Linux (Render), use default asyncio event loop policy (no changes needed)

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
    # Remove path components
    filename = Path(filename).name
    # Remove dangerous characters (keep only alphanumeric, spaces, dots, dashes, underscores)
    filename = re.sub(r'[^\w\s.-]', '', filename)
    # Remove any remaining path separators that might have slipped through
    filename = filename.replace('/', '').replace('\\', '')
    # Limit length
    return filename[:200]

# ============================================
# GLOBAL STATE
# ============================================

background_tasks: Set[asyncio.Task] = set()

# ============================================
# BACKGROUND INITIALIZATION
# ============================================

async def background_init():
    """Initialize RAG systems in background after startup."""
    try:
        await asyncio.sleep(10)  # Wait for app to be ready
        
        logger.info("background_init_started")
        
        # Get active businesses using SQLAlchemy ORM (not raw SQL)
        async with async_session_factory() as session:
            result = await session.execute(
                select(distinct(BusinessDocument.business_id)).where(
                    BusinessDocument.status == 'active'
                )
            )
            businesses = [row[0] for row in result.fetchall()]
        
        logger.info("background_init_businesses_found", count=len(businesses))
        
        # Initialize RAG for each business
        for business_id in businesses:
            try:
                logger.info("background_rag_init_started", business_id=business_id)
                await initialize_rag(business_id=business_id)
                logger.info("background_rag_init_completed", business_id=business_id)
            except Exception as e:
                logger.error("background_rag_init_failed", 
                           business_id=business_id, 
                           error=str(e))
        
        logger.info("background_init_completed")
        
    except Exception as e:
        logger.error("background_init_critical_failure", error=str(e), exc_info=True)
    finally:
        # Cleanup task reference
        current_task = asyncio.current_task()
        if current_task:
            background_tasks.discard(current_task)

# ============================================
# LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    try:
        logger.info("application_startup_initiated", 
                   platform=sys.platform,
                   python_version=sys.version)
        
        # Initialize monitoring
        logger.info("initializing_monitoring")
        setup_monitoring(app)
        logger.info("monitoring_initialized")
        
        # Initialize database
        logger.info("initializing_database")
        await setup_database()
        logger.info("database_initialized")
        
        # Initialize graph and store in app.state (not globals)
        logger.info("initializing_graph")
        app.state.graph = await initialize_graph()
        logger.info("graph_initialized")
        
        app.state.ready = True
        app.state.background_tasks = background_tasks
        
        logger.info("application_startup_completed", status="READY")
        
        # Start background RAG initialization with proper tracking
        task = asyncio.create_task(background_init())
        background_tasks.add(task)
        task.add_done_callback(lambda t: background_tasks.discard(t))
        logger.info("background_init_scheduled")
        
        yield
        
    except Exception as e:
        logger.error("application_startup_failed", 
                    error=str(e), 
                    error_type=type(e).__name__,
                    exc_info=True)
        raise
    finally:
        # Shutdown
        logger.info("application_shutdown_initiated")
        app.state.ready = False
        
        # Cancel and cleanup background tasks
        for task in list(background_tasks):
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        if background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*background_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("background_tasks_shutdown_timeout")
        
        # Close SQLAlchemy engine
        await engine.dispose()
        logger.info("sqlalchemy_engine_closed")
        
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

# CORS middleware - use environment variable for origins
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins == "*":
    logger.warning("cors_all_origins_enabled", 
                  message="Using wildcard CORS - not recommended for production")

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
        # Always cleanup, even if error occurs
        contextvars.unbind_contextvars("request_id")

# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Simple health check - returns immediately"""
    health = {
        "status": "healthy" if app.state.ready else "starting",
        "database": "connected" if (store and saver) else "disconnected",
        "graph": "initialized" if app.state.graph else "not_initialized"
    }
    
    if not app.state.ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health
        )
    
    return health

@app.get("/readiness", tags=["Health"])
async def readiness_check():
    """Detailed readiness check"""
    try:
        # Test database connection with proper SQLAlchemy select
        async with async_session_factory() as session:
            await session.execute(select(1))
        
        db_status = "ready"
    except Exception as e:
        db_status = f"error: {str(e)[:100]}"
    
    # Use app.state for atomic check
    is_ready = app.state.ready and db_status == "ready"
    
    return {
        "status": "ready" if is_ready else "not_ready",
        "database": db_status,
        "graph": "initialized" if app.state.graph else "not_initialized"
    }

# ============================================
# CHAT ENDPOINTS
# ============================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("60/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """Send a message to the chatbot"""
    if not app.state.ready or not app.state.graph:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready. Please try again."
        )
    
    try:
        logger.info(
            "chat_request_received",
            business_id=chat_request.business_id,
            user_id_hash=hash(chat_request.user_id) % 10000  # Hash PII
        )
        
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
        
        # Invoke graph with timeout (using wait_for for better compatibility)
        input_message = {"messages": [HumanMessage(content=chat_request.message)]}
        result = await asyncio.wait_for(
            app.state.graph.ainvoke(input_message, config),
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
            detail="Failed to process message"
        )

# ============================================
# DOCUMENT MANAGEMENT
# ============================================

@app.post("/documents/upload", response_model=DocumentResponse, tags=["Documents"])
async def upload_document(
    business_id: str = Form(...),
    file: UploadFile = File(...),
    document_name: Optional[str] = Form(None)
):
    """Upload a document file and register it for RAG"""
    
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
        
        # Sanitize filename to prevent path traversal attacks
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
        
        # Save file using async I/O
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
        
        logger.info("file_uploaded_successfully", business_id=business_id)
        
        return DocumentResponse(
            business_id=business_id,
            document_path=str(file_path),
            status="uploaded",
            message="File uploaded successfully. RAG will initialize on first search."
        )
        
    except Exception as e:
        # Clean up file if registration fails (using async)
        if file_path:
            try:
                if await asyncio.to_thread(file_path.exists):
                    await asyncio.to_thread(file_path.unlink)
                    logger.info("cleanup_file_deleted", path=str(file_path))
            except Exception as cleanup_error:
                logger.warning("file_cleanup_failed", 
                             path=str(file_path), 
                             error=str(cleanup_error))
        
        logger.error("file_upload_failed", 
                    business_id=business_id, 
                    error=str(e),
                    error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file"
        )
    finally:
        await file.close()

@app.get("/documents/{business_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(business_id: str):
    """Get registered document for a business"""
    try:
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

@app.delete("/documents/{business_id}", tags=["Documents"])
async def delete_document(business_id: str, delete_file: bool = False):
    """Delete/deactivate a business document"""
    try:
        # Get document before deleting
        doc_path = await get_business_document(business_id)
        
        if not doc_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document found for business_id: {business_id}"
            )
        
        # Mark as inactive in database using SQLAlchemy ORM
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
        
        # Optionally delete physical file (using async)
        if delete_file:
            try:
                if await asyncio.to_thread(os.path.exists, doc_path):
                    await asyncio.to_thread(os.remove, doc_path)
                    message = "Document deleted from database and file system"
                    logger.info("document_file_deleted", business_id=business_id, path=doc_path)
                else:
                    message = "Document marked as deleted (file not found)"
            except Exception as file_error:
                logger.error("document_file_deletion_failed", 
                           business_id=business_id,
                           error=str(file_error))
                message = "Document marked as deleted (file deletion failed)"
        else:
            message = "Document marked as deleted in database"
        
        # Clear RAG cache with proper error handling
        try:
            from cashews import cache
            await cache.delete(f"rag:{business_id}")
            logger.info("rag_cache_cleared", business_id=business_id)
        except Exception as cache_error:
            logger.warning("cache_clear_failed", 
                         business_id=business_id, 
                         error=str(cache_error))
            # Continue anyway - not critical
        
        return {
            "status": "deleted",
            "business_id": business_id,
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("document_deletion_failed", 
                    business_id=business_id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )

# ============================================
# ADMIN ENDPOINTS
# ============================================

@app.post("/admin/initialize-rag/{business_id}", tags=["Admin"])
async def admin_initialize_rag(business_id: str, force_reinit: bool = False):
    """Manually trigger RAG initialization for a business"""
    try:
        logger.info("manual_rag_init_started", 
                   business_id=business_id,
                   force_reinit=force_reinit)
        
        result = await initialize_rag(
            business_id=business_id, 
            force_reinit=force_reinit
        )
        
        logger.info("manual_rag_init_completed", 
                   business_id=business_id,
                   result_status=result.get("status"))
        
        return {
            "status": "success",
            "business_id": business_id,
            "result": result
        }
    except Exception as e:
        logger.error("manual_rag_init_failed", 
                    business_id=business_id, 
                    error=str(e),
                    error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG initialization failed: {str(e)}"
        )

@app.post("/admin/clear-cache", tags=["Admin"])
async def clear_cache(business_id: Optional[str] = None):
    """Clear application cache"""
    try:
        from cashews import cache
        
        if business_id:
            # Clear specific business cache
            await cache.delete(f"rag:{business_id}")
            logger.info("cache_cleared_for_business", business_id=business_id)
            message = f"Cache cleared for business {business_id}"
        else:
            # Clear all cache
            await cache.clear()
            logger.info("cache_cleared_all")
            message = "All cache cleared"
        
        return {
            "status": "success",
            "message": message
        }
    except Exception as e:
        logger.error("cache_clear_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@app.get("/admin/background-tasks", tags=["Admin"])
async def get_background_tasks():
    """Get status of background tasks"""
    tasks_info = []
    for task in app.state.background_tasks:
        tasks_info.append({
            "done": task.done(),
            "cancelled": task.cancelled(),
        })
    
    return {
        "total_tasks": len(app.state.background_tasks),
        "tasks": tasks_info
    }

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )

# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "WhatsApp Chatbot API",
        "version": "1.0.0",
        "status": "running" if app.state.ready else "starting",
        "endpoints": {
            "health": "/health",
            "readiness": "/readiness",
            "docs": "/docs",
            "chat": "/chat",
            "upload": "/documents/upload",
            "get_document": "/documents/{business_id}",
            "delete_document": "/documents/{business_id}",
            "admin": "/admin"
        }
    }

# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    
    logger.info("starting_uvicorn_server", 
               host="0.0.0.0", 
               port=port)
    
    uvicorn.run(
        app,  # Use app object directly
        host="0.0.0.0",
        port=port,
        log_level="info"
    )