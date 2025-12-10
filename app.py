"""
FastAPI application for WhatsApp chatbot with monitoring
"""

import asyncio
import sys
import os
import time
import shutil
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain
from langchain_core.messages import HumanMessage

# SQLAlchemy
from sqlalchemy import select

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
# Import from bot.py
from bot import (
    initialize_graph,
    register_business_document,
    get_business_document,
    logger,
    store,
    saver,
    async_session_factory,
    BusinessDocument
)

from monitoring import setup_monitoring

# ============================================
# UPLOAD DIRECTORY
# ============================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ============================================
# PYDANTIC MODELS FOR API
# ============================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message", max_length=2000)
    business_id: str = Field(..., description="Business identifier")
    user_id: str = Field(..., description="User identifier")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    thread_id: str
    business_id: str
    user_id: str

class DocumentUploadRequest(BaseModel):
    """Request model for document registration"""
    business_id: str = Field(..., description="Business identifier")
    document_path: str = Field(..., description="Path to document (local or S3)")
    document_name: Optional[str] = Field(None, description="Document name")
    document_type: Optional[str] = Field(None, description="Document type (csv, pdf, txt)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DocumentResponse(BaseModel):
    """Response model for document operations"""
    business_id: str
    document_path: Optional[str]
    status: str
    message: str

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    database: str
    graph: str
    monitoring: str

class UserProfileResponse(BaseModel):
    """Response model for user profile"""
    business_id: str
    user_id: str
    name: Optional[str]
    location: Optional[str]
    address: Optional[str]
    cart: Optional[Dict[str, int]]
    human_active: Optional[bool]

class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history"""
    thread_id: str
    business_id: str
    user_id: str
    messages: List[Dict[str, Any]]

# ============================================
# GLOBAL STATE
# ============================================

graph = None
app_ready = False

# ============================================
# LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global graph, app_ready
    
    try:
        logger.info("application_startup_initiated")
        
        # Initialize monitoring (ONLY ONCE)
        setup_monitoring(
            prometheus_port=9090,
            otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
            auto_instrument_db=True
        )
        logger.info("monitoring_initialized")
        
        # Initialize database and graph
        from bot import setup_database, initialize_graph as init_graph
        await setup_database()
        graph = await init_graph()
        
        app_ready = True
        logger.info("application_startup_completed")
        
        yield
        
    except Exception as e:
        logger.error("application_startup_failed", error=str(e), exc_info=True)
        raise
    finally:
        # Shutdown
        logger.info("application_shutdown_initiated")
        app_ready = False
        
        # Close connection pool
        from bot import _pool, engine
        if _pool and not _pool.closed:
            await _pool.close()
            logger.info("connection_pool_closed")
        
        # Close SQLAlchemy engine
        await engine.dispose()
        logger.info("sqlalchemy_engine_closed")
        
        logger.info("application_shutdown_completed")

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="WhatsApp Chatbot API",
    description="AI-powered WhatsApp chatbot with RAG and cart management",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
    max_age=3600,
)

# ============================================
# RATE LIMITING - CORRECT SETUP
# ============================================
from slowapi.middleware import SlowAPIMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.add_middleware(SlowAPIMiddleware)  # only once
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================
# MIDDLEWARE
# ============================================
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to logs"""
    import uuid
    request_id = str(uuid.uuid4())

    from structlog import contextvars
    contextvars.bind_contextvars(request_id=request_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    contextvars.unbind_contextvars("request_id")
    return response


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    global graph, store, saver, app_ready
    
    health = {
        "status": "healthy" if app_ready else "unhealthy",
        "database": "connected" if (store and saver) else "disconnected",
        "graph": "initialized" if graph else "not_initialized",
        "monitoring": "enabled"
    }
    
    if not app_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health
        )
    
    return health

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint (informational)
    Actual metrics are served by prometheus_client on port 9090
    """
    return {
        "message": "Prometheus metrics available at http://localhost:9090/metrics",
        "port": 9090
    }

# ============================================
# CHAT ENDPOINTS
# ============================================

# ============================================
# CHAT ENDPOINTS - FIXED
# ============================================

# ============================================
# CHAT ENDPOINTS - FIXED
# ============================================

# ============================================
# CHAT ENDPOINTS - FIXED
# ============================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("60/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """
    Send a message to the chatbot
    
    This endpoint handles user messages and returns AI responses.
    Supports conversation continuity via thread_id.
    """
    global graph
    
    if not app_ready or not graph:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready. Please try again."
        )
    
    try:
        logger.info(
            "chat_request_received",
            business_id=chat_request.business_id,
            user_id=chat_request.user_id,
            message_length=len(chat_request.message)
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
        
        # Invoke graph
        input_message = {"messages": [HumanMessage(content=chat_request.message)]}
        async with asyncio.timeout(30):
            result = await graph.ainvoke(input_message, config)
        
        # Extract AI response
        ai_response = result["messages"][-1].content
        
        logger.info(
            "chat_response_generated",
            business_id=chat_request.business_id,
            user_id=chat_request.user_id,
            response_length=len(ai_response)
        )
        
        return ChatResponse(
            response=ai_response,
            thread_id=thread_id,
            business_id=chat_request.business_id,
            user_id=chat_request.user_id
        )
        
    except Exception as e:
        logger.error(
            "chat_request_failed",
            business_id=chat_request.business_id,
            user_id=chat_request.user_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )



@app.get("/chat/history/{thread_id}", response_model=ConversationHistoryResponse, tags=["Chat"])
async def get_conversation_history(
    thread_id: str,
    business_id: str,
    user_id: str
):
    """
    Get conversation history for a thread
    """
    global graph, saver
    
    if not app_ready or not graph or not saver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "business_id": business_id,
                "user_id": user_id
            }
        }
        
        # Get state from checkpointer
        state = await graph.aget(config)
        
        if not state or not state.values.get("messages"):
            return ConversationHistoryResponse(
                thread_id=thread_id,
                business_id=business_id,
                user_id=user_id,
                messages=[]
            )
        
        # Format messages
        messages = [
            {
                "type": msg.type if hasattr(msg, 'type') else "unknown",
                "content": msg.content if hasattr(msg, 'content') else str(msg),
                "role": getattr(msg, 'role', None)
            }
            for msg in state.values["messages"]
        ]
        
        return ConversationHistoryResponse(
            thread_id=thread_id,
            business_id=business_id,
            user_id=user_id,
            messages=messages
        )
        
    except Exception as e:
        logger.error("get_history_failed", thread_id=thread_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
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
    """
    Upload a document file and register it for RAG
    
    Supports: CSV, TXT, PDF files
    Max size: 50MB
    """
    
    # Validate file type
    allowed_extensions = {".csv", ".txt", ".pdf", ".json"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
        )
    
    # Validate file size (50MB max)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: 50MB, got: {file_size / 1024 / 1024:.2f}MB"
        )
    
    file_path = None
    try:
        async with asyncio.timeout(120):
            # Create business directory
            business_dir = UPLOAD_DIR / business_id
            business_dir.mkdir(exist_ok=True)
            
            # Save file with sanitized name
            safe_filename = f"{int(time.time())}_{file.filename}"
            file_path = business_dir / safe_filename
            
            logger.info(
                "file_upload_started",
                business_id=business_id,
                filename=file.filename,
                size_mb=f"{file_size / 1024 / 1024:.2f}"
            )
            
            # Save file
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Register in database
            doc_name = document_name or file.filename
            await register_business_document(
                business_id=business_id,
                document_path=str(file_path),
                document_name=doc_name,
                document_type=file_ext.replace(".", ""),
                metadata={
                    "original_filename": file.filename,
                    "file_size": file_size,
                    "uploaded_at": datetime.now().isoformat()
                }
            )
            
            logger.info(
                "file_uploaded_successfully",
                business_id=business_id,
                path=str(file_path)
            )
            
            return DocumentResponse(
                business_id=business_id,
                document_path=str(file_path),
                status="uploaded",
                message=f"File uploaded successfully. RAG will initialize on first search."
            )
            
    except Exception as e:
        # Clean up file if registration fails
        if file_path and file_path.exists():
            file_path.unlink()
        
        logger.error(
            "file_upload_failed",
            business_id=business_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )
    finally:
        await file.close()


@app.post("/documents/register-local", response_model=DocumentResponse, tags=["Documents"])
async def register_local_document(
    business_id: str = Form(...),
    file_path: str = Form(...),
    document_name: Optional[str] = Form(None)
):
    """
    Register an existing local file without uploading
    """
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}"
        )
    
    try:
        file_ext = Path(file_path).suffix.replace(".", "").lower()
        doc_name = document_name or Path(file_path).name
        async with asyncio.timeout(10):
            await register_business_document(
                business_id=business_id,
                document_path=file_path,
                document_name=doc_name,
                document_type=file_ext,
                metadata={
                    "registration_type": "local_path",
                    "registered_at": datetime.now().isoformat()
                }
            )
            
            return DocumentResponse(
                business_id=business_id,
                document_path=file_path,
                status="registered",
                message="Local file registered successfully"
            )
            
    except Exception as e:
        logger.error("local_file_registration_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register file: {str(e)}"
        )


@app.get("/documents/{business_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(business_id: str):
    """
    Get registered document for a business
    """
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
        logger.error("get_document_failed", business_id=business_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}"
        )


@app.get("/documents/{business_id}/stats", tags=["Documents"])
async def get_document_stats(business_id: str):
    """
    Get statistics about a business document
    """
    try:
        doc_path = await get_business_document(business_id)
        
        if not doc_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document found for business_id: {business_id}"
            )
        
        # Get file stats
        if os.path.exists(doc_path):
            file_size = os.path.getsize(doc_path)
            file_stat = os.stat(doc_path)
            
            return {
                "business_id": business_id,
                "document_path": doc_path,
                "file_size_mb": f"{file_size / 1024 / 1024:.2f}",
                "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "exists": True
            }
        else:
            return {
                "business_id": business_id,
                "document_path": doc_path,
                "exists": False,
                "message": "File path registered but file not found"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_document_stats_failed", business_id=business_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document stats: {str(e)}"
        )


@app.delete("/documents/{business_id}", tags=["Documents"])
async def delete_document(business_id: str, delete_file: bool = False):
    """
    Delete/deactivate a business document
    """
    try:
        # Get document before deleting
        doc_path = await get_business_document(business_id)
        
        if not doc_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document found for business_id: {business_id}"
            )
        
        # Mark as inactive in database
        async with async_session_factory() as session:
            result = await session.execute(
                select(BusinessDocument).where(
                    BusinessDocument.business_id == business_id
                )
            )
            doc = result.scalar_one_or_none()
            
            if doc:
                doc.status = 'deleted'
                await session.commit()
        
        # Optionally delete physical file
        if delete_file and os.path.exists(doc_path):
            os.remove(doc_path)
            message = "Document deleted from database and file system"
        else:
            message = "Document marked as deleted in database"
        
        # Clear cache for this business
        from cashews import cache
        await cache.delete(f"rag:{business_id}")
        
        logger.info("document_deleted", business_id=business_id, file_deleted=delete_file)
        
        return {
            "status": "deleted",
            "business_id": business_id,
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("document_deletion_failed", business_id=business_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

# ============================================
# USER PROFILE MANAGEMENT
# ============================================

@app.get("/profile/{business_id}/{user_id}", response_model=UserProfileResponse, tags=["Profile"])
async def get_user_profile(business_id: str, user_id: str):
    """
    Get user profile information
    """
    global store
    
    if not store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    try:
        async with asyncio.timeout(10):
            namespace = ("profile", business_id, user_id)
            memory = await store.aget(namespace, "user_memory")
            
            if not memory:
                return UserProfileResponse(
                    business_id=business_id,
                    user_id=user_id,
                    name=None,
                    location=None,
                    address=None,
                    cart=None,
                    human_active=None
                )
            
            profile_data = memory.value
            
            return UserProfileResponse(
                business_id=business_id,
                user_id=user_id,
                name=profile_data.get("name"),
                location=profile_data.get("location"),
                address=profile_data.get("address"),
                cart=profile_data.get("cart"),
                human_active=profile_data.get("human_active")
            )
        
    except Exception as e:
        logger.error(
            "get_profile_failed",
            business_id=business_id,
            user_id=user_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve profile: {str(e)}"
        )

# ============================================
# ADMIN ENDPOINTS
# ============================================

@app.post("/admin/clear-cache", tags=["Admin"])
async def clear_cache():
    """
    Clear application cache
    """
    try:
        from cashews import cache
        await cache.clear()
        logger.info("cache_cleared")
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error("cache_clear_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@app.delete("/admin/conversation/{thread_id}", tags=["Admin"])
async def delete_conversation(thread_id: str):
    """
    Delete conversation history (admin only)
    """
    # TODO: Implement conversation deletion from checkpoint
    return {
        "status": "not_implemented",
        "message": "Conversation deletion not yet implemented"
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
        error=str(exc)
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )

# ============================================
# SUPABASE HEALTH
# ============================================
@app.get("/health/supabase", tags=["Health"])
async def supabase_health():
    """
    Detailed Supabase connection health check
    """
    health = {
        "status": "unknown",
        "connection_mode": "unknown",
        "pool_stats": {},
        "database_stats": {},
        "error": None
    }
    
    try:
        from bot import _pool, POSTGRES_URI
        
        # Determine connection mode
        if ':6543' in POSTGRES_URI:
            health["connection_mode"] = "pooler (recommended)"
        elif ':5432' in POSTGRES_URI:
            health["connection_mode"] = "direct (not recommended for production)"
        else:
            health["connection_mode"] = "unknown"
        
        if _pool is None or _pool.closed:
            health["status"] = "pool_not_ready"
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health
            )
        
        # Get pool stats
        health["pool_stats"] = {
            "min_size": _pool.min_size,
            "max_size": _pool.max_size,
            "is_closed": _pool.closed,
        }
        
        # Test connection and get database stats
        try:
            async with asyncio.timeout(10):
                async with _pool.connection() as conn:
                    # ✅ FIX: Use cursor().execute() + fetchone() instead of fetchval()
                    async with conn.cursor() as cur:
                        # Basic connectivity
                        await cur.execute("SELECT 1")
                        result = await cur.fetchone()
                        
                        # Get connection count (if we have permissions)
                        try:
                            await cur.execute("""
                                SELECT count(*) 
                                FROM pg_stat_activity 
                                WHERE datname = current_database()
                            """)
                            row = await cur.fetchone()
                            health["database_stats"]["active_connections"] = row[0] if row else "unknown"
                        except Exception:
                            health["database_stats"]["active_connections"] = "no_permission"
                        
                        # Get database size
                        try:
                            await cur.execute("""
                                SELECT pg_size_pretty(pg_database_size(current_database()))
                            """)
                            row = await cur.fetchone()
                            health["database_stats"]["database_size"] = row[0] if row else "unknown"
                        except Exception:
                            health["database_stats"]["database_size"] = "no_permission"
                        
                        if result and result[0] == 1:
                            health["status"] = "healthy"
                    
        except asyncio.TimeoutError:
            health["status"] = "timeout"
            health["error"] = "Connection test timed out after 10 seconds"
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
        
        status_code = status.HTTP_200_OK if health["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content=health
        )
        
    except Exception as e:
        health["status"] = "error"
        health["error"] = str(e)
        logger.error("supabase_health_check_failed", error=str(e), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health
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
        "status": "running" if app_ready else "starting",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "metrics": "http://localhost:9090/metrics",
            "chat": "/chat",
            "documents": "/documents",
            "profile": "/profile"
        }
    }

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  
        log_level="info"
    )