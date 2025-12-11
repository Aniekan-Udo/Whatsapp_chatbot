"""
FastAPI application for WhatsApp chatbot with monitoring
"""

import asyncio
import os
import time
import shutil
from typing import Optional, List, Dict, Any
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
from sqlalchemy import select

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Import from bot.py
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
# GLOBAL STATE
# ============================================

graph = None
app_ready = False

# ============================================
# BACKGROUND INITIALIZATION
# ============================================

async def background_init():
    """Initialize RAG systems in background after startup."""
    await asyncio.sleep(10)  # Wait for app to be ready
    
    try:
        # Get active businesses
        async with async_session_factory() as session:
            result = await session.execute(
                "SELECT DISTINCT business_id FROM business_documents WHERE status = 'active'"
            )
            businesses = [row[0] for row in result.fetchall()]
        
        # Initialize RAG for each business
        for business_id in businesses:
            try:
                logger.info("background_rag_init_started", business_id=business_id)
                from rag import initialize_rag
                await initialize_rag(business_id=business_id)
                logger.info("background_rag_init_completed", business_id=business_id)
            except Exception as e:
                logger.error("background_rag_init_failed", 
                           business_id=business_id, 
                           error=str(e))
    except Exception as e:
        logger.error("background_init_error", error=str(e))

# ============================================
# LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global graph, app_ready
    
    try:
        logger.info("application_startup_initiated")
        
        # Initialize monitoring
        setup_monitoring(app)
        logger.info("monitoring_initialized")
        
        # Initialize database
        await setup_database()
        logger.info("database_initialized")
        
        # Initialize graph
        graph = await initialize_graph()
        logger.info("graph_initialized")
        
        # Mark as ready
        app_ready = True
        logger.info("application_startup_completed")
        
        # Start background RAG initialization
        asyncio.create_task(background_init())
        logger.info("background_init_scheduled")
        
        yield
        
    except Exception as e:
        logger.error("application_startup_failed", error=str(e), exc_info=True)
        raise
    finally:
        # Shutdown
        logger.info("application_shutdown_initiated")
        app_ready = False
        
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    """Simple health check - returns immediately"""
    global graph, store, saver, app_ready
    
    health = {
        "status": "healthy" if app_ready else "starting",
        "database": "connected" if (store and saver) else "disconnected",
        "graph": "initialized" if graph else "not_initialized"
    }
    
    if not app_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health
        )
    
    return health

@app.get("/readiness", tags=["Health"])
async def readiness_check():
    """Detailed readiness check"""
    try:
        # Test database connection
        async with async_session_factory() as session:
            await session.execute("SELECT 1")
        
        db_status = "ready"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ready" if (app_ready and db_status == "ready") else "not_ready",
        "database": db_status,
        "graph": "initialized" if graph else "not_initialized"
    }

# ============================================
# CHAT ENDPOINTS
# ============================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("60/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """Send a message to the chatbot"""
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
            user_id=chat_request.user_id
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
        
        # Invoke graph with timeout
        input_message = {"messages": [HumanMessage(content=chat_request.message)]}
        async with asyncio.timeout(30):
            result = await graph.ainvoke(input_message, config)
        
        # Extract AI response
        ai_response = result["messages"][-1].content
        
        logger.info(
            "chat_response_generated",
            business_id=chat_request.business_id,
            user_id=chat_request.user_id
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
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
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
            detail=f"File type {file_ext} not supported. Allowed: {ALLOWED_EXTENSIONS}"
        )
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: 50MB"
        )
    
    file_path = None
    try:
        # Create business directory
        business_dir = UPLOAD_DIR / business_id
        business_dir.mkdir(exist_ok=True)
        
        # Save file with timestamp
        safe_filename = f"{int(time.time())}_{file.filename}"
        file_path = business_dir / safe_filename
        
        logger.info(
            "file_upload_started",
            business_id=business_id,
            filename=file.filename
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
        
        logger.info("file_uploaded_successfully", business_id=business_id)
        
        return DocumentResponse(
            business_id=business_id,
            document_path=str(file_path),
            status="uploaded",
            message="File uploaded. RAG will initialize on first search."
        )
        
    except Exception as e:
        # Clean up file if registration fails
        if file_path and file_path.exists():
            file_path.unlink()
        
        logger.error("file_upload_failed", business_id=business_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
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
        logger.error("get_document_failed", business_id=business_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}"
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
        
        # Clear RAG cache
        try:
            from cashews import cache
            await cache.delete(f"rag:{business_id}")
        except:
            pass
        
        logger.info("document_deleted", business_id=business_id)
        
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
# ADMIN ENDPOINTS
# ============================================

@app.post("/admin/initialize-rag/{business_id}", tags=["Admin"])
async def admin_initialize_rag(business_id: str):
    """Manually trigger RAG initialization for a business"""
    try:
        
        logger.info("manual_rag_init_started", business_id=business_id)
        
        result = await initialize_rag(business_id=business_id, force_reinit=True)
        
        logger.info("manual_rag_init_completed", business_id=business_id)
        
        return {
            "status": "success",
            "business_id": business_id,
            "result": result
        }
    except Exception as e:
        logger.error("manual_rag_init_failed", business_id=business_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/admin/clear-cache", tags=["Admin"])
async def clear_cache():
    """Clear application cache"""
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
            "readiness": "/readiness",
            "docs": "/docs",
            "chat": "/chat",
            "documents": "/documents"
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
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info"
    )