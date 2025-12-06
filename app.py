import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

import uvicorn
import os
import logging
import uuid
from typing import Optional, Dict, Set
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from concurrent.futures import ThreadPoolExecutor
background_executor = ThreadPoolExecutor(max_workers=3)

from bot import (
    initialize_graph, 
    setup_database, 
    get_pool,
    ensure_pool_health,  
    initialize_rag, 
    start_file_monitoring, 
    refresh_rag,
    register_business_document,
    store, 
    saver
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File upload configuration
UPLOAD_ROOT_PATH = Path("knowledge_base_uploads")
ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".csv", ".xlsx", ".doc", ".docx", ".txt"}

# Global state
graph = None
rag_initialized_businesses: Dict[str, bool] = {}
file_observers = {}
db_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown."""
    global graph
    
    logger.info("🚀 Starting application...")
    
    try:
        # FAST STARTUP: Just compile graph, no DB calls yet
        from bot import builder
        
        # Compile without checkpointer/store (DB not needed yet)
        graph = builder.compile()
        logger.info("✅ Application ready (database will initialize on first request)")
        
        # Ensure upload dir exists
        UPLOAD_ROOT_PATH.mkdir(parents=True, exist_ok=True)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    for observer in file_observers.values():
        try:
            observer.stop()
            observer.join()
        except Exception as e:
            logger.error(f"Error stopping observer: {e}")
    
    try:
        pool = await get_pool()
        if pool and not pool.closed:
            await pool.close()
            logger.info("Database pool closed")
    except Exception as e:
        logger.error(f"Error closing database pool: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="WhatsApp Assistant API",
    description="API for AI-powered WhatsApp business assistant with memory and RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    business_id: str = Field(..., description="Unique identifier for the business")
    user_id: str = Field(..., description="Unique identifier for the user/customer")
    message: str = Field(..., description="User's message")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID (auto-generated if not provided)")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    thread_id: str = Field(..., description="Conversation thread ID")
    business_id: str
    user_id: str


class InitRAGRequest(BaseModel):
    business_id: str = Field(..., description="Business ID to initialize RAG for")
    doc_path: Optional[str] = Field(None, description="Path to knowledge base document (CSV/PDF/DOCX)")


class RefreshRAGRequest(BaseModel):
    business_id: str = Field(..., description="Business ID to refresh RAG for")
    doc_path: Optional[str] = Field(None, description="Path to updated knowledge base document")


class UserProfileResponse(BaseModel):
    business_id: str
    user_id: str
    name: Optional[str] = None
    location: Optional[str] = None
    address: Optional[str] = None
    cart: Optional[Dict[str, int]] = None
    human_active: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    database: str
    rag_systems: Dict[str, bool]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def ensure_db_initialized():
    """Initialize database on first request (lazy initialization)."""
    global graph, db_initialized
    
    if db_initialized:
        return
    
    logger.info("🔧 Initializing database on first request...")
    
    try:
        graph = await initialize_graph()
        db_initialized = True
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Database initialization failed: {str(e)}"
        )


async def ensure_rag_initialized(business_id: str):
    """Ensure RAG is initialized for a specific business."""
    if business_id in rag_initialized_businesses:
        return
    
    logger.info(f"🔧 Initializing RAG for business: {business_id}")
    
    try:
        await initialize_rag(business_id=business_id)
        rag_initialized_businesses[business_id] = True
        logger.info(f"✅ RAG initialized for business: {business_id}")
    except Exception as e:
        logger.error(f"❌ RAG initialization failed for {business_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG initialization failed for business {business_id}: {str(e)}"
        )


def allowed_file(filename: str) -> bool:
    """Check if a filename has an allowed extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


async def process_and_refresh_rag_task(business_id: str, file_path: Path):
    """Async background task to refresh RAG after file upload."""
    logger.info(f"Background task: Processing {file_path} for business {business_id}")
    try:
        # FIXED: Use doc_path (singular) to match bot.py
        await refresh_rag(business_id=business_id, doc_path=str(file_path))
        rag_initialized_businesses[business_id] = True
        logger.info(f"✅ Background task: RAG refreshed for {business_id} with file {file_path.name}")
    except Exception as e:
        logger.error(f"❌ Background RAG refresh failed for {business_id}: {e}", exc_info=True)
        # Mark as not initialized so it can retry on next request
        rag_initialized_businesses.pop(business_id, None)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "WhatsApp Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and service status."""
    db_status = "not_initialized"
    
    if db_initialized:
        try:
            pool = await asyncio.wait_for(ensure_pool_health(), timeout=2.0) 
            
            async with pool.connection() as conn:
                result = await conn.execute("SELECT 1")
                await result.fetchone()
            db_status = "connected"
        except asyncio.TimeoutError:
            db_status = "timeout"
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            db_status = "error"
    else:
        db_status = "initializing"
    
    return {
        "status": "healthy",
        "database": db_status,
        "rag_systems": rag_initialized_businesses
    }



import threading
from concurrent.futures import ThreadPoolExecutor

# Create thread pool for background work (outside of HTTP lifecycle)
background_executor = ThreadPoolExecutor(max_workers=3)

@app.post("/upload", tags=["Knowledge Base"])
async def upload_document(
    business_id: str = Form(...),
    document_file: UploadFile = File(...),
):
    """Upload document - returns immediately, processes in background."""
    
    try:
        logger.info(f"=== UPLOAD REQUEST STARTED ===")
        
        # Ensure DB is ready
        await ensure_db_initialized()
        
        # Validate and save file (FAST - under 5 seconds)
        filename = document_file.filename
        if not filename or not allowed_file(filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type")
        
        business_upload_dir = UPLOAD_ROOT_PATH / business_id
        business_upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = business_upload_dir / unique_filename
        
        # Save file
        content = await document_file.read()
        await asyncio.to_thread(file_path.write_bytes, content)
        logger.info(f"File saved to {file_path}")
        
        # Register in database (FAST)
        await register_business_document(
            business_id=business_id,
            document_path=str(file_path),
            document_name=filename,
            document_type=file_extension.replace('.', '').lower()
        )
        
        # Mark RAG as processing
        await mark_rag_processing(business_id)
        
        # 🔥 KEY FIX: Submit to thread pool (TRULY detached from HTTP request)
        background_executor.submit(
            run_rag_in_background,
            business_id,
            str(file_path)
        )
        
        logger.info("=== UPLOAD REQUEST COMPLETED (background processing started) ===")
        
        # ✅ Return IMMEDIATELY (under 5 seconds total)
        return JSONResponse(
            content={
                "status": "processing",
                "message": f"Document '{filename}' uploaded successfully. Processing in background (estimated 10-15 minutes).",
                "file_id": unique_filename,
                "business_id": business_id,
                "check_status_url": f"/rag/status/{business_id}",
                "estimated_completion": "10-15 minutes"
            }, 
            status_code=202
        )
        
    except Exception as e:
        logger.error(f"❌ UPLOAD FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


def run_rag_in_background(business_id: str, file_path: str):
    """Run RAG initialization in a completely separate thread.
    
    This runs outside the HTTP request lifecycle, so Render won't timeout.
    """
    import asyncio
    
    try:
        logger.info(f"Starting background RAG processing for {business_id}")
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async processing
        loop.run_until_complete(
            process_and_refresh_rag_task(business_id, Path(file_path))
        )
        
        loop.close()
        logger.info(f"Background RAG processing completed for {business_id}")
        
    except Exception as e:
        logger.error(f"Background RAG processing failed for {business_id}: {e}", exc_info=True)
        # Update status to failed
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(mark_rag_failed(business_id, str(e)))
            loop.close()
        except Exception as status_error:
            logger.error(f"Failed to update status: {status_error}")


async def mark_rag_processing(business_id: str):
    """Mark RAG as processing in database."""
    try:
        pool = await ensure_pool_health()
        async with pool.connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_status (
                    business_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT NOW(),
                    error_message TEXT
                )
            """)
            
            await conn.execute("""
                INSERT INTO rag_status (business_id, status, started_at)
                VALUES (%s, 'processing', NOW())
                ON CONFLICT (business_id) DO UPDATE SET
                    status = 'processing',
                    started_at = NOW(),
                    error_message = NULL
            """, (business_id,))
    except Exception as e:
        logger.error(f"Failed to mark RAG as processing: {e}")


async def mark_rag_failed(business_id: str, error: str):
    """Mark RAG as failed in database."""
    try:
        pool = await ensure_pool_health()
        async with pool.connection() as conn:
            await conn.execute("""
                UPDATE rag_status
                SET status = 'failed', error_message = %s
                WHERE business_id = %s
            """, (error, business_id))
    except Exception as e:
        logger.error(f"Failed to mark RAG as failed: {e}")


# Update the status endpoint to check database
@app.get("/rag/status/{business_id}", tags=["Knowledge Base"])
async def get_rag_status(business_id: str):
    """Check RAG initialization status."""
    try:
        pool = await ensure_pool_health()
        async with pool.connection() as conn:
            result = await conn.execute("""
                SELECT status, started_at, error_message
                FROM rag_status
                WHERE business_id = %s
            """, (business_id,))
            
            row = await result.fetchone()
        
        if not row:
            return {
                "business_id": business_id,
                "status": "not_started",
                "rag_initialized": False
            }
        
        status, started_at, error_message = row
        
        response = {
            "business_id": business_id,
            "status": status,
            "rag_initialized": rag_initialized_businesses.get(business_id, False),
            "started_at": started_at.isoformat() if started_at else None
        }
        
        if status == "processing" and started_at:
            elapsed_minutes = (datetime.now() - started_at).total_seconds() / 60
            response["elapsed_minutes"] = round(elapsed_minutes, 1)
            response["estimated_remaining"] = max(0, round(15 - elapsed_minutes, 1))
        
        if status == "failed":
            response["error_message"] = error_message
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get RAG status: {e}")
        return {
            "business_id": business_id,
            "status": "unknown",
            "rag_initialized": rag_initialized_businesses.get(business_id, False)
        }


# Update process_and_refresh_rag_task to update status
async def process_and_refresh_rag_task(business_id: str, file_path: Path):
    """Process RAG with status updates."""
    try:
        logger.info(f"Background task: Processing {file_path} for business {business_id}")
        
        # Do the work
        await refresh_rag(business_id=business_id, doc_path=str(file_path))
        
        # Mark as completed
        rag_initialized_businesses[business_id] = True
        
        pool = await ensure_pool_health()
        async with pool.connection() as conn:
            await conn.execute("""
                UPDATE rag_status
                SET status = 'completed'
                WHERE business_id = %s
            """, (business_id,))
        
        logger.info(f"✅ Background task: RAG completed for {business_id}")
        
    except Exception as e:
        logger.error(f"❌ Background RAG refresh failed for {business_id}: {e}", exc_info=True)
        
        # Mark as failed
        rag_initialized_businesses.pop(business_id, None)
        
        try:
            pool = await ensure_pool_health()
            async with pool.connection() as conn:
                await conn.execute("""
                    UPDATE rag_status
                    SET status = 'failed', error_message = %s
                    WHERE business_id = %s
                """, (str(e), business_id))
        except Exception as status_error:
            logger.error(f"Failed to update error status: {status_error}")

            
@app.get("/rag/status/{business_id}", tags=["Knowledge Base"])
async def get_rag_status(business_id: str):
    """Check if RAG is initialized and ready for a business."""
    return {
        "business_id": business_id,
        "rag_initialized": rag_initialized_businesses.get(business_id, False),
        "status": "ready" if rag_initialized_businesses.get(business_id) else "initializing"
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Send a message to the assistant and get a response."""
    
    try:
        # Ensure database is initialized
        await ensure_db_initialized()
        
        # 🔥 KEY CHANGE: Check if RAG is ready, don't wait for it
        if request.business_id not in rag_initialized_businesses:
            # Try to initialize (non-blocking check)
            try:
                await ensure_rag_initialized(request.business_id)
            except Exception as e:
                # RAG not ready yet - return helpful message
                return ChatResponse(
                    response="Our knowledge base is still being prepared. Please try again in a few moments. You can check status at /rag/status/" + request.business_id,
                    thread_id=request.thread_id or f"{request.business_id}_{request.user_id}_temp",
                    business_id=request.business_id,
                    user_id=request.user_id
                )
        
        # RAG is ready, proceed with chat
        thread_id = request.thread_id or f"{request.business_id}_{request.user_id}_{uuid.uuid4().hex[:8]}"
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": request.user_id,
                "business_id": request.business_id
            }
        }
        
        input_state = {
            "messages": [HumanMessage(content=request.message)]
        }
        
        logger.info(f"Processing message from user {request.user_id}")
        
        result = await asyncio.wait_for(
            graph.ainvoke(input_state, config),
            timeout=30.0
        )
        
        ai_messages = [
            msg for msg in result["messages"] 
            if hasattr(msg, 'content') and msg.__class__.__name__ == 'AIMessage'
        ]
        
        if not ai_messages:
            raise HTTPException(status_code=500, detail="No response generated")
        
        response_text = ai_messages[-1].content
        
        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            business_id=request.business_id,
            user_id=request.user_id
        )
    
    except asyncio.TimeoutError:
        logger.error(f"Chat timeout for user {request.user_id}")
        raise HTTPException(status_code=504, detail="Request timeout - please try again")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@app.post("/rag/initialize", tags=["Knowledge Base"])
async def initialize_rag_endpoint(request: InitRAGRequest):
    """Initialize RAG system for a business."""
    try:
        await ensure_db_initialized()
        await ensure_rag_initialized(request.business_id)
        
        return {
            "status": "success",
            "message": f"RAG initialized for business {request.business_id}",
            "business_id": request.business_id
        }
    
    except Exception as e:
        logger.error(f"RAG initialization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG: {str(e)}")


@app.post("/rag/refresh", tags=["Knowledge Base"])
async def refresh_rag_endpoint(request: RefreshRAGRequest):
    """Refresh RAG system for a business (e.g., when knowledge base is updated)."""
    try:
        await ensure_db_initialized()
        # FIXED: Use doc_path (singular) to match bot.py
        await refresh_rag(business_id=request.business_id, doc_path=request.doc_path)
        rag_initialized_businesses[request.business_id] = True
        
        return {
            "status": "success",
            "message": f"RAG refreshed for business {request.business_id}",
            "business_id": request.business_id
        }
    
    except Exception as e:
        logger.error(f"RAG refresh error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to refresh RAG: {str(e)}")


@app.get("/profile/{business_id}/{user_id}", response_model=UserProfileResponse, tags=["User Profile"])
async def get_user_profile(business_id: str, user_id: str):
    """Retrieve a user's profile information."""
    try:
        await ensure_db_initialized()
        
        pool = await ensure_pool_health()
        
        async with pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT name, location, address, cart, human_active
                FROM user_profiles
                WHERE business_id = %s AND user_id = %s
                """,
                (business_id, user_id)
            )
            
            row = await result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="User profile not found")
            
            # Parse cart from JSON
            import json
            cart_data = None
            if row[3]:
                if isinstance(row[3], str):
                    cart_data = json.loads(row[3])
                else:
                    cart_data = row[3]
            
            return UserProfileResponse(
                business_id=business_id,
                user_id=user_id,
                name=row[0],
                location=row[1],
                address=row[2],
                cart=cart_data,
                human_active=row[4]
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve profile: {str(e)}")


@app.delete("/profile/{business_id}/{user_id}", tags=["User Profile"])
async def delete_user_profile(business_id: str, user_id: str):
    """Delete a user's profile."""
    try:
        await ensure_db_initialized()

        pool = await ensure_pool_health()
        
        async with pool.connection() as conn:
            # Delete from user_profiles table
            await conn.execute(
                "DELETE FROM user_profiles WHERE business_id = %s AND user_id = %s",
                (business_id, user_id)
            )
            
            # Delete from store (LangGraph memory)
            namespace = ("profile", business_id, user_id)
            try:
                items = await store.asearch(namespace)
                for item in items:
                    await store.adelete(namespace, item.key)
            except Exception as store_error:
                logger.warning(f"Error deleting from store: {store_error}")
        
        return {
            "status": "success",
            "message": f"Profile deleted for user {user_id} in business {business_id}"
        }
    
    except Exception as e:
        logger.error(f"Profile deletion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete profile: {str(e)}")


@app.get("/conversations/{business_id}/{user_id}", tags=["Conversations"])
async def get_conversation_history(business_id: str, user_id: str):
    """Retrieve conversation history for a user from the store."""
    try:
        await ensure_db_initialized()
        namespace = ("profile", business_id, user_id)
        
        # Get user memory from store
        memory_data = await store.aget(namespace, "user_memory")
        
        if not memory_data or not getattr(memory_data, "value", None):
            return {
                "business_id": business_id,
                "user_id": user_id,
                "profile": None,
                "message": "No conversation history found"
            }
        
        return {
            "business_id": business_id,
            "user_id": user_id,
            "profile": memory_data.value
        }
    
    except Exception as e:
        logger.error(f"Conversation retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversations: {str(e)}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)