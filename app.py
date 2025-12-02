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


import uvicorn
import os
import logging
import uuid
from typing import Optional, List, Dict, Any, Set
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from bot import initialize_graph,setup_database,get_pool,initialize_rag,start_file_monitoring,refresh_rag,store,saver

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
rag_initialized = {}
file_observers = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown."""
    global graph
    logger.info("=" * 60)
    logger.info("Starting FastAPI application...")
    logger.info("=" * 60)
    
    try:
        # Initialize database and graph
        logger.info("Step 1: Initializing database and graph...")
        graph = await initialize_graph()
        logger.info("✅ Graph initialized successfully")
        
        # Note: setup_database() is already called inside initialize_graph()
        # so we don't need to call it again here
        
        # Verify database connection
        logger.info("Step 2: Verifying database connection...")
        pool = await get_pool()
        async with pool.connection() as conn:
            result = await conn.execute("SELECT COUNT(*) FROM user_profiles")
            count = (await result.fetchone())[0]
            logger.info(f"✅ Database connected - Found {count} user profiles")
            
            # Check other tables
            for table in ['store', 'checkpoints', 'writes', 'business_documents']:
                result = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                table_count = (await result.fetchone())[0]
                logger.info(f"  - {table}: {table_count} records")
        
        # Ensure the upload path exists
        if not UPLOAD_ROOT_PATH.exists():
            UPLOAD_ROOT_PATH.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created upload directory: {UPLOAD_ROOT_PATH}")
        else:
            logger.info(f"✅ Upload directory exists: {UPLOAD_ROOT_PATH}")
        
        logger.info("=" * 60)
        logger.info("✅ APPLICATION READY TO RECEIVE REQUESTS")
        logger.info("=" * 60)
        logger.info(f"API Docs: http://localhost:8000/docs")
        logger.info(f"Health Check: http://localhost:8000/health")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ STARTUP FAILED: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        raise
    
    yield  # Application runs here
    
    # Cleanup
    logger.info("=" * 60)
    logger.info("Shutting down application...")
    logger.info("=" * 60)
    
    # Stop file observers
    for business_id, observer in file_observers.items():
        logger.info(f"Stopping file observer for business {business_id}...")
        observer.stop()
        observer.join()
    
    # Close database pool
    pool = await get_pool()
    if pool and not pool.closed:
        await pool.close()
        logger.info("✅ Database pool closed")
    
    logger.info("=" * 60)
    logger.info("✅ Application shutdown complete")
    logger.info("=" * 60)

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


# Pydantic models
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


# Helper functions
async def ensure_rag_initialized(business_id: str, doc_path: Optional[str] = None):
    """Ensure RAG is initialized for a business before processing messages."""
    global rag_initialized, file_observers
    
    if business_id not in rag_initialized:
        logger.info(f"Initializing RAG for business {business_id}...")
        try:
            await initialize_rag(business_id=business_id, doc_path=doc_path)
            rag_initialized[business_id] = True
            
            # Start file monitoring for this business
            if business_id not in file_observers:
                observer = start_file_monitoring(business_id)
                if observer:
                    file_observers[business_id] = observer
            
            logger.info(f"✅ RAG initialized for business {business_id}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG for business {business_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize knowledge base: {str(e)}"
            )


def allowed_file(filename: str) -> bool:
    """Checks if a filename has an allowed extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def process_and_refresh_rag_task(business_id: str, file_path: Path):
    """Synchronous task to be run in the background to refresh RAG."""
    logger.info(f"Background task: Processing {file_path} for business {business_id}")
    try:
        asyncio.run(refresh_rag(business_id=business_id, doc_path=str(file_path)))
        logger.info(f"Background task: RAG refreshed successfully for {business_id} with file {file_path.name}")
    except Exception as e:
        logger.error(f"Background RAG refresh failed for {business_id} with {file_path.name}: {e}")


# API Endpoints

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
    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"
    
    return {
        "status": "healthy" if db_status == "connected" else "unhealthy",
        "database": db_status,
        "rag_systems": rag_initialized
    }


@app.post("/upload", tags=["Knowledge Base"])
async def upload_document(
    business_id: str = Form(..., description="Unique identifier for the business to assign the document to"),
    document_file: UploadFile = File(..., description="Document or PDF file to upload and refresh RAG with"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Uploads a document/PDF from the user's device and triggers a RAG refresh 
    for the specified business in the background.
    """
    
    filename = document_file.filename
    
    if not filename or not allowed_file(filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type or no file selected. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Create a business-specific directory
    business_upload_dir = UPLOAD_ROOT_PATH / business_id
    if not business_upload_dir.exists():
        business_upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the file with a unique name
    file_extension = Path(filename).suffix
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = business_upload_dir / unique_filename

    try:
        content = await document_file.read()
        await asyncio.to_thread(file_path.write_bytes, content)
        logger.info(f"File saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"There was an error saving the file: {e}")
    
    # Add the RAG refresh task to the background
    background_tasks.add_task(process_and_refresh_rag_task, business_id, file_path)
    
    return JSONResponse(content={
        "status": "processing",
        "message": f"Document '{filename}' uploaded successfully. RAG refresh initiated in the background for business {business_id}.",
        "file_id": unique_filename,
        "business_id": business_id
    }, status_code=202)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the assistant and get a response.
    """
    try:
        # Ensure RAG is initialized for this business
        await ensure_rag_initialized(request.business_id)
        
        # Generate thread_id if not provided
        thread_id = request.thread_id or f"{request.business_id}_{request.user_id}_{uuid.uuid4().hex[:8]}"
        
        # Prepare config
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": request.user_id,
                "business_id": request.business_id
            }
        }
        
        # Create input state with HumanMessage
        input_state = {
            "messages": [HumanMessage(content=request.message)]
        }
        
        logger.info(f"Processing message from user {request.user_id} (business: {request.business_id})")
        
        # Invoke the graph
        result = await graph.ainvoke(input_state, config)
        
        # Extract the last AI message
        ai_messages = [msg for msg in result["messages"] if hasattr(msg, 'content') and msg.__class__.__name__ == 'AIMessage']
        
        if not ai_messages:
            raise HTTPException(status_code=500, detail="No response generated")
        
        response_text = ai_messages[-1].content
        
        logger.info(f"Generated response for user {request.user_id}")
        
        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            business_id=request.business_id,
            user_id=request.user_id
        )
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@app.post("/rag/initialize", tags=["Knowledge Base"])
async def initialize_rag_endpoint(request: InitRAGRequest):
    """
    Initialize RAG system for a specific business.
    """
    try:
        await ensure_rag_initialized(request.business_id, request.doc_path)
        
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
    """
    Refresh RAG system for a business (e.g., when knowledge base is updated).
    """
    try:
        await refresh_rag(business_id=request.business_id, doc_path=request.doc_path)
        
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
    """
    Retrieve a user's profile information.
    """
    try:
        pool = await get_pool()
        
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
    """
    Delete a user's profile and conversation history.
    """
    try:
        pool = await get_pool()
        
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
async def get_conversation_history(business_id: str, user_id: str, thread_id: Optional[str] = None):
    """
    Retrieve conversation history for a user from the store.
    """
    try:
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


# Add this to the BOTTOM of your app.py file (replace existing if __name__ == "__main__")
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
# if __name__ == "__main__":
#     import sys
#     import asyncio
#     import uvicorn
    
#     # Set event loop policy BEFORE creating the loop
#     if sys.platform == 'win32':
#         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
#     # Create and set the event loop with the correct policy
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
    
#     print("✅ Event loop policy set and loop created with WindowsSelectorEventLoopPolicy")
    
#     # Run uvicorn programmatically with our loop
#     config = uvicorn.Config(
#         "app:app",
#         host="0.0.0.0",
#         port=8000,
#         loop="asyncio",  # Use asyncio loop type
#         reload=False
#     )
#     server = uvicorn.Server(config)
    
#     # Run with our event loop
#    loop.run_until_complete(server.serve())