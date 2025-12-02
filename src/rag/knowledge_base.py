import os
import hashlib
import logging
import asyncio
from typing import Optional, Any, List
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings
from sqlalchemy import text

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Manages the RAG knowledge base asynchronously using PostgreSQL with pgvector.
    All data is persisted in PostgreSQL - no local file storage needed.
    """
    
    def __init__(self):
        self._vector_store: Optional[PGVector] = None
        self._retriever: Optional[Any] = None
        self._file_hash: Optional[str] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._is_initialized = False
        self._collection_name = "kb1_collection"
        self._async_engine: Optional[AsyncEngine] = None
    
    async def _get_async_engine(self) -> AsyncEngine:
        """Create or return the async SQLAlchemy engine."""
        if self._async_engine is None:
            # Convert postgresql:// to postgresql+asyncpg://
            async_uri = settings.POSTGRES_URI.replace('postgresql://', 'postgresql+asyncpg://')
            
            # Create engine with execution options to disable prepared statements
            self._async_engine = create_async_engine(
                async_uri,
                execution_options={
                    "postgresql_prepared_statement_cache_size": 0
                },
                pool_pre_ping=True
            )
            logger.info("Created async database engine")
        return self._async_engine
    
    # --- ASYNC HELPERS ---

    async def _aget_file_hash(self, path: str) -> str:
        """Compute file hash to detect changes by running I/O in a thread."""
        def sync_hash():
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        
        file_hash = await asyncio.to_thread(sync_hash)
        logger.debug(f"Computed file hash for {path}: {file_hash}")
        return file_hash
    
    async def _get_stored_file_hash(self) -> Optional[str]:
        """Retrieve the stored file hash from PostgreSQL metadata."""
        try:
            if not self._vector_store:
                return None
            
            # Access the collection metadata asynchronously
            async def get_hash():
                try:
                    # For async PGVector, we need to access metadata differently
                    # The collection object should support async operations
                    if hasattr(self._vector_store._collection, 'aget_metadata'):
                        metadata = await self._vector_store._collection.aget_metadata()
                    else:
                        # Fallback to sync access in thread
                        metadata = await asyncio.to_thread(
                            lambda: self._vector_store._collection.metadata
                        )
                    return metadata.get('file_hash') if metadata else None
                except Exception as e:
                    logger.debug(f"Error getting metadata: {e}")
                    return None
            
            stored_hash = await get_hash()
            return stored_hash
        except Exception as e:
            logger.debug(f"Could not retrieve stored hash: {e}")
            return None
    
    async def _check_collection_exists(self) -> bool:
        """Check if the collection already exists in PostgreSQL."""
        try:
            engine = await self._get_async_engine()
            
            # Create a temporary PGVector instance with async engine
            temp_store = PGVector(
                collection_name=self._collection_name,
                connection=engine,
                embeddings=self._embeddings,
                use_jsonb=True
            )
            
            # Check if collection has documents (run count in thread if sync)
            count = await asyncio.to_thread(
                lambda: temp_store._collection.count()
            )
            logger.info(f"Found existing collection with {count} documents")
            return count > 0
        except Exception as e:
            logger.debug(f"Collection check failed (likely doesn't exist): {e}")
            return False
    
    async def ainitialize(self, doc_path: Optional[str] = None, force_rebuild: bool = False) -> None:
        """Asynchronously initialize the knowledge base."""
        if self._is_initialized and not force_rebuild:
            logger.info("Knowledge base already initialized")
            return
        
        document_path = doc_path or settings.kb_doc_path
        
        logger.info(f"Initializing knowledge base with document: {document_path}")
        logger.info(f"Using PostgreSQL for vector storage")
        
        try:
            # Get async engine
            engine = await self._get_async_engine()
            
            # Ensure vector extension exists first
            try:
                async with engine.begin() as conn:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    logger.info("Vector extension ensured")
            except Exception as e:
                logger.debug(f"Vector extension setup: {e}")
            
            # Load Embeddings (CPU-bound, run in a thread)
            def load_embeddings():
                return HuggingFaceEmbeddings(
                        model_name=settings.embedding_model,
                        model_kwargs={'device': 'cpu'},  # or 'cuda'
                        encode_kwargs={'normalize_embeddings': True}
                    )
                
            
            self._embeddings = await asyncio.to_thread(load_embeddings)
            logger.info(f"Embeddings model '{settings.embedding_model}' loaded on CPU")
            
            # Check if document exists
            if not os.path.exists(document_path):
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Calculate current file hash
            current_hash = await self._aget_file_hash(document_path)
            
            # Check for existing collection
            if not force_rebuild:
                collection_exists = await self._check_collection_exists()
                
                if collection_exists:
                    logger.info("Loading existing knowledge base from PostgreSQL...")
                    
                    # Load existing vector store with async engine
                    self._vector_store = PGVector(
                        collection_name=self._collection_name,
                        connection=engine,
                        embeddings=self._embeddings,
                        use_jsonb=True
                    )
                    
                    # Check if file has changed
                    stored_hash = await self._get_stored_file_hash()
                    
                    if stored_hash and stored_hash == current_hash:
                        logger.info("File unchanged, using existing knowledge base")
                        
                        self._retriever = self._vector_store.as_retriever(
                            search_kwargs={"k": settings.retriever_k}
                        )
                        
                        self._file_hash = current_hash
                        self._is_initialized = True
                        logger.info("✅ Loaded existing knowledge base from PostgreSQL")
                        return
                    else:
                        logger.info("File has changed, rebuilding knowledge base...")
            
            # Build from scratch
            logger.info("Building knowledge base from scratch in PostgreSQL...")
            await self._abuild_knowledge_base(document_path, current_hash)
            self._file_hash = current_hash
            self._is_initialized = True
            
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}", exc_info=True)
            raise
    
    async def _abuild_knowledge_base(self, document_path: str, file_hash: str) -> None:
        """Asynchronously build knowledge base from documents."""
        logger.info("Building knowledge base from scratch...")
        
        engine = await self._get_async_engine()
        
        # Initialize splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        logger.info(f"Using chunk size: {settings.chunk_size}, chunk overlap: {settings.chunk_overlap}")
        
        # Load documents
        try:
            if document_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(document_path)
            elif document_path.lower().endswith(".docx"):
                loader = Docx2txtLoader(document_path)
            elif document_path.lower().endswith(".csv"):
                loader = CSVLoader(file_path=document_path, encoding="utf-8-sig")
            else:
                raise ValueError(f"Unsupported file format: {document_path}")
            
            logger.info(f"Loading documents using loader: {loader.__class__.__name__}")
            
            # Document loading is I/O-intensive, run in a thread
            docs: List[Document] = await asyncio.to_thread(loader.load)
            logger.info(f"Loaded {len(docs)} documents from {document_path}")
            
            if not docs:
                raise ValueError(f"No documents were loaded from {document_path}. The file may be empty or corrupted.")
        
        except Exception as e:
            logger.error(f"Failed to load documents: {e}", exc_info=True)
            raise
        
        # Split documents
        try:
            doc_chunks = await asyncio.to_thread(splitter.split_documents, docs)
            logger.info(f"Split documents into {len(doc_chunks)} chunks")
            
            if not doc_chunks:
                error_msg = (
                    f"Document splitting produced zero chunks from {document_path}. "
                    f"This could mean: (1) the document is empty, (2) the content is too short, "
                    f"or (3) the chunk_size ({settings.chunk_size}) is too large."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            logger.error(f"Error during document splitting: {e}", exc_info=True)
            raise
        
        # Create vector store in PostgreSQL
        try:
            logger.info(f"Creating PostgreSQL vector store with collection '{self._collection_name}'")
            
            # Drop existing collection if it exists (using async method)
            try:
                await PGVector.adrop_collection(
                    collection_name=self._collection_name,
                    connection=engine
                )
                logger.info("Dropped existing collection")
            except Exception as e:
                logger.debug(f"No existing collection to drop: {e}")
            
            # Create new vector store with documents using ASYNC method
            self._vector_store = await PGVector.afrom_documents(
                documents=doc_chunks,
                embedding=self._embeddings,
                collection_name=self._collection_name,
                connection=engine,
                use_jsonb=True,
                pre_delete_collection=False,  # We already dropped it
                collection_metadata={"file_hash": file_hash}  # Store hash for change detection
            )
            logger.info(f"✅ Vector store created in PostgreSQL with {len(doc_chunks)} documents")
            
            # Create retriever
            self._retriever = self._vector_store.as_retriever(
                search_kwargs={"k": settings.retriever_k}
            )
            logger.info("✅ Knowledge base setup complete")
        
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}", exc_info=True)
            raise
    
    # --- ASYNC SEARCH METHOD ---

    async def asearch(self, query: str):
        """Asynchronously search the knowledge base."""
        if self._retriever is None:
            if not self._is_initialized:
                await self.ainitialize()

            if self._retriever is None:
                raise RuntimeError("Knowledge base not initialized and failed to initialize during search.")

        # Use async retrieval
        return await self._retriever.ainvoke(query)
    
    # --- ASYNC REFRESH ---
    
    async def arefresh(self, doc_path: Optional[str] = None) -> None:
        """Asynchronously refresh the knowledge base."""
        logger.info("🔄 Refreshing knowledge base...")
        self._vector_store = None
        self._retriever = None
        self._file_hash = None
        self._is_initialized = False
        await self.ainitialize(doc_path, force_rebuild=True)
        logger.info("✅ Knowledge base refreshed!")
    
    # --- ASYNC CLEANUP ---
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up knowledge base resources...")
        self._vector_store = None
        self._retriever = None
        self._embeddings = None
        self._is_initialized = False
        
        # Close the async engine
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None
            logger.info("Async engine disposed")

# Singleton instance
kb = KnowledgeBase()