import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    API_KEY: str
    
    # Database
    POSTGRES_URI: str = "postgresql://postgres:nkereuwem@localhost:5432/chatbot"
    
    # Models
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG Settings
    
    kb_doc_path: str =r"C:/Users/HP/Desktop/WAHA/data/synthetic_restaurant_menu_10000.csv" #"/app/data/synthetic_restaurant_menu_10000.csv"

    # config.py
    #persist_dir: str = "/app/chroma_rag_KB1"  # Absolute path instead of ./
    chunk_size: int = 1500
    chunk_overlap: int = 200
    retriever_k: int = 5
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_prefix = ""  # No prefix
        case_sensitive = False

settings = Settings()