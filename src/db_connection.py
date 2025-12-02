from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from dotenv import load_dotenv
import os
import logging
from config import settings

load_dotenv()
logger = logging.getLogger(__name__)

POSTGRES_URI = settings.POSTGRES_URI
if not POSTGRES_URI:
    raise ValueError("POSTGRES_URI not set")


# After all imports and before any functions
store = None
saver = None

# Make sure this is BEFORE the setup_database function
# -----------------------------
# Connect to DB
# -----------------------------

_pool: AsyncConnectionPool = None

async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=POSTGRES_URI, 
            min_size=1,
            max_size=10, 
            open=False  # Don't auto-open
        )
        await _pool.open()  # Explicitly open
    return _pool

# Initialize these properly in setup_database
store = None
saver = None

async def setup_database():
    global store, saver
    print("=== SETUP_DATABASE CALLED ===")
    pool = await get_pool()
    print(f"Pool created: {pool}")
    if store is None:
        print("Creating new store...")
        store = AsyncPostgresStore(pool)
        await store.setup()
        print(f"Store created and setup: {store}")
    else:
        print(f"Store already exists: {store}")
    if saver is None:
        print("Creating new saver...")
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        print(f"Saver created and setup: {saver}")
    logger.info("Database setup completed")
