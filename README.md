-----

#WhatsApp Assistant AI Backend

This repository hosts the FastAPI backend for the WhatsApp Assistant, powered by **LangGraph** for conversational state management and **Supabase** (PostgreSQL/pgvector) for persistence and Retrieval-Augmented Generation (RAG).

-----

## 1. Project Overview

The backend provides a stateful chat API that utilizes **PostgreSQL** for storing conversation history (`checkpoints`) and **pgvector** for managing the business's knowledge base.

  * **Framework:** FastAPI (Python)
  * **Conversation Engine:** LangGraph
  * **Database/Vector Store:** Supabase (PostgreSQL + pgvector)
  * **Deployment:** Designed for local development via Uvicorn/ngrok.

-----

## ⚙️ 2. Local Setup & Installation

### A. Environment Variables

Create a file named `.env` in the root directory and fill it with your credentials:

```ini
# --- Supabase Database (PostgreSQL) ---

POSTGRES_URI_POOLER=postgresql://postgres.ifamzlkiifmqgpiuqmsv:nkereuwem@aws-1-eu-north-1.pooler.supabase.com:6543/postgres

# --- Supabase Vector Store (for RAG) ---

POSTGRES_URI=postgresql://postgres.ifamzlkiifmqgpiuqmsv:nkereuwem@aws-1-eu-north-1.pooler.supabase.com:5432/postgres


```

### B. Dependencies

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd WAHA
    ```
2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate 
    ```
3.  **Install Requirements:** (Assuming you have a `requirements.txt` with `fastapi`, `uvicorn`, `langgraph`, `psycopg`, `langchain-supabase`, etc.)
    ```bash
    pip install -r requirements.txt
    ```

-----

## ▶️ 3. Running the Server

### A. Start the FastAPI Service

Run the server locally on port 8000.

```bash
# Recommended command using the reload flag for development:
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### B. Expose to External Teams (Using ngrok)

To allow the frontend team to access your local machine, use ngrok in a separate terminal window:

1.  **Download and set up ngrok** if you haven't already.
2.  **Create the tunnel:**
    ```bash
    ngrok http 8000
    ```
3.  **Note the Public URL:** ngrok will provide a temporary URL (e.g., `https://example-id.ngrok-free.app`). **This is your BASE\_URL.**

-----

## 🔗 4. API Endpoints (For Frontend Team)

The server exposes its full specification at `/openapi.json` and a visual interface at `/docs`. Use the public ngrok URL as the base.

**Base URL Example (Changes on ngrok restart):** `https://9f6113b108d7.ngrok-free.app`

| Endpoint | Method | Path | Description |
| :--- | :--- | :--- | :--- |
| **Chat/Conversation** | `POST` | `/chat` | **Core endpoint.** Sends a user message. Automatically initializes RAG system if needed. |
| **Knowledge Upload** | `POST` | `/upload` | **File Upload.** Accepts `multipart/form-data` to index new documents (PDF, CSV, etc.) for a business. |
| **Health Check** | `GET` | `/health` | Verifies status of API, database, and RAG initialization. |
| **Docs UI** | `GET` | `/docs` | View and test all endpoints directly. |

### Example `/chat` Payload

The `/chat` endpoint requires a JSON body:

```json
{
    "business_id": "JovitPizza",
    "user_id": "whatsapp_user_12345",
    "message": "What are the hours of operation?"
}
```

-----

## 📚 5. Documentation and Data

The full, machine-readable API specification is committed to the repository for client generation:

  * **OpenAPI Schema:** `./docs/openapi.json`

This file defines all required data models, request bodies, and responses.