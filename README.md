# WhatsApp Restaurant Assistant

An AI assistant that chats with your customers on **WhatsApp**, answers questions about your menu, takes their orders, and hands them over to your staff when they're ready to pay — 24 hours a day.

---

## What is this?

Imagine a restaurant that gets hundreds of WhatsApp messages a day:
*"Do you have vegetarian options?"*, *"How much is the pepperoni pizza?"*, *"Add two burgers to my order."*

Answering all of those by hand takes a lot of time. This project is a **smart WhatsApp assistant** that does it automatically. A customer sends a normal WhatsApp message, and the assistant replies within seconds like a friendly member of staff would.

It's built for restaurants, but it works for any business with a list of products. You just give it your menu or catalogue.

## What can it do?

| The customer says… | The assistant… |
|---|---|
| "What spicy dishes do you have?" | Searches **your menu** and suggests matching dishes, with prices |
| "Add 2 chicken wraps please" | Adds them to the customer's **cart** |
| "Remove one wrap" / "What's in my cart?" | Updates or shows the cart |
| "My name is Ada, I live at 12 Main St" | **Remembers** the customer's name and delivery address for next time |
| "I'm ready to order" | Checks that it has a delivery address (and asks for one if not), then **passes the order to a human** to finish |

It also:
- **Remembers each conversation**, so customers can come back later and carry on where they left off.
- **Only answers from your own information.** You upload your menu, and the assistant uses that instead of guessing.
- **Supports more than one business.** Each business has its own menu, name and description.
- **Ignores group chats** and only replies to private messages.

---

## How does it work? (the simple version)

```
 Customer sends a WhatsApp message
            │
            ▼
 ┌─────────────────────┐
 │  WhatsApp connector │  (WAHA: links a normal WhatsApp number to the system)
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │   The assistant     │  Understands what the customer wants:
 │   (AI "brain")      │  a question? add to cart? ready to order?
 └─────────────────────┘
       │           │
       ▼           ▼
  Looks up     Updates the cart /
  your menu    remembers customer details
       │           │
       └─────┬─────┘
             ▼
  Writes a friendly reply and sends it back on WhatsApp
```

- **WAHA** is a free tool that connects a regular WhatsApp account to the software. You link it once by scanning a QR code, the same way you'd use WhatsApp Web.
- **The AI brain** is a large language model (Llama 3.3, run through a service called Groq). It reads the message and decides what to do.
- **Your menu** is uploaded as a file (for example a spreadsheet). The assistant searches it to answer questions accurately. A sample menu with 10,000 dishes is included: `synthetic_restaurant_menu_10000.csv`.
- **A database** stores conversations, carts and customer details, so nothing is lost between messages.

---

## What you need before starting

You don't need to be a programmer to understand the setup, but someone comfortable with a computer terminal will need to run it.

1. **A computer or server** with [Docker](https://www.docker.com/products/docker-desktop/) installed. Docker runs the whole system with a couple of commands.
2. **A WhatsApp number** for the business. A spare phone number is best.
3. **A free Groq account** for the AI: <https://console.groq.com>. Create an API key there.
4. **A free Cohere account** so the assistant can search the menu: <https://dashboard.cohere.com>. Create an API key there.
5. **A PostgreSQL database with the `pgvector` extension.** A free [Supabase](https://supabase.com) project works well.
6. *(Optional)* **An Opik account** (<https://www.comet.com/opik>) if you want to monitor how the AI is performing.

> **Tip:** An **API key** is like a password that lets this software use an online service on your behalf. Keep them private and never share or publish them.

---

## Setting it up, step by step

### Step 1 — Download the project

```bash
git clone https://github.com/Aniekan-Udo/Whatsapp_chatbot.git
cd Whatsapp_chatbot
```

### Step 2 — Add your settings

Create a file called `.env` in the project folder and paste this in, replacing the placeholder values with your own:

```ini
# AI (Groq) — your Groq API key
API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# Menu search (Cohere)
COHERE_API_KEY=your_cohere_api_key
COHERE_EMBED_MODEL=embed-english-light-v3.0

# Database (from your Supabase project settings → Database)
POSTGRES_URI=postgresql://user:password@host:5432/postgres
POSTGRES_URI_POOLER=postgresql://user:password@host:6543/postgres

# Optional monitoring
OPIK_API_KEY=your_opik_api_key
```

The `.env` file is private. It's already set up so it **won't** be uploaded to GitHub.

### Step 3 — Start everything

```bash
docker compose up -d --build
```

This starts two things:
- **WAHA** (the WhatsApp connector) at <http://localhost:3000>
- **The assistant** at <http://localhost:8001>

### Step 4 — Connect your WhatsApp number

1. Open <http://localhost:3000> in your browser (the WAHA dashboard).
2. Start the **`default`** session and scan the QR code with the business phone. In WhatsApp, go to **Settings → Linked devices → Link a device**.
3. In the session settings, add a **webhook** pointing to `http://chatbot:8000/webhook/waha` for the **`message`** event. This tells WAHA to pass every new message to the assistant.

### Step 5 — Tell the assistant about your business

Open <http://localhost:8001/docs>. This is a simple web page where you can try every feature without writing code.

1. **Set your business name and description:** use `POST /business/{business_id}/config`. Pick a short ID for your business, e.g. `default`.
2. **Upload your menu:** use `POST /documents/upload` with the same business ID and your menu file (`.csv`, `.pdf`, `.txt` or `.json`, up to 50 MB).

> **Note:** Messages that arrive on WhatsApp are handled under the business ID **`default`**, so use `default` for the business you connect to WhatsApp.

### Step 6 — Try it!

Send a WhatsApp message to the business number from another phone, e.g. *"Hi, what vegetarian dishes do you have?"* You should get a reply within a few seconds.

---

## Putting it online

To keep the assistant running all the time, you can host it online on [Render](https://render.com). The project already includes a Render setup file (`render.yaml`). See **[Deployment.MD](Deployment.MD)** for step-by-step instructions.

---

## Current limitations

- **Ordering hands off to a human.** When a customer is ready to order, the assistant confirms their address and passes them to your staff. It doesn't take payments by itself.
- **One WhatsApp number = one business.** Messages from WhatsApp all use the `default` business. Several businesses can use the chat API, but only one is connected to WhatsApp at a time.
- **WAHA runs without a dashboard password** in the default setup. Add a password and change the default API key before using it on a public server.

---

## For developers

<details>
<summary>Click to expand technical details</summary>

### Tech stack

| Component | Technology |
|---|---|
| API server | FastAPI (served with Gunicorn + Uvicorn workers) |
| Conversation flow | LangGraph (state graph with tool routing) |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Menu search (RAG) | LlamaIndex + Cohere embeddings + PostgreSQL `pgvector` |
| Memory / checkpoints | LangGraph Postgres checkpointer + store; SQLAlchemy (async) |
| WhatsApp | [WAHA](https://waha.devlike.pro/) (WhatsApp HTTP API) |
| Caching | cashews (in-memory or Redis) |
| Rate limiting | slowapi (60 chat requests/min) |
| Logging & metrics | structlog (JSON logs, request IDs), Prometheus client, Opik |

### Project files

| File | What it does |
|---|---|
| `app.py` | FastAPI app: chat, WhatsApp webhook, document upload, business config, health checks |
| `bot.py` | The LangGraph assistant: prompts, cart logic, customer memory, menu search, DB models |
| `whatsapp_service.py` | Sends replies through the WAHA API |
| `monitoring.py` | Structured logging and Prometheus metrics |
| `config.py` | Settings loaded from `.env` |
| `test_whatsapp.py` | Test script for sending WhatsApp messages |
| `whatsapp.ipynb` | Development notebook |
| `synthetic_restaurant_menu_10000.csv` | Sample menu data |
| `Dockerfile`, `docker-compose.yml`, `gunicorn.conf.py`, `render.yaml` | Running and deployment |

### Conversation flow

```
START → chatbot ─┬─ search_menu      → rag_search                 ─┐
                 ├─ add_to_cart      → add_to_cart                 │
                 ├─ remove_from_cart → remove_cart_item            ├─→ chatbot
                 ├─ view_cart        → view_cart                   │
                 ├─ update_profile   → write_memory                │
                 ├─ ready_to_order   → check_address_and_finalize ─┘
                 └─ (plain reply)    → END
```

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Send a message and get the assistant's reply (`live: true` also sends it on WhatsApp) |
| `POST` | `/webhook/waha` | Receives incoming WhatsApp messages from WAHA |
| `POST` / `GET` | `/business/{business_id}/config` | Set / read business name and description |
| `POST` | `/documents/upload` | Upload a menu or knowledge file |
| `GET` / `DELETE` | `/documents/{business_id}` | View / remove a business's document |
| `POST` | `/admin/initialize-rag/{business_id}` | Manually (re)build the menu search index |
| `GET` | `/ping`, `/health`, `/readiness` | Health checks |
| `GET` | `/docs` | Interactive API documentation |

Example `/chat` request:

```json
{
  "business_id": "default",
  "user_id": "+2348000000000",
  "message": "What pizzas do you have under $10?",
  "live": false
}
```

### Running without Docker

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8001
```

You'll still need WAHA running (e.g. `docker compose up -d waha`) and `WAHA_URL=http://localhost:3000` in `.env`.

### All environment variables

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | Yes | Groq API key |
| `POSTGRES_URI` | Yes | Postgres connection (vector store) |
| `POSTGRES_URI_POOLER` | Yes | Postgres pooled connection (checkpoints/memory) |
| `COHERE_API_KEY` | Yes | Cohere API key for embeddings |
| `GROQ_MODEL` | — | Default `llama-3.3-70b-versatile` |
| `COHERE_EMBED_MODEL` | — | Default `embed-english-light-v3.0` |
| `WAHA_URL` | — | Default `http://localhost:3000` (set automatically in Docker) |
| `REDIS_URI` / `CACHE_URL` | — | Cache backend; in-memory if not set |
| `OPIK_API_KEY`, `OPIK_WORKSPACE` | — | LLM monitoring |
| `TEST_WHATSAPP_NUMBER` | — | If set, `live` chat replies go to this number instead |
| `ALLOWED_ORIGINS` | — | Comma-separated CORS origins (default `*`) |
| `PORT` | — | Server port |

</details>
