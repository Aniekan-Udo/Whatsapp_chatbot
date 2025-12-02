# --- Base image ---
FROM python:3.10-slim

# --- Set work directory ---
WORKDIR /app

# --- Install OS dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Copy dependencies ---
COPY requirements.txt .

# --- Install Python dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy all project files ---
COPY . .

# --- Expose port for FastAPI ---
EXPOSE 8000

# --- Start FastAPI server ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
