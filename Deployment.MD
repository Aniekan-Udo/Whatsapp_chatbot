Deployment Guide (Render)
This application is configured for easy deployment on Render.

Prerequisites
A Render account.
Your GitHub repository connected to Render.
The following environment variables ready:
POSTGRES_URI: Connection string to your Supabase Vector Store.
POSTGRES_URI_POOLER: Connection string to your Supabase Transaction Pooler (port 6543).
API_KEY: Groq API Key.
OPIK_API_KEY: Opik API Key (for monitoring).
COHERE_API_KEY: Cohere API Key (for embeddings).


Option 1: Blueprints (Recommended)
Go to your Render Dashboard.
Click New + and select Blueprint.
Connect this repository.
Render will auto-detect 
render.yaml
 and prompt you for the required environment variables.
Click Apply.


Option 2: Manual Setup
Create a new Web Service.
Connect your repository.
Select Docker as the Runtime.
Set the Start Command to:
gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
Add the environment variables listed in Prerequisites.
Health Check
Once deployed, visit https://your-app-name.onrender.com/ping to verify the service is running.


