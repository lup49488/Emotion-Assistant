# Mindful React Frontend

Standalone React + Vite client for the FastAPI `/api/v1` contract.

## Run locally

1. Start the FastAPI service from the repository root:
   `D:\software\Python312\python.exe -m uvicorn api_server:app --host 127.0.0.1 --port 8000`
2. In this directory run `npm.cmd install` once, then `npm.cmd run dev`.
3. Open `http://127.0.0.1:5173`.

By default Vite connects to `http://127.0.0.1:8000`. Copy `.env.example` to
`.env.local` and change `VITE_API_BASE_URL` when the API is hosted elsewhere.

The browser only uses the signed session and CSRF cookies issued by FastAPI.
Never place an API key, access password, or `API_SESSION_SECRET` in a Vite
environment file, because `VITE_*` values are included in the browser bundle.
