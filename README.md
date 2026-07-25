# Serenova

Serenova is an emotion-aware chatbot project with three main surfaces:

- a Gradio desktop-style GUI in `Web_GUI.py`
- a FastAPI HTTP API in `api_server.py`
- a React/Vite frontend in `frontend/`

The application combines provider-backed or local Hugging Face chat models, emotion-aware memory, conversation archives, mood check-ins, knowledge-base RAG, and style-reference RAG.

## Live Site

The deployed app is available at:

[https://chat.serenova.dev](https://chat.serenova.dev)

## Repository Layout

```text
.
|-- api_server.py              # FastAPI adapter and versioned /api/v1 routes
|-- Web_GUI.py                 # Gradio user interface
|-- chatbot.py                 # Chat orchestration, memory, safety, RAG/style injection
|-- llm_providers.py           # Local HF and OpenAI-compatible provider streaming
|-- knowledge_store.py         # Knowledge-base ingestion, retrieval, quality checks
|-- style_store.py             # Style-reference ingestion and retrieval
|-- api_contracts.py           # Pydantic API request/response models
|-- deployment_check.py        # Repeatable local deployment checks
|-- frontend/                  # React + Vite client
|-- knowledge_base/            # Shared RAG documents and derived index files
|-- style_base/                # Style reference documents and derived index files
|-- users/                     # Local per-user data when STORAGE_BACKEND=json
`-- tests: test_*.py           # Pytest coverage
```

## Requirements

- Python 3.10+
- Node.js/npm for the React frontend
- A model provider API key, unless you run `LLM_PROVIDER=local_hf`

Install Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU PyTorch, install the matching CUDA wheel before installing the rest of the requirements. The checked-in `requirements.txt` uses the plain `torch==2.5.1` spec, which installs the CPU wheel by default.

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## Configuration

Copy `.env.example` to `.env` and fill only the values you need:

```powershell
Copy-Item .env.example .env
```

Common provider settings:

```env
LLM_PROVIDER=openai_compatible
LLM_API_BASE_URL=https://api.deepseek.com
LLM_API_KEY=replace-with-your-key
LLM_API_MODEL=deepseek-chat
```

Provider-specific alternatives are also supported:

- `DEEPSEEK_API_KEY` with `LLM_PROVIDER=deepseek`
- `OPENAI_API_KEY` with `LLM_PROVIDER=openai`
- `OPENROUTER_API_KEY` with `LLM_PROVIDER=openrouter`
- `LLM_PROVIDER=local_hf` to load `CHAT_MODEL_NAME` locally

Keep secrets in `.env`, `.env.local`, or the server environment. Do not put API keys in `frontend/.env*`, because `VITE_*` variables are bundled into browser JavaScript.

For the FastAPI signed-cookie flow, set a stable secret before real use:

```env
API_SESSION_SECRET=replace-with-at-least-32-random-characters
```

## Run Locally

### Gradio GUI

```powershell
python Web_GUI.py
```

By default, the GUI listens on `http://127.0.0.1:7860`.

### FastAPI

```powershell
python api_server.py
```

By default, the API listens on `http://127.0.0.1:8000`.

Useful endpoints:

- `GET /health`
- `GET /api/v1/status`
- `GET /api/v1/contract`
- `POST /api/v1/auth/login`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`
- `GET /api/v1/rag/status`

The API chat request fields `use_knowledge` and `use_style` default to `false`. API clients must send them as `true` when they want retrieved knowledge/style context injected into the prompt.

### React Frontend

Start FastAPI first, then:

```powershell
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The frontend reads `VITE_API_BASE_URL` from `frontend/.env.local`; the example default points to `http://127.0.0.1:8000`.

## User Access

The app uses a per-user access key. In the GUI, enter a User ID and access key, then save/verify it. The FastAPI frontend authenticates with `/api/v1/auth/login`, then uses the signed session cookie plus CSRF token issued by the API.

User data is stored in:

- local JSON files under `users/` when `STORAGE_BACKEND=json`
- SQLite when `STORAGE_BACKEND=sqlite`

Use `migrate_storage.py` before switching an existing JSON workspace to SQLite.

## Knowledge RAG

Knowledge-base source documents live under:

```text
knowledge_base/documents/
```

Supported source formats include `.txt`, `.md`, `.markdown`, `.csv`, `.json`, `.pdf`, and `.docx`. PDF and Word ingestion require `pypdf` and `python-docx`.

Rebuild the knowledge index after adding or removing documents:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "import knowledge_store; print(knowledge_store.rebuild_knowledge_index()); print(knowledge_store.knowledge_status())"
```

The API also exposes administrator-protected RAG management routes for uploads, rebuilds, search diagnostics, quality reports, feedback, and evaluation jobs.

## Style RAG

Style-reference source documents live under:

```text
style_base/documents/
```

Supported formats include `.txt`, `.md`, `.markdown`, `.csv`, `.json`, and `.jsonl`.

Rebuild the style index after changing style documents:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "import style_store; print(style_store.rebuild_style_index()); print(style_store.style_status())"
```

Archived raw style material can be kept under `style_base/archive/`; it is preserved for reference but excluded from default retrieval.

## Testing

Run Python tests:

```powershell
python -m pytest -q
```

Run focused API/RAG tests:

```powershell
python -m pytest test_knowledge_store.py test_api_server.py -q
```

Run frontend checks:

```powershell
cd frontend
npm run lint
npm run build
npm run test:e2e
```

Run the deployment checklist:

```powershell
python deployment_check.py --skip-dependencies --smoke-api --frontend-build
```

Add `--tests` to include the full pytest suite and `--smoke-web` to launch and probe the Gradio app.

## Public Test Deployment

See `DEPLOYMENT_CLOUDFLARE.md` for the Cloudflare Tunnel layout. The recommended public test shape is one hostname serving both the React app and FastAPI:

- `/api/*` -> FastAPI
- `/*` -> React static frontend

Set `API_PUBLIC_MODE=true`, `API_COOKIE_SECURE=true`, a stable `API_SESSION_SECRET`, and explicit `API_TRUSTED_HOSTS` before exposing the service.

## Notes For Windows And Chinese Text

When inspecting Chinese source files or Markdown from PowerShell, prefer UTF-8-aware commands:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "from pathlib import Path; print(Path('style_store.py').read_text(encoding='utf-8')[:500])"
```

This avoids terminal mojibake that can make valid UTF-8 text look corrupted.

## Security Notes

- Do not commit `.env`, `.env.local`, API keys, access keys, or session secrets.
- Do not expose the GUI/API publicly without HTTPS, secure cookies, host allowlisting, and a stable session secret.
- Keep browser-facing Vite variables free of secrets.
- RAG and style libraries are shared application data; deleting a user account does not delete shared corpus files.
