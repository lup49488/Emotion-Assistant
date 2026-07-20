# Cloudflare Tunnel Public Test Deployment

This project can be exposed for public testing through one public hostname.
Keep the React frontend and FastAPI API under that same hostname:

- `https://chat.example.com/` -> React static files
- `https://chat.example.com/api/...` -> FastAPI

This layout is required by the signed session and CSRF-cookie design. Do not
place the frontend at `app.example.com` and the API at `api.example.com` unless
the cookie/CSRF design is changed first.

## 1. Configure the server environment

Create or update the server-side `.env`. Keep this file private and never put
these values in `frontend/.env*`.

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=replace-with-your-server-side-key
API_SESSION_SECRET=replace-with-a-unique-random-value-at-least-32-characters
API_COOKIE_SECURE=true
API_COOKIE_SAMESITE=lax
API_CORS_ORIGINS=https://chat.example.com
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8000
API_PRELOAD_MODELS=true
```

Replace `chat.example.com` with the hostname configured in Cloudflare. If you
use a different API provider, configure its provider-specific key instead.

## 2. Build and run the frontend locally on the server

The production frontend must call the same public origin, so set the API base
to `/` while building:

```bash
cd /path/to/NLP_Project/frontend
VITE_API_BASE_URL=/ npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

For a long-running deployment, run the preview process under a service manager
or replace it with a dedicated static server such as Nginx. `vite preview` is
appropriate for a small public test but is not a full production web server.

Start FastAPI separately from the repository root:

```bash
python3 -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Verify locally before exposing either service:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/v1/status
curl -I http://127.0.0.1:4173/
```

`/health` is safe for a reverse proxy or uptime monitor. It reports storage,
RAG, warmup state, and aggregate request metrics without exposing credentials,
user data, model prompts, or API keys. Every API response also includes an
`X-Request-ID` header; use that value to correlate a browser failure with the
server log.

Run the same checks used by CI before a manual deployment:

```bash
python3 deployment_check.py --skip-dependencies --smoke-api --frontend-build
```

## 3. Configure Cloudflare Tunnel

In Cloudflare Zero Trust, create or select a named Tunnel and add the following
two published application routes in this order:

| Public hostname | Path | Local service |
| --- | --- | --- |
| `chat.example.com` | `/api/*` | `http://127.0.0.1:8000` |
| `chat.example.com` | `/*` | `http://127.0.0.1:4173` |

The dashboard creates the DNS record for the hostname. If you use a locally
managed tunnel, the equivalent `config.yml` is:

```yaml
tunnel: YOUR_TUNNEL_UUID
credentials-file: /home/YOUR_USER/.cloudflared/YOUR_TUNNEL_UUID.json

ingress:
  - hostname: chat.example.com
    path: ^/api/.*
    service: http://127.0.0.1:8000
  - hostname: chat.example.com
    service: http://127.0.0.1:4173
  - service: http_status:404
```

Validate the rule order before running it:

```bash
cloudflared tunnel ingress validate
cloudflared tunnel ingress rule https://chat.example.com/api/v1/session
cloudflared tunnel ingress rule https://chat.example.com/
```

For a remotely managed tunnel, install it as a service using the token copied
from the Cloudflare dashboard:

```bash
sudo cloudflared service install YOUR_TUNNEL_TOKEN
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

## 4. Public checks

1. Open `https://chat.example.com` and log in.
2. In browser developer tools, confirm that the session and CSRF cookies are
   both `Secure` and sent to the same hostname.
3. Send a chat message and confirm that `POST /api/v1/chat/stream` succeeds.
4. Check the server logs and `systemctl status cloudflared`.
   Search logs by `request_id=<X-Request-ID>` when investigating a failed chat request.

Use a named tunnel for this project. Quick Tunnels (`trycloudflare.com`) are
not suitable because they do not support the application's SSE chat stream.
