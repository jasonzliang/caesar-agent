# Caesar Web Server

A web GUI for the Caesar autonomous AI research agent. Submit a query,
watch the knowledge graph grow live, read the cited final artifact.

Repo layout:

    web_server/
      api/   FastAPI backend (Python 3.10–3.13) — wraps CaesarAgent.explore()
      ui/    Next.js 15 frontend (Node 20+)     — single-origin proxy for /api/*

The browser only talks to port 3000. Next.js rewrites /api/* to the FastAPI
process server-side, so there's no CORS pain even from a remote laptop.

Python version: 3.10–3.13 only (the repo's `requires-python` is
`>=3.10,<3.14`). chromadb 1.5.2 — the version pinned into existing deployment
venvs, though `requirements.txt` no longer pins it — uses
pydantic.v1.BaseSettings, which breaks on 3.14+. If python3 is 3.14, install
3.13 first (`uv python install 3.13`) and create the venv from that
interpreter.


--------------------------------------------------------------------------
## Two ways to run it

  FOREGROUND (./launch.sh)
    For local dev, demos, one-off runs. Dies when the terminal closes.

  AS A SERVICE (./install-service.sh)
    For long-running deploys. Autostarts at boot, restarts on crash.
    Linux uses a systemd --user unit; macOS uses a launchd LaunchAgent.

Both paths use launch.sh under the hood and accept the same env vars.


--------------------------------------------------------------------------
## Path 1 — Foreground

Start:

    cd web_server
    export OPENAI_API_KEY=sk-...
    ./launch.sh

Open http://localhost:3000 (or http://<server-lan-ip>:3000 from another
laptop). uvicorn binds :8090, Next.js binds :3000, both on 0.0.0.0 — except
with --password or --public, where the API drops to 127.0.0.1 and is reachable
only through the UI proxy. Caesar's ChromaDB subprocess listens on
localhost:8091 (CAESAR_CHROMA_PORT), persisting under CAESAR_WEB_DATA_DIR/chroma.

To restart: Ctrl-C and re-run ./launch.sh. If you backgrounded the launcher and have no terminal to Ctrl-C in (CI, tool runner, etc.), stop with `kill $(lsof -ti:3000,8090)`.

Common variations:

    ./launch.sh --password 's3cret'          # require login (binds API to 127.0.0.1)
    CAESAR_PASSWORD='s3cret' ./launch.sh     # same, but keeps it out of `ps` output
    ./launch.sh --public                     # public bring-your-own-key mode (also binds API to 127.0.0.1)
    CAESAR_DRY_RUN=1 ./launch.sh             # synthetic 5s run, no LLM calls — for UI dev
    API_PORT=9000 UI_PORT=3001 ./launch.sh   # alternate ports

`--password` is readable by any local account via `/proc/<pid>/cmdline` for as
long as the process lives, so prefer `CAESAR_PASSWORD` (or the `DEMO_PASSWORD`
alias, which is the name the app itself reads) on a shared host.


--------------------------------------------------------------------------
## Path 2 — Install as a service

install-service.sh autodetects your OS. Neither path touches root.

Install:

    cd web_server
    ./install-service.sh --password 's3cret'   # install + start with auth
    ./install-service.sh --public              # install + start in public BYO-key mode
    ./install-service.sh                       # install + start, no auth

The password is written into the unit as `Environment=CAESAR_PASSWORD` (mode
0600) rather than passed on the command line, so it stays out of `ps`.

The unit sources ~/.bashrc then ~/.zshrc before running, so your shell's
LLM API keys reach the service. On Linux the installer also runs
`loginctl enable-linger` so the service starts before login (warns if it
needs sudo on your distro).

Multiple instances on one host (Linux/systemd only) need an id and explicit
ports; launch.sh derives `.logs-b`, `.next-b` and `api/data-b` from the id, and
the unit is named `caesar-web-b.service`:

    ./install-service.sh --instance-id b \
        --api-port 8092 --ui-port 3001 --chroma-port 8093 --password 'creative'
    ./install-service.sh --instance-id b --uninstall

### Day-to-day — Linux (systemd --user)

    Start          systemctl --user start caesar-web
    Stop           systemctl --user stop caesar-web
    Restart        systemctl --user restart caesar-web
    Status         systemctl --user status caesar-web
    Live logs      journalctl --user -u caesar-web -f
    Unit file      ~/.config/systemd/user/caesar-web.service

### Day-to-day — macOS (launchd)

    Start          launchctl kickstart gui/$(id -u)/com.caesar.web
    Stop           launchctl bootout gui/$(id -u)/com.caesar.web
    Restart        launchctl kickstart -k gui/$(id -u)/com.caesar.web
    Status         launchctl print gui/$(id -u)/com.caesar.web
    Live logs      tail -f web_server/.logs/launchd.log
    Unit file      ~/Library/LaunchAgents/com.caesar.web.plist

Application logs (uvicorn, Next.js) always land in web_server/.logs/api.log
and web_server/.logs/ui.log on both platforms:

    tail -f web_server/.logs/api.log web_server/.logs/ui.log

### After pulling code changes

`launch.sh` builds the Next.js UI and starts a fresh FastAPI process every
time it runs, so restarting the service is enough to pick up backend code,
frontend code, preset YAML, and dependency changes already present in the
checkout.

If you are running in foreground mode instead of a service, restart after
code changes with Ctrl-C and then `./launch.sh`.

Linux:

    cd /path/to/rome
    git pull --ff-only
    systemctl --user restart caesar-web
    systemctl --user status caesar-web
    curl http://127.0.0.1:8090/health

macOS:

    cd /path/to/rome
    git pull --ff-only
    launchctl kickstart -k gui/$(id -u)/com.caesar.web
    launchctl print gui/$(id -u)/com.caesar.web
    curl http://127.0.0.1:8090/health

Watch `web_server/.logs/ui-build.log` during restart if you changed frontend
code. The server is ready when the service is running and `/health` returns
`{"ok":true}`.

### Change launch options (e.g. update the password)

    ./install-service.sh --uninstall
    ./install-service.sh --password 'new-password'

### Uninstall

    ./install-service.sh --uninstall


--------------------------------------------------------------------------
## Configuration

All env vars are optional; defaults are sensible for a single-host demo.

    API_HOST              0.0.0.0   Backend bind addr. Forced to 127.0.0.1 with --password or --public.
    API_PORT              8090      Backend port.
    UI_PORT               3000      Frontend port.
    CAESAR_CHROMA_PORT    8091      ChromaDB subprocess port. Must differ between
                                    two caesar-web instances on the same host.
    CAESAR_WEB_DATA_DIR   api/data/ SQLite DB + per-run artifacts live here.
    CAESAR_WEB_LOGS_DIR   .logs/    api.log / ui.log / ui-build.log.
    CAESAR_INSTANCE_ID    (unset)   [a-z0-9][a-z0-9_-]{0,31}. Suffixes the data,
                                    logs and .next dirs so instances don't collide.
    CAESAR_PASSWORD       (unset)   Login password; DEMO_PASSWORD is an alias.
    PUBLIC_MODE           0         Set 1 (or pass --public) for bring-your-own-key mode.
    CAESAR_DRY_RUN        0         Set 1 for synthetic runs (no LLM calls, ~5s).
    CAESAR_MAX_CONCURRENT 8         Refuse new runs past this in-flight count (429).
    LOG_LEVEL             INFO      Use DEBUG for verbose SSE / job logs.

    OPENAI_API_KEY        required when CAESAR_DRY_RUN=0 and PUBLIC_MODE=0
    ANTHROPIC_API_KEY     optional
    GOOGLE_API_KEY        optional
    BRAVE_API_KEY         optional

In public mode launch.sh unsets OPENAI_API_KEY, CHROMA_OPENAI_API_KEY,
ANTHROPIC_API_KEY, GOOGLE_API_KEY and OPENROUTER_API_KEY before starting the
API: the server holds no key of its own and every submission must carry the
caller's.

Preset ids in `web_server/api/app/config.py` are persisted in SQLite
(`runs.preset`). If you rename a preset id, either keep a compatibility alias
in code or migrate existing rows in `CAESAR_WEB_DATA_DIR/caesar_web.sqlite`;
changing only the UI label is safe.


--------------------------------------------------------------------------
## Remote access

From another machine on the same LAN: open http://<host-ip>:3000. Both
processes bind 0.0.0.0 by default, so the only thing to check is the
host's firewall:

    Linux:  sudo ufw allow 3000/tcp
    macOS:  System Settings → Network → Firewall → Allow ingress on TCP 3000

### Public HTTPS via Tailscale Funnel (recommended)

The simplest way to expose the server to the public internet with a
managed HTTPS cert and no port forwarding. This is how the live demo
host is set up.

One-time setup:

  1. Install Tailscale and authenticate:  https://tailscale.com/download
  2. In the Tailscale admin console: enable HTTPS (DNS settings) and
     grant the Funnel capability to the device (Access controls).

Enable the funnel:

    tailscale funnel --bg 3000

The server is now reachable at https://<machine>.<tailnet>.ts.net. The
config persists across reboots because tailscaled runs as a system
service; no extra autostart wiring needed.

Verify / inspect:

    tailscale funnel status

For a custom domain (e.g. caesar.example.com), add a CNAME from your
DNS pointing at the .ts.net hostname.

### Alternative: Caddy + Let's Encrypt

If you can't or don't want to use Tailscale, run Caddy in front:

    caesar.example.com {
        reverse_proxy localhost:3000
    }

Caddy provisions a Let's Encrypt cert automatically and terminates TLS
on port 443. Requires the host to be publicly reachable on ports 80/443
(port forwarding / firewall).


--------------------------------------------------------------------------
## API endpoints

    GET    /health                                    Liveness check -> {"ok":true}.
    GET    /version                                   Version, commit sha, uptime, public_mode.
    GET    /whoami                                    Public mode: caller's recovery code + admin flag.
    POST   /restore                                   Public mode: adopt a previous identity.
    GET    /presets                                   Available run presets.
    GET    /models                                    Synthesis-model choices (public-mode override).
    POST   /runs        { query, preset }             Submit a new run.
    GET    /runs?limit=N                              Recent runs, newest first (default 50, max 200).
    GET    /runs/{id}                                 Full run record + events.
    POST   /runs/{id}/retry                           Re-run a failed run.
    DELETE /runs/{id}                                 Cancel + delete a run.
    DELETE /runs?confirm=yes                          Wipe ALL runs. Irreversible.
    GET    /runs/{id}/stream                          Server-Sent Events progress stream.
    GET    /runs/{id}/graph?iter=latest|N             NetworkX node-link JSON.
    GET    /runs/{id}/synthesis?draft=latest|merged|N Synthesized artifact + sources.
    GET    /runs/{id}/search-results                  Seed search page parsed to JSON.
    GET    /runs/{id}/file/{path}                     Serve a file from the run's repository.

OpenAPI docs: http://localhost:8090/docs.

With --password set, the auth gate lives in the Next.js middleware (/login
+ caesar_auth cookie); the FastAPI process is bound to 127.0.0.1 and
reachable only via the UI's same-origin proxy.

### Public mode

`--public` (or `PUBLIC_MODE=1`) turns the server into a bring-your-own-key
deployment. The middleware mints an opaque, HttpOnly `caesar_id` cookie per
browser as its tenant identity — it does not gate anything; FastAPI enforces
per-row ownership off that cookie — and every run must carry the caller's own
OpenAI key in the request body. `GET /whoami` returns the cookie value as a
recovery code and `POST /restore` re-adopts it after a cookie reset.

Public and password mode co-exist. In public mode the password is not a login
gate but an optional admin step-up: entering it at /login sets the same
`caesar_auth` cookie, which elevates that one browser to see and wipe every
user's runs. Empty password = admin disabled.


--------------------------------------------------------------------------
## Development

The `.venv` referenced below lives at `web_server/api/.venv` and is created
automatically by `launch.sh` on first run. To create it manually without
launching the server:

    cd web_server/api
    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"

Backend tests + lint:

    cd web_server/api
    .venv/bin/pytest -q
    .venv/bin/ruff check app tests

Smoke tests force CAESAR_DRY_RUN=1, isolate per-test SQLite, and run the
full submit → poll → fetch loop. The suite defaults to `-n 8`
(api/pyproject.toml addopts) — ~24s that way against ~44s serial. Use
`.venv/bin/pytest -q -n 0` for a serial debug run.

UI checks:

    cd web_server/ui
    npm run typecheck
    npm run build

Architecture in three lines:

  - No job queue. A single asyncio JobPool owns every run; CaesarAgent.explore()
    is blocking, so each run is dispatched to asyncio.to_thread. A watchdog
    coroutine tails the artifact dir + console log to emit SSE progress events.

  - SSE replay-then-tail. /runs/{id}/stream first replays every persisted
    RunEvent from SQLite, then attaches to the in-memory queue, so refresh
    or reconnect shows full history without duplicates.

  - Crash recovery. On startup the lifespan archives stale checkpoints from
    terminal runs, resumes non-terminal runs that still have a live
    checkpoint, and fails any orphaned running/queued rows.


--------------------------------------------------------------------------
## License

Apache 2.0, same as Caesar.
