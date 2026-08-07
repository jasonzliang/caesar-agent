#!/usr/bin/env bash
# launch.sh — start the Caesar web server (FastAPI + Next.js)
#
# Auto-installs missing dependencies on first run, then boots both processes
# bound to 0.0.0.0 so a remote laptop can reach the UI on the LAN.
#
# Usage:
#   ./launch.sh                       # boot in real mode (uses LLM API keys)
#   CAESAR_DRY_RUN=1 ./launch.sh      # boot in dry-run mode (no LLM calls)
#   API_PORT=9000 ./launch.sh         # override port
#   ./launch.sh --password 's3cret'   # require login w/ this password
#   CAESAR_PASSWORD='s3cret' ./launch.sh  # same, but keeps it out of `ps` output
#   ./launch.sh --public              # public bring-your-own-key mode
#
# Env overrides:
#   CAESAR_PASSWORD    login password (preferred over --password: argv is
#                      world-readable through /proc/<pid>/cmdline).
#                      DEMO_PASSWORD is still honoured as an alias -- it is the
#                      name the app itself reads, and what older units set.
#   API_HOST           default 0.0.0.0
#   API_PORT           default 8090   (8000 is commonly taken by ChromaDB)
#   UI_PORT            default 3000
#   CAESAR_DRY_RUN     default 0
#   CAESAR_MAX_CONCURRENT  default 8
#   NODE_VERSION       fallback Node version, default 20.18.0
#   NODE20_ROOT        fallback Node install dir, default $HOME/.local/node20
#   NODE20_BIN         fallback Node bin dir, default $NODE20_ROOT/bin

set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
# The password comes from the environment by preference: argv is world-readable
# via /proc/<pid>/cmdline, so `--password` leaks it to every local account for as
# long as the server runs. install-service.sh writes it as Environment= in the
# 0600 unit file.
#
# Precedence, highest first:
#   --password <value>   explicit flag
#   CAESAR_PASSWORD      the CAESAR_* family every other setting uses, and the
#                        name the container entrypoint and Helm chart already use
#   DEMO_PASSWORD        alias: the name the app itself reads (middleware,
#                        /api/auth/login, api/app/config.py) and what units
#                        written before this set
PASSWORD="${CAESAR_PASSWORD:-${DEMO_PASSWORD:-}}"
PUBLIC_MODE="${PUBLIC_MODE:-0}"
while [ $# -gt 0 ]; do
    case "$1" in
        --password|-p)
            [ $# -ge 2 ] || { echo "$1 requires a value" >&2; exit 1; }
            [ -n "$2" ] || { echo "$1 cannot be empty (use --help to see how to disable auth)" >&2; exit 1; }
            PASSWORD="$2"
            shift 2
            ;;
        --password=*)
            PASSWORD="${1#--password=}"
            [ -n "$PASSWORD" ] || { echo "--password cannot be empty" >&2; exit 1; }
            shift
            ;;
        --public)
            PUBLIC_MODE="1"
            shift
            ;;
        --help|-h)
            grep -E '^# ' "$0" | head -25
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

# Mode semantics for a set password:
#   * non-public + password → full login gate (Next.js middleware; every page
#     requires the caesar_auth cookie).
#   * public   + password → the site stays open (anonymous per-browser runs),
#     and the password is an OPTIONAL admin step-up: entering it at /login
#     elevates that browser to admin, which can see and wipe every user's runs.
# The password is threaded to BOTH the UI (issues the cookie) and the API
# (validates it) below.

# ---------------------------------------------------------------------------
# Tiny output helpers
# ---------------------------------------------------------------------------
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
    B=$(tput bold) D=$(tput dim) RST=$(tput sgr0)
    GRN=$(tput setaf 2) YEL=$(tput setaf 3) RED=$(tput setaf 1)
else
    B="" D="" RST="" GRN="" YEL="" RED=""
fi
step() { printf "%s==>%s %s\n" "${B}${GRN}" "${RST}" "$*"; }
info() { printf "    %s%s%s\n" "${D}" "$*" "${RST}"; }
warn() { printf "%s!%s  %s\n" "${B}${YEL}" "${RST}" "$*"; }
fail() { printf "%s✗%s  %s\n" "${B}${RED}" "${RST}" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8090}"
UI_PORT="${UI_PORT:-3000}"
CAESAR_DRY_RUN="${CAESAR_DRY_RUN:-0}"
CAESAR_MAX_CONCURRENT="${CAESAR_MAX_CONCURRENT:-8}"
MEM0_TELEMETRY="${MEM0_TELEMETRY:-false}"
export MEM0_TELEMETRY

# Localhost-only bind: when --password is set the auth gate (Next.js
# middleware in the UI) must not be bypassable by hitting the API port
# directly on the LAN. Public mode enforces per-browser ownership the same
# way (the cookie only reaches FastAPI through the Next proxy), so it also
# binds the API to 127.0.0.1. Either mode decouples the localhost bind from
# the LAN-open default.
if [ -n "$PASSWORD" ] || [ "$PUBLIC_MODE" = "1" ]; then
    API_HOST="127.0.0.1"
fi

# Public mode is bring-your-own-key: every run must supply its own OpenAI key
# (the API rejects a submission without one). Strip operator LLM keys from the
# environment so the server is FAIL-CLOSED. rome/llm_handler.py resolves the key
# as `config["api_key"] or os.getenv(...)`, so without this a missed per-run
# injection seam would silently fall back to and bill the operator's key instead
# of erroring. Web-search keys (BRAVE_API_KEY) are server-funded and kept.
if [ "$PUBLIC_MODE" = "1" ]; then
    unset OPENAI_API_KEY CHROMA_OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY OPENROUTER_API_KEY
fi

API_DIR="api"
UI_DIR="ui"
VENV="$API_DIR/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
UVICORN="$VENV/bin/uvicorn"

# Prefer `uv pip` over plain pip — orders of magnitude faster on cold install
# and resolves dependency conflicts more reliably. Falls back to the venv's
# pip if uv isn't on PATH.
if command -v uv >/dev/null 2>&1; then
    INSTALL="uv pip install --python $PY"
else
    INSTALL="$PIP install"
fi
# Optional instance ID: when set (e.g., "b"), all per-instance paths
# get a "-${ID}" suffix so multiple caesar-web instances can coexist on
# one machine. Default (empty) keeps the legacy layout (.logs, .next,
# api/data) so existing deployments are untouched. CAESAR_WEB_DATA_DIR,
# CAESAR_WEB_LOGS_DIR, and NEXT_DIST_DIR may still be set explicitly to
# override individual paths.
CAESAR_INSTANCE_ID="${CAESAR_INSTANCE_ID:-}"
# Lowercase-only token: bans whitespace, newlines, slashes, path traversal,
# and case-insensitive-FS collisions (b vs B). First char alphanumeric.
# Cap 32 chars to stay clear of NAME_MAX on the suffixed directories.
# Char-class substitution catches embedded newlines that bash regex's $
# anchor misses; explicit length check pairs with it.
if [ -n "$CAESAR_INSTANCE_ID" ]; then
    if [ ${#CAESAR_INSTANCE_ID} -gt 32 ] \
        || [ "${CAESAR_INSTANCE_ID//[!a-z0-9_-]/}" != "$CAESAR_INSTANCE_ID" ] \
        || ! [[ "${CAESAR_INSTANCE_ID:0:1}" =~ [a-z0-9] ]]; then
        fail "CAESAR_INSTANCE_ID '$CAESAR_INSTANCE_ID' must match [a-z0-9][a-z0-9_-]{0,31}"
    fi
fi
INSTANCE_SUFFIX=""
[ -n "$CAESAR_INSTANCE_ID" ] && INSTANCE_SUFFIX="-${CAESAR_INSTANCE_ID}"

# When running under systemd, cross-check the unit name against the env-set
# ID so a typo (e.g. caesar-web-c.service with Environment=CAESAR_INSTANCE_ID=b)
# fails loud at boot instead of silently sharing instance B's data dir.
# Skips the check when SYSTEMD_UNIT_NAME isn't set (manual launch, dev).
if [ -n "${SYSTEMD_UNIT_NAME:-}" ]; then
    expected_id="${SYSTEMD_UNIT_NAME%.service}"
    expected_id="${expected_id#caesar-web}"
    expected_id="${expected_id#-}"
    if [ "$expected_id" != "$CAESAR_INSTANCE_ID" ]; then
        fail "unit '$SYSTEMD_UNIT_NAME' implies CAESAR_INSTANCE_ID='$expected_id' but env has '$CAESAR_INSTANCE_ID'"
    fi
fi

LOGS_DIR="${CAESAR_WEB_LOGS_DIR:-.logs${INSTANCE_SUFFIX}}"
mkdir -p "$LOGS_DIR"
API_LOG="$LOGS_DIR/api.log"
UI_LOG="$LOGS_DIR/ui.log"

# Next.js distDir for build/start — kept in sync across the build and
# launch lines below. Exported so next.config.mjs picks it up.
export NEXT_DIST_DIR="${NEXT_DIST_DIR:-.next${INSTANCE_SUFFIX}}"

# Default the FastAPI data dir to a suffixed sibling so SQLite + chroma
# + runs/ don't collide between instances. User can still override
# CAESAR_WEB_DATA_DIR explicitly (e.g. to point at an external disk).
if [ -z "${CAESAR_WEB_DATA_DIR:-}" ]; then
    export CAESAR_WEB_DATA_DIR="$(pwd)/api/data${INSTANCE_SUFFIX}"
fi

# ---------------------------------------------------------------------------
# Local Node fallback
# ---------------------------------------------------------------------------
# Use the caller's Node when it is new enough; otherwise fall back to a
# user-local Node install. If the fallback is missing, install a pinned Node
# release without sudo. This keeps launch.sh usable in non-interactive shells
# without changing the user's global shell or system Node default.
NODE_VERSION="${NODE_VERSION:-20.18.0}"
NODE20_ROOT="${NODE20_ROOT:-$HOME/.local/node20}"
NODE20_BIN="${NODE20_BIN:-$NODE20_ROOT/bin}"

node_major() {
    if ! command -v node >/dev/null 2>&1; then
        echo 0
        return
    fi
    node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0
}

node_toolchain_ok() {
    local major
    major=$(node_major)
    [ "${major:-0}" -ge 20 ] 2>/dev/null && command -v npm >/dev/null 2>&1
}

prepend_local_node() {
    PATH="$NODE20_BIN:$PATH"
    export PATH
}

node_platform() {
    local os arch
    os=$(uname -s)
    arch=$(uname -m)
    case "$os:$arch" in
        Linux:x86_64|Linux:amd64) echo "linux-x64" ;;
        Linux:aarch64|Linux:arm64) echo "linux-arm64" ;;
        Darwin:x86_64) echo "darwin-x64" ;;
        Darwin:arm64) echo "darwin-arm64" ;;
        *) echo "" ;;
    esac
}

install_local_node() {
    [ -n "$NODE_VERSION" ] || fail "NODE_VERSION cannot be empty"
    case "$NODE20_ROOT" in
        ""|"/"|"$HOME"|"$HOME/")
            fail "Unsafe NODE20_ROOT: $NODE20_ROOT"
            ;;
    esac
    command -v curl >/dev/null 2>&1 || fail "curl not found on PATH; required to install local Node $NODE_VERSION"
    command -v tar >/dev/null 2>&1 || fail "tar not found on PATH; required to install local Node $NODE_VERSION"

    local platform archive url tmp
    platform=$(node_platform)
    [ -n "$platform" ] || fail "No Node binary mapping for $(uname -s)/$(uname -m). Install Node 20+ manually."

    archive="node-v$NODE_VERSION-$platform.tar.xz"
    url="https://nodejs.org/dist/v$NODE_VERSION/$archive"
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/caesar-node.XXXXXX")

    step "Installing local Node $NODE_VERSION ($platform)"
    if ! curl -fsSL "$url" -o "$tmp/$archive"; then
        rm -rf "$tmp"
        fail "Node download failed: $url"
    fi
    mkdir -p "$(dirname "$NODE20_ROOT")"
    rm -rf "$NODE20_ROOT.tmp"
    mkdir -p "$NODE20_ROOT.tmp"
    if ! tar -xJf "$tmp/$archive" -C "$NODE20_ROOT.tmp" --strip-components=1; then
        rm -rf "$tmp" "$NODE20_ROOT.tmp"
        fail "Node archive extraction failed"
    fi
    rm -rf "$NODE20_ROOT"
    mv "$NODE20_ROOT.tmp" "$NODE20_ROOT"
    rm -rf "$tmp"
    info "Installed Node at $NODE20_ROOT"
}

if ! node_toolchain_ok; then
    if [ -x "$NODE20_BIN/node" ]; then
        prepend_local_node
    fi
fi
if ! node_toolchain_ok; then
    install_local_node
    prepend_local_node
fi

# ---------------------------------------------------------------------------
# Toolchain checks
# ---------------------------------------------------------------------------
# Suggest a copy-pasteable install command for the user's platform on
# missing/too-old toolchain failures. Node gets a user-local fallback above;
# everything else only gets a hint. The hint is best-effort: macOS via brew,
# Linux via apt/dnf/pacman; anything else falls back to a generic message.
install_hint() {
    local tool="$1"
    case "$(uname -s)" in
        Darwin)
            case "$tool" in
                python3) echo "brew install python@3.13" ;;
                node|npm) echo "brew install node" ;;
                curl)    echo "brew install curl  # (usually preinstalled on macOS)" ;;
            esac
            ;;
        Linux)
            if   command -v apt-get >/dev/null 2>&1; then
                case "$tool" in
                    python3) echo "sudo apt install python3 python3-venv" ;;
                    # Default apt nodejs is often <20 — point at NodeSource.
                    node|npm) echo "see https://github.com/nodesource/distributions for Node 20+" ;;
                    curl)    echo "sudo apt install curl" ;;
                esac
            elif command -v dnf >/dev/null 2>&1; then
                case "$tool" in
                    python3) echo "sudo dnf install python3" ;;
                    node|npm) echo "sudo dnf install nodejs" ;;
                    curl)    echo "sudo dnf install curl" ;;
                esac
            elif command -v pacman >/dev/null 2>&1; then
                case "$tool" in
                    python3) echo "sudo pacman -S python" ;;
                    node|npm) echo "sudo pacman -S nodejs npm" ;;
                    curl)    echo "sudo pacman -S curl" ;;
                esac
            fi
            ;;
    esac
}

need_tool() {
    local tool="$1"
    if ! command -v "$tool" >/dev/null 2>&1; then
        local hint
        hint=$(install_hint "$tool")
        if [ -n "$hint" ]; then
            fail "$tool not found on PATH. Install with: $hint"
        else
            fail "$tool not found on PATH."
        fi
    fi
}

need_tool python3
# Final guards after the local Node fallback above.
need_tool node
need_tool npm
need_tool curl

# Version preflight — catches old Python / Node up front instead of letting
# pip's resolver or `next` blow up later with a confusing message.
PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3,10) else 0)' 2>/dev/null || echo 0)
if [ "$PY_OK" != "1" ]; then
    hint=$(install_hint python3)
    fail "Python 3.10+ required (have $(python3 -V 2>&1)).${hint:+ Install: $hint}"
fi

NODE_MAJOR=$(node_major)
if ! [ "${NODE_MAJOR:-0}" -ge 20 ] 2>/dev/null; then
    hint=$(install_hint node)
    fail "Node 20+ required (have $(node -v 2>&1)).${hint:+ Install: $hint}"
fi

# ---------------------------------------------------------------------------
# 1. API: venv + Python deps
# ---------------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
    step "Creating Python venv at $VENV"
    python3 -m venv "$VENV"
fi

# Install web-server FastAPI deps + dev tooling, but only if a sentinel
# import is missing — keeps re-launches fast.
if ! "$PY" -c "import fastapi, sse_starlette, sqlalchemy, aiosqlite, greenlet" 2>/dev/null; then
    step "Installing API package + FastAPI deps (api/pyproject.toml)"
    $INSTALL --quiet -e "${API_DIR}[dev]"
else
    info "API FastAPI deps already installed"
fi

# Install Caesar runtime deps from rome/requirements.txt — minus pygraphviz
# which needs system graphviz and isn't required for the server. Skipped
# if a sentinel set is already importable AND any pinned versions match.
#
# Sentinel set covers the heaviest deps; if any are missing we re-run the
# full install. Smaller deps (networkx, beautifulsoup4, curl_cffi, PyPDF2,
# tiktoken, etc.) are pulled in transitively via the sentinel set or
# requirements.txt itself.
#
# We *also* validate chromadb's version against its requirements.txt pin
# (currently 1.5.2 — see chroma-core/chroma#4038 / SQLite-pool deadlock in
# 1.5.3+). Without this check, a venv that picked up 1.5.9 from a previous
# unpinned install would silently keep deadlocking on every relaunch.
EXPECTED_CHROMADB=""
if [ -f "../requirements.txt" ]; then
    EXPECTED_CHROMADB=$(sed -n 's/^chromadb==\([0-9][0-9.]*\).*/\1/p' ../requirements.txt | head -1)
fi
SENTINEL_PY="import litellm, chromadb, llama_index, anthropic, ddgs"
if [ -n "$EXPECTED_CHROMADB" ]; then
    SENTINEL_PY="$SENTINEL_PY; assert chromadb.__version__ == '$EXPECTED_CHROMADB', f'chromadb {chromadb.__version__} != pin $EXPECTED_CHROMADB'"
fi
if ! "$PY" -c "$SENTINEL_PY" 2>/dev/null; then
    step "Installing Caesar runtime deps (rome/requirements.txt, minus pygraphviz)"
    [ -f "../requirements.txt" ] || fail "../requirements.txt not found — Caesar can't run without it. Run launch.sh from web_server/ inside the rome repo."
    if [ -n "$EXPECTED_CHROMADB" ]; then
        installed=$("$PY" -c "import chromadb; print(chromadb.__version__)" 2>/dev/null || echo "absent")
        if [ "$installed" != "$EXPECTED_CHROMADB" ] && [ "$installed" != "absent" ]; then
            info "chromadb $installed installed; downgrading to $EXPECTED_CHROMADB (pin)"
        fi
    fi
    # Strip pygraphviz line; everything else is fair game.
    grep -v -E "^pygraphviz\s*$" ../requirements.txt > "$LOGS_DIR/req.txt"
    $INSTALL --quiet -r "$LOGS_DIR/req.txt" || \
        fail "Caesar deps install failed — see install output above"
else
    info "Caesar runtime deps already installed${EXPECTED_CHROMADB:+ (chromadb $EXPECTED_CHROMADB ✓)}"
fi

# ---------------------------------------------------------------------------
# Pre-flight: make sure the run will have what it needs.
# ---------------------------------------------------------------------------
if [ "$CAESAR_DRY_RUN" != "1" ]; then
    if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ] \
       && [ -z "${GOOGLE_API_KEY:-}" ]; then
        warn "No LLM API key found in environment (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY)."
        info "Export the relevant key(s) in your shell, or run with CAESAR_DRY_RUN=1 to skip real LLM calls."
        # Don't exit — Caesar's preset YAML decides which provider to use,
        # and the user might be relying on a single specific key.
    fi
    if [ -z "${BRAVE_API_KEY:-}" ]; then
        info "BRAVE_API_KEY not set — Caesar will fall back to DuckDuckGo (already the default for nano/mini presets)."
    fi
fi

# ---------------------------------------------------------------------------
# 2. UI: npm install + production build
# ---------------------------------------------------------------------------
UI_INSTALL_STAMP="$UI_DIR/node_modules/.package-lock.json"
if [ ! -d "$UI_DIR/node_modules" ] \
    || [ ! -x "$UI_DIR/node_modules/.bin/next" ] \
    || [ ! -f "$UI_INSTALL_STAMP" ] \
    || [ "$UI_DIR/package.json" -nt "$UI_INSTALL_STAMP" ] \
    || { [ -f "$UI_DIR/package-lock.json" ] && [ "$UI_DIR/package-lock.json" -nt "$UI_INSTALL_STAMP" ]; }; then
    step "Installing UI deps ($UI_DIR/package.json)"
    (cd "$UI_DIR" && npm ci --no-audit --no-fund --silent)
else
    info "UI deps already installed"
fi

# Build for production: hides the Next.js dev tools indicator, removes
# source maps + verbose error pages from the public-facing demo, and
# serves optimized bundles. Cost is ~30-60s on each launch — for a
# faster local dev loop, run `npm run dev` in $UI_DIR directly.
DIST="$UI_DIR/$NEXT_DIST_DIR"
if [ ! -f "$DIST/BUILD_ID" ] \
    || [ "$UI_DIR/package.json" -nt "$DIST/BUILD_ID" ] \
    || { [ -f "$UI_DIR/package-lock.json" ] && [ "$UI_DIR/package-lock.json" -nt "$DIST/BUILD_ID" ]; } \
    || [ "$UI_DIR/tailwind.config.ts" -nt "$DIST/BUILD_ID" ] \
    || [ -n "$(find "$UI_DIR/app" "$UI_DIR/components" "$UI_DIR/lib" \
        -newer "$DIST/BUILD_ID" -type f 2>/dev/null | head -1)" ]; then
    step "Building UI (production)"
    # API_INTERNAL_URL must be set HERE, not just at start: Next.js bakes
    # rewrites() into the per-distDir routes manifest at build time. Without
    # this, a non-default API_PORT (e.g. a second instance) bakes the 8090
    # fallback and the /api proxy silently targets the wrong backend.
    (cd "$UI_DIR" && API_INTERNAL_URL="http://127.0.0.1:$API_PORT" \
        npm run build > "../$LOGS_DIR/ui-build.log" 2>&1) \
        || fail "UI build failed — see $LOGS_DIR/ui-build.log"
else
    info "UI build is current (skipping rebuild)"
fi

# ---------------------------------------------------------------------------
# 3. Port checks (don't try to fight a process we don't own)
# ---------------------------------------------------------------------------
port_in_use() {
    lsof -ti:"$1" 2>/dev/null | head -1 || true
}

api_pid_in_use=$(port_in_use "$API_PORT")
ui_pid_in_use=$(port_in_use "$UI_PORT")
if [ -n "$api_pid_in_use" ]; then
    cmd=$(ps -p "$api_pid_in_use" -o comm= 2>/dev/null || echo "?")
    warn "Port :$API_PORT is held by PID $api_pid_in_use ($cmd)."
    info "Free it (\`kill $api_pid_in_use\`) or set API_PORT=<n> to use a different port."
    exit 1
fi
if [ -n "$ui_pid_in_use" ]; then
    cmd=$(ps -p "$ui_pid_in_use" -o comm= 2>/dev/null || echo "?")
    warn "Port :$UI_PORT is held by PID $ui_pid_in_use ($cmd)."
    info "Free it (\`kill $ui_pid_in_use\`) or set UI_PORT=<n> to use a different port."
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Boot
# ---------------------------------------------------------------------------
API_PID=""
UI_PID=""
CLEANED=0
cleanup() {
    [ "$CLEANED" = "1" ] && return
    CLEANED=1
    echo
    step "Shutting down…"
    if [ -n "$API_PID" ]; then kill "$API_PID" 2>/dev/null || true; fi
    if [ -n "$UI_PID"  ]; then kill "$UI_PID"  2>/dev/null || true; fi
    wait 2>/dev/null || true
}
# Ctrl-C / kill: tear down and exit immediately.
# Any other script exit (including `set -e` failures after API_PID is
# set): tear down silently. CLEANED makes cleanup re-entrant so both
# paths can fire without double-killing.
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

step "Starting API on $API_HOST:$API_PORT (dry_run=$CAESAR_DRY_RUN)"
# Put the venv's bin on PATH so any subprocess Caesar spawns (e.g. `chroma run`)
# resolves to the venv-pinned binary, not whatever older copy lives in the
# user's conda/system PATH and would otherwise bring a version mismatch.
VENV_BIN_ABS="$(cd "$VENV/bin" && pwd)"
(
    cd "$API_DIR"
    PATH="$VENV_BIN_ABS:$PATH" \
    CAESAR_DRY_RUN="$CAESAR_DRY_RUN" \
    CAESAR_MAX_CONCURRENT="$CAESAR_MAX_CONCURRENT" \
    PUBLIC_MODE="$PUBLIC_MODE" \
    DEMO_PASSWORD="$PASSWORD" \
    PYTHONFAULTHANDLER=1 \
    exec "../$UVICORN" app.main:app \
        --host "$API_HOST" --port "$API_PORT" \
        > "../$API_LOG" 2>&1
) &
API_PID=$!

# Wait up to ~15s for API readiness
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
        info "API ready (PID $API_PID)"
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        fail "API process died during startup — see $API_LOG"
    fi
    sleep 0.25
done
curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 \
    || fail "API failed to come up within 15s — see $API_LOG"

step "Starting UI on $UI_PORT (auth=$([ -n "$PASSWORD" ] && echo on || echo off), public=$([ "$PUBLIC_MODE" = "1" ] && echo on || echo off))"
(
    cd "$UI_DIR"
    PORT="$UI_PORT" \
    API_INTERNAL_URL="http://127.0.0.1:$API_PORT" \
    DEMO_PASSWORD="$PASSWORD" \
    PUBLIC_MODE="$PUBLIC_MODE" \
    exec npm run start > "../$UI_LOG" 2>&1
) &
UI_PID=$!

# Wait up to ~30s for UI readiness (Next.js can be slow on cold start)
for _ in $(seq 1 120); do
    if curl -sf -o /dev/null -w "%{http_code}" "http://127.0.0.1:$UI_PORT/" 2>/dev/null \
        | grep -q "^[23]"; then
        info "UI ready (PID $UI_PID)"
        break
    fi
    if ! kill -0 "$UI_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null || true
        fail "UI process died during startup — see $UI_LOG"
    fi
    sleep 0.25
done

# ---------------------------------------------------------------------------
# 5. Banner
# ---------------------------------------------------------------------------
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || \
         ipconfig getifaddr en1 2>/dev/null || \
         hostname 2>/dev/null || echo "<your-lan-ip>")

cat <<EOF

${B}${GRN}✓ Caesar web server is up.${RST}

    Local:        ${B}http://localhost:$UI_PORT${RST}
    Cross-laptop: ${B}http://$LAN_IP:$UI_PORT${RST}

    API docs:     http://localhost:$API_PORT/docs
    Logs:         $API_LOG, $UI_LOG
    Mode:         $([ "$CAESAR_DRY_RUN" = "1" ] && echo "${YEL}dry-run${RST} (synthetic artifacts, no LLM)" || echo "real (LLM API keys required)")
    Auth:         $([ -n "$PASSWORD" ] && { [ "$PUBLIC_MODE" = "1" ] && echo "${GRN}admin step-up${RST} — password unlocks see/wipe-all at /login" || echo "${GRN}password required${RST} — login at /login (API bound to 127.0.0.1)"; } || echo "open (no login)")
    Public:       $([ "$PUBLIC_MODE" = "1" ] && echo "${GRN}on${RST} — bring-your-own-key (anonymous per-browser runs, API bound to 127.0.0.1)" || echo "off")

  Press Ctrl-C to stop.

EOF

# Foreground; if either child dies, exit non-zero so systemd's
# Restart=on-failure auto-restarts the unit. Capture the dead child's
# exit status with `|| s=$?` so `set -e` doesn't kill us before we log.
# A clean (status 0) child death still counts as a failure from the
# launcher's perspective — the children shouldn't exit on their own.
child_status=0
# `wait -n` needs Bash 4.3+, but macOS still ships Bash 3.2 (GPLv2), where it
# errors out ("wait: -n: invalid option") and the launcher would tear both
# servers down at boot. Emulate it: poll both PIDs and reap whichever exits
# first. Bash reaps background children asynchronously while caching their
# exit status, so `kill -0` sees the death and the follow-up `wait <pid>`
# still yields the real status. `|| child_status=$?` keeps `set -e` from
# bailing before we capture it.
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$UI_PID" 2>/dev/null; do
    sleep 1
done
if ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID" 2>/dev/null || child_status=$?
else
    wait "$UI_PID" 2>/dev/null || child_status=$?
fi
warn "One of the processes exited unexpectedly (status $child_status). Tailing logs:"
tail -n 30 "$API_LOG" "$UI_LOG" 2>/dev/null || true
exit $(( child_status == 0 ? 1 : child_status ))
