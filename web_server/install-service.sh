#!/usr/bin/env bash
# install-service.sh — install/uninstall caesar-web as an autostart service.
#
# Autodetects platform:
#   Linux  → systemd --user unit at ~/.config/systemd/user/caesar-web.service
#   macOS  → launchd LaunchAgent at ~/Library/LaunchAgents/com.caesar.web.plist
#
# Either way the server starts at login/boot, restarts on crash, runs
# without an attached terminal. Nothing touches root or system paths.
#
# Usage:
#   ./install-service.sh                       # install + start (no auth)
#   ./install-service.sh --password 's3cret'   # install with login password
#   ./install-service.sh --public             # install in public BYO-key mode
#   ./install-service.sh --uninstall           # stop and remove the unit
#
# Multi-instance (Linux/systemd only):
#   ./install-service.sh --instance-id b \
#       --api-port 8092 --ui-port 3001 --chroma-port 8093 \
#       --password 'creative'
#   ./install-service.sh --instance-id b --uninstall
#
# Adds CAESAR_INSTANCE_ID=b to the unit; launch.sh derives all per-instance
# paths (.logs-b, .next-b, api/data-b) from it. The unit is named
# caesar-web-b.service so multiple instances coexist.
#
# Can be run from any directory; the service uses this script's directory as
# the WorkingDirectory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
PASSWORD=""
PUBLIC_MODE=0
UNINSTALL=0
INSTANCE_ID=""
API_PORT=""
UI_PORT=""
CHROMA_PORT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --password|-p)
            [ $# -ge 2 ] || { echo "$1 requires a value" >&2; exit 1; }
            [ -n "$2" ] || { echo "$1 cannot be empty" >&2; exit 1; }
            PASSWORD="$2"
            shift 2
            ;;
        --password=*)
            PASSWORD="${1#--password=}"
            [ -n "$PASSWORD" ] || { echo "--password cannot be empty" >&2; exit 1; }
            shift
            ;;
        --public)
            PUBLIC_MODE=1
            shift
            ;;
        --instance-id|-i)
            INSTANCE_ID="$2"; shift 2 ;;
        --instance-id=*)
            INSTANCE_ID="${1#--instance-id=}"; shift ;;
        --api-port)
            API_PORT="$2"; shift 2 ;;
        --api-port=*)
            API_PORT="${1#--api-port=}"; shift ;;
        --ui-port)
            UI_PORT="$2"; shift 2 ;;
        --ui-port=*)
            UI_PORT="${1#--ui-port=}"; shift ;;
        --chroma-port)
            CHROMA_PORT="$2"; shift 2 ;;
        --chroma-port=*)
            CHROMA_PORT="${1#--chroma-port=}"; shift ;;
        --uninstall)
            UNINSTALL=1
            shift
            ;;
        --help|-h)
            grep -E '^# ' "$SCRIPT_PATH" | head -30
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

# public + password now co-exist (mirrors launch.sh): in public mode the
# password is an optional admin step-up (see + wipe all users' runs), not a
# full login gate. Both are threaded into the unit's launch invocation.

# Validate INSTANCE_ID with the same rule launch.sh uses. Reject the bad
# inputs (uppercase, whitespace, slashes, etc.) at install time so a typo
# is caught before the unit ever runs.
if [ -n "$INSTANCE_ID" ]; then
    if [ ${#INSTANCE_ID} -gt 32 ] \
        || [ "${INSTANCE_ID//[!a-z0-9_-]/}" != "$INSTANCE_ID" ] \
        || ! [[ "${INSTANCE_ID:0:1}" =~ [a-z0-9] ]]; then
        echo "--instance-id '$INSTANCE_ID' must match [a-z0-9][a-z0-9_-]{0,31}" >&2
        exit 1
    fi
    # Require explicit ports when an instance ID is set: instance A's defaults
    # would silently collide with this new instance.
    if [ "$UNINSTALL" = "0" ]; then
        [ -n "$API_PORT" ]    || { echo "--api-port required when --instance-id is set" >&2; exit 1; }
        [ -n "$UI_PORT" ]     || { echo "--ui-port required when --instance-id is set" >&2; exit 1; }
        [ -n "$CHROMA_PORT" ] || { echo "--chroma-port required when --instance-id is set" >&2; exit 1; }
    fi
fi

# ---------------------------------------------------------------------------
# Output helpers (mirror launch.sh)
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
# Platform detection
# ---------------------------------------------------------------------------
case "$(uname -s)" in
    Linux)   PLATFORM="linux" ;;
    Darwin)  PLATFORM="macos" ;;
    *) fail "Unsupported platform: $(uname -s) — only Linux (systemd) and macOS (launchd) are wired up." ;;
esac

[ -f "./launch.sh" ] || fail "launch.sh not found in $SCRIPT_DIR."

WORKING_DIR="$(pwd -P)"
LOGS_DIR="$WORKING_DIR/.logs"
mkdir -p "$LOGS_DIR"

# Build the launch invocation embedded into the unit/plist.
#
# Password delivery is platform-split, because anything on a command line is
# world-readable through /proc/<pid>/cmdline (`ps -ef` shows it to every local
# account) for the whole life of the process:
#
# Both platforms deliver it by environment, never on the command line:
#   Linux → Environment=CAESAR_PASSWORD in the 0600 unit file
#   macOS → EnvironmentVariables in the plist (also 0600 below)
# launch.sh reads CAESAR_PASSWORD directly, so neither platform needs to pass
# --password and process argv stays clean on both. That also lifts the old
# restriction on passwords containing a single quote, which the plist's argv
# quoting could not carry.
# --public and a password co-exist (see above), so the flags compose instead of
# being an if/elif chain: an earlier version let the password branch silently
# drop --public.
# XML character-data escaping for the plist. Done with sed, not bash pattern
# substitution: since bash 5.2 an unquoted & in a ${v//p/r} replacement means
# "the text that matched", so ${v//</&lt;} yields "<lt;" rather than "&lt;". The
# xml_command escaping below had that bug latent -- the service command contains
# 2>/dev/null, which became 2>gt;/dev/null on any bash newer than the 3.2 macOS
# ships. sed's \& is an explicit literal and behaves the same on both.
xml_escape() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

LAUNCH_INVOCATION="./launch.sh"
if [ "$PUBLIC_MODE" = "1" ]; then
    LAUNCH_INVOCATION="$LAUNCH_INVOCATION --public"
fi
PASSWORD_UNIT_ENV=""
PASSWORD_PLIST_ENV=""
if [ -n "$PASSWORD" ]; then
    case "$PASSWORD" in
        *$'\n'*) fail "Password contains a newline, which breaks unit/plist parsing." ;;
    esac
    # systemd Environment=: the whole KEY=VALUE is double-quoted, so only \ "
    # and % (the specifier sigil) need escaping. $ and ' are literal.
    sysesc=$PASSWORD
    sysesc=${sysesc//\\/\\\\}   # \ -> \\   (first, so later escapes aren't doubled)
    sysesc=${sysesc//\"/\\\"}   # " -> \"
    sysesc=${sysesc//%/%%}      # % -> %%
    PASSWORD_UNIT_ENV="Environment=\"CAESAR_PASSWORD=$sysesc\"
"
    xmlesc=$(xml_escape "$PASSWORD")
    PASSWORD_PLIST_ENV="    <key>EnvironmentVariables</key>
    <dict>
        <key>CAESAR_PASSWORD</key>
        <string>$xmlesc</string>
    </dict>
"
fi
# Linux command line: no password on it. macOS keeps the argv form.
SERVICE_COMMAND="source \$HOME/.bashrc 2>/dev/null; source \$HOME/.zshrc 2>/dev/null; cd \"$WORKING_DIR\"; exec $LAUNCH_INVOCATION"
MAC_SERVICE_COMMAND="$SERVICE_COMMAND"

# ---------------------------------------------------------------------------
# Linux: systemd --user
# ---------------------------------------------------------------------------
linux_install() {
    command -v systemctl >/dev/null 2>&1 || fail "systemctl not found, this script needs systemd."

    # Per-instance unit name + logs path. INSTANCE_ID empty -> legacy layout.
    local unit_name="caesar-web${INSTANCE_ID:+-$INSTANCE_ID}.service"
    local unit_path="$HOME/.config/systemd/user/$unit_name"
    local logs_dir="$WORKING_DIR/.logs${INSTANCE_ID:+-$INSTANCE_ID}"

    if [ -f "$unit_path" ]; then
        fail "Unit already exists at $unit_path. Remove with: ./install-service.sh ${INSTANCE_ID:+--instance-id $INSTANCE_ID }--uninstall"
    fi

    mkdir -p "$(dirname "$unit_path")"
    mkdir -p "$logs_dir"

    # Build instance-specific Environment= block. Empty when INSTANCE_ID
    # isn't set so the legacy unit content is byte-identical to before.
    local instance_env=""
    if [ -n "$INSTANCE_ID" ]; then
        instance_env="Environment=CAESAR_INSTANCE_ID=$INSTANCE_ID
Environment=API_PORT=$API_PORT
Environment=UI_PORT=$UI_PORT
Environment=CAESAR_CHROMA_PORT=$CHROMA_PORT
"
    fi

    step "Writing unit file at $unit_path"
    cat > "$unit_path" <<EOF
[Unit]
Description=Caesar web server${INSTANCE_ID:+ ($INSTANCE_ID)} (FastAPI + Next.js) via launch.sh
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$WORKING_DIR
# Pass the unit's own name so launch.sh can cross-check it against
# CAESAR_INSTANCE_ID and fail loud on a typo (caesar-web-c.service with
# INSTANCE_ID=b would otherwise silently share instance b's data dir).
Environment=SYSTEMD_UNIT_NAME=%n
${PASSWORD_UNIT_ENV}${instance_env}# Source shell rc so LLM API keys exported there reach launch.sh, then cd
# back: many users' rc files end in a 'cd ~/somewhere' that overrides
# systemd's WorkingDirectory and breaks the './launch.sh' relative path.
ExecStart=/bin/bash -c '$SERVICE_COMMAND'
Restart=on-failure
RestartSec=10s
StandardOutput=append:$logs_dir/systemd.log
StandardError=append:$logs_dir/systemd.log
KillMode=mixed
TimeoutStopSec=30s

[Install]
WantedBy=default.target
EOF
    # Password lives in this file when set.
    chmod 600 "$unit_path"

    step "Enabling and starting $unit_name"
    systemctl --user daemon-reload
    systemctl --user enable --now "$unit_name"

    if loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q "Linger=yes"; then
        info "loginctl: lingering already enabled (autostart on boot ✓)"
    elif loginctl enable-linger 2>/dev/null; then
        info "loginctl: lingering enabled for $USER (autostart on boot ✓)"
    else
        warn "loginctl enable-linger failed without privilege — service won't start before login."
        info "Run manually:  sudo loginctl enable-linger $USER"
    fi

    local unit_short="${unit_name%.service}"
    cat <<EOF

${B}${GRN}✓ $unit_short installed (systemd --user).${RST}

    Unit:      $unit_path
    Status:    ${B}systemctl --user status $unit_short${RST}
    Logs:      ${B}journalctl --user -u $unit_short -f${RST}
               (also $logs_dir/systemd.log)
    Restart:   ${B}systemctl --user restart $unit_short${RST}
    Uninstall: ${B}./install-service.sh ${INSTANCE_ID:+--instance-id $INSTANCE_ID }--uninstall${RST}

EOF
}

linux_uninstall() {
    local unit_name="caesar-web${INSTANCE_ID:+-$INSTANCE_ID}.service"
    local unit_path="$HOME/.config/systemd/user/$unit_name"

    if [ ! -f "$unit_path" ]; then
        warn "No unit found at $unit_path, nothing to uninstall."
        exit 0
    fi
    step "Stopping and disabling $unit_name"
    systemctl --user disable --now "$unit_name" 2>/dev/null || true
    rm -f "$unit_path"
    systemctl --user daemon-reload
    step "✓ Service uninstalled (unit file removed)"
    info "loginctl disable-linger left as-is. Disable manually if you have no other user services using boot autostart."
}

# ---------------------------------------------------------------------------
# macOS: launchd LaunchAgent
# ---------------------------------------------------------------------------
macos_install() {
    command -v launchctl >/dev/null 2>&1 || fail "launchctl not found — needed for macOS LaunchAgents."

    local label="com.caesar.web"
    local plist_path="$HOME/Library/LaunchAgents/$label.plist"
    local uid; uid=$(id -u)

    if [ -f "$plist_path" ]; then
        fail "LaunchAgent already exists at $plist_path. Remove with: ./install-service.sh --uninstall"
    fi

    mkdir -p "$(dirname "$plist_path")"

    # Escape XML-special chars in the service command. The bash double-
    # quoted password is fine inside <string>…</string> (`"` is legal in
    # XML element content); only &/</> need entity encoding.
    local xml_command
    xml_command=$(xml_escape "$MAC_SERVICE_COMMAND")

    step "Writing LaunchAgent at $plist_path"
    cat > "$plist_path" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>$xml_command</string>
    </array>
${PASSWORD_PLIST_ENV}    <key>WorkingDirectory</key>
    <string>$WORKING_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOGS_DIR/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$LOGS_DIR/launchd.log</string>
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
EOF
    chmod 600 "$plist_path"

    step "Loading $label into launchd"
    # bootout any leftover from a prior install before bootstrapping fresh.
    launchctl bootout "gui/$uid/$label" 2>/dev/null || true
    launchctl bootstrap "gui/$uid" "$plist_path"

    cat <<EOF

${B}${GRN}✓ caesar-web installed (launchd LaunchAgent).${RST}

    Plist:     $plist_path
    Status:    ${B}launchctl print gui/$uid/$label${RST}
    Logs:      ${B}tail -f $LOGS_DIR/launchd.log${RST}
    Restart:   ${B}launchctl kickstart -k gui/$uid/$label${RST}
    Uninstall: ${B}./install-service.sh --uninstall${RST}

  LaunchAgents start at login (autostart on boot is implicit — the
  agent loads as soon as you log in to a graphical session).

EOF
}

macos_uninstall() {
    local label="com.caesar.web"
    local plist_path="$HOME/Library/LaunchAgents/$label.plist"
    local uid; uid=$(id -u)

    if [ ! -f "$plist_path" ]; then
        warn "No LaunchAgent found at $plist_path — nothing to uninstall."
        exit 0
    fi
    step "Stopping and removing $label"
    launchctl bootout "gui/$uid/$label" 2>/dev/null || true
    rm -f "$plist_path"
    step "✓ Service uninstalled (LaunchAgent removed)"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
if [ "$UNINSTALL" = "1" ]; then
    case "$PLATFORM" in
        linux) linux_uninstall ;;
        macos) macos_uninstall ;;
    esac
else
    case "$PLATFORM" in
        linux) linux_install ;;
        macos) macos_install ;;
    esac
fi
