#!/usr/bin/env bash
# Install the daemon as a launchd user agent.
# Idempotent: re-running unloads the old plist, re-renders, reloads.
set -euo pipefail

HIPPO_REPO="$(cd "$(dirname "$0")/.." && pwd)"
HIPPO_HOME="$HOME"
HIPPO_USER="$(whoami)"
HIPPO_UV="$(command -v uv || true)"
if [[ -z "$HIPPO_UV" || ! -x "$HIPPO_UV" ]]; then
  echo "error: 'uv' not found on PATH; install uv first (https://docs.astral.sh/uv/)" >&2
  exit 1
fi
PLIST_LABEL="com.${HIPPO_USER}.memory-daemon"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
TEMPLATE="$HIPPO_REPO/launchd/memory-daemon.plist.template"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$HOME/.claude/debug"

# Render template
sed \
  -e "s|HIPPO_REPO|$HIPPO_REPO|g" \
  -e "s|HIPPO_HOME|$HIPPO_HOME|g" \
  -e "s|HIPPO_USER|$HIPPO_USER|g" \
  -e "s|HIPPO_UV|$HIPPO_UV|g" \
  "$TEMPLATE" > "$PLIST_PATH"

# Unload if loaded
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "Installed $PLIST_LABEL"
launchctl print "gui/$(id -u)/$PLIST_LABEL" | head -20
