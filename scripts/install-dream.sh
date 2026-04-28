#!/usr/bin/env bash
# Install the dream-heavy launchd agent (nightly 3am schedule).
# Idempotent: re-running unloads the old plist, re-renders, reloads.
set -euo pipefail

HIPPO_REPO="$(cd "$(dirname "$0")/.." && pwd)"
HIPPO_HOME="$HOME"
HIPPO_USER="$(whoami)"

PLIST_LABEL="com.${HIPPO_USER}.dream-heavy"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
TEMPLATE="$HIPPO_REPO/launchd/dream-heavy.plist.template"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$HOME/.claude/debug"

# Render template
sed \
  -e "s|HIPPO_REPO|$HIPPO_REPO|g" \
  -e "s|HIPPO_HOME|$HIPPO_HOME|g" \
  -e "s|HIPPO_USER|$HIPPO_USER|g" \
  "$TEMPLATE" > "$PLIST_PATH"

# Unload if loaded
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "Installed $PLIST_LABEL -> $PLIST_PATH"
