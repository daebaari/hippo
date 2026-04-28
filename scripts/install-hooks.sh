#!/usr/bin/env bash
# Install Hippo hooks into ~/.claude/settings.json (idempotent).
# Backs up settings.json before each modification.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SETTINGS="$HOME/.claude/settings.json"
BACKUP="$HOME/.claude/settings.json.pre-hippo-$(date +%Y%m%d-%H%M%S)"

if [ ! -f "$SETTINGS" ]; then
  echo "ERROR: $SETTINGS not found. Is Claude Code installed?" >&2
  exit 1
fi

cp "$SETTINGS" "$BACKUP"
echo "Backup written to $BACKUP"

# Install hook scripts as symlinks in ~/.claude/hooks/
mkdir -p "$HOME/.claude/hooks"
ln -sf "$REPO/hooks/userprompt-submit.sh" "$HOME/.claude/hooks/hippo-userprompt-submit.sh"

# Add UserPromptSubmit hook entry via jq (idempotent: removes old hippo entry first)
jq --arg cmd "$HOME/.claude/hooks/hippo-userprompt-submit.sh" '
  .hooks.UserPromptSubmit |= (
    (. // [])
    | map(select((.hooks // []) | map(.command) | index($cmd) | not))
    | . + [{"hooks": [{"type": "command", "command": $cmd}]}]
  )
' "$SETTINGS" > /tmp/settings.json.new
mv /tmp/settings.json.new "$SETTINGS"

echo "Installed UserPromptSubmit hook → $HOME/.claude/hooks/hippo-userprompt-submit.sh"

ln -sf "$REPO/hooks/stop.sh" "$HOME/.claude/hooks/hippo-stop.sh"

# Add Stop hook entry via jq (idempotent: removes old hippo entry first)
jq --arg cmd "$HOME/.claude/hooks/hippo-stop.sh" '
  .hooks.Stop |= (
    (. // [])
    | map(select((.hooks // []) | map(.command) | index($cmd) | not))
    | . + [{"hooks": [{"type": "command", "command": $cmd}]}]
  )
' "$SETTINGS" > /tmp/settings.json.new
mv /tmp/settings.json.new "$SETTINGS"

echo "Installed Stop hook → $HOME/.claude/hooks/hippo-stop.sh"

ln -sf "$REPO/hooks/precompact.sh" "$HOME/.claude/hooks/hippo-precompact.sh"

# Add PreCompact hook entry via jq (idempotent: removes old hippo entry first)
jq --arg cmd "$HOME/.claude/hooks/hippo-precompact.sh" '
  .hooks.PreCompact |= (
    (. // [])
    | map(select((.hooks // []) | map(.command) | index($cmd) | not))
    | . + [{"hooks": [{"type": "command", "command": $cmd}]}]
  )
' "$SETTINGS" > /tmp/settings.json.new
mv /tmp/settings.json.new "$SETTINGS"

echo "Installed PreCompact hook → $HOME/.claude/hooks/hippo-precompact.sh"
