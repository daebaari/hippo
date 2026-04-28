#!/usr/bin/env bash
# Remove Hippo hooks + launchd agents. Memory data at ~/.claude/memory/
# and ~/.claude/projects/*/memory/ is preserved.
set -euo pipefail

USER="$(whoami)"

echo "==> Hippo uninstaller (memory data preserved)"

# Unload + remove launchd plists
for label in memory-daemon dream-heavy; do
  plist="$HOME/Library/LaunchAgents/com.${USER}.${label}.plist"
  if [ -f "$plist" ]; then
    launchctl unload "$plist" 2>/dev/null || true
    rm -f "$plist"
    echo "  removed $plist"
  fi
done

# Remove hook symlinks
for h in hippo-userprompt-submit.sh hippo-stop.sh hippo-precompact.sh; do
  rm -f "$HOME/.claude/hooks/$h"
done

# Remove slash command symlink
rm -f "$HOME/.claude/commands/hippo-dream.md"

# Strip hippo entries from settings.json
SETTINGS="$HOME/.claude/settings.json"
if [ -f "$SETTINGS" ]; then
  BACKUP="${SETTINGS}.pre-hippo-uninstall-$(date +%Y%m%d-%H%M%S)"
  cp "$SETTINGS" "$BACKUP"
  jq '
    .hooks.UserPromptSubmit |= ((. // []) | map(select((.hooks // []) | map(.command) | map(test("hippo-")) | any | not)))
    | .hooks.Stop          |= ((. // []) | map(select((.hooks // []) | map(.command) | map(test("hippo-")) | any | not)))
    | .hooks.PreCompact    |= ((. // []) | map(select((.hooks // []) | map(.command) | map(test("hippo-")) | any | not)))
  ' "$BACKUP" > "$SETTINGS"
  echo "  stripped hippo entries from $SETTINGS (backup: $BACKUP)"
fi

echo
echo "Uninstall complete. Memory data at:"
echo "  ~/.claude/memory/                     (global)"
echo "  ~/.claude/projects/*/memory/          (per-project)"
echo "is preserved. Delete manually if desired."
