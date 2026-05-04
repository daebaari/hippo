#!/usr/bin/env bash
# One-shot installer for Hippo on a fresh macOS machine.
#
# Idempotent: every step is safe to re-run. Doesn't touch existing
# memory data; only manages hooks, launchd, and Claude Code settings.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Hippo installer"
echo "    Repo: $REPO"
echo

# --- Step 1: deps ---
echo "==> Syncing Python deps via uv..."
cd "$REPO"
uv sync
echo

# --- Step 2: schema migrations on global store ---
echo "==> Initializing global memory store..."
uv run python -c "
from hippo.storage.multi_store import Scope, open_store
s = open_store(Scope.global_())
print(f'Global store ready at {s.memory_dir}')
s.conn.close()
"
echo

# --- Step 3: daemon plist + load ---
echo "==> Installing daemon launchd agent..."
"$REPO/scripts/install-daemon.sh"
echo

# --- Step 4: dream-heavy plist + load ---
echo "==> Installing dream-heavy launchd agent..."
"$REPO/scripts/install-dream.sh"
echo

# --- Step 5: hooks + slash commands ---
echo "==> Installing hooks into ~/.claude/settings.json..."
"$REPO/scripts/install-hooks.sh"
echo

# --- Step 5.5: bin/ on PATH (so `hippo dream`, `hippo status`, etc. work) ---
echo "==> Adding $REPO/bin to PATH"
PATH_LINE="export PATH=\"$REPO/bin:\$PATH\""
case "${SHELL:-/bin/zsh}" in
  *zsh)  RC="$HOME/.zshrc" ;;
  *bash) RC="$HOME/.bashrc" ;;
  *)     RC="" ;;
esac
if [ -z "$RC" ]; then
  echo "    Unrecognized shell ($SHELL). Add this line to your shell rc manually:"
  echo "      $PATH_LINE"
elif [ -f "$RC" ] && grep -Fq "$REPO/bin" "$RC"; then
  echo "    Already present in $RC — skipping."
else
  printf '\n# hippo memory CLI\n%s\n' "$PATH_LINE" >> "$RC"
  echo "    Appended to $RC. Run \`source $RC\` or open a new shell to pick it up."
fi
echo

# --- Step 6: bootstrap migration prompt ---
echo "==> Bootstrap migration"
read -r -p "Run bootstrap migration over an existing memory dir now? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
  read -r -p "Memory dir [~/.claude/projects/-Users-keon-kaleon-kaleon/memory]: " mem_dir
  mem_dir="${mem_dir:-$HOME/.claude/projects/-Users-keon-kaleon-kaleon/memory}"
  read -r -p "Project name [kaleon]: " proj
  proj="${proj:-kaleon}"
  "$REPO/bin/dream-bootstrap" --memory-dir "$mem_dir" --scope "$proj"
fi
echo

# --- Verification ---
echo "==> Post-install verification"
launchctl list | grep -E "(memory-daemon|dream-heavy)" || echo "  (launchd agents not visible — check System Settings → Background Items)"
uv run memory-stats || true
echo
echo "Hippo install complete."
echo "Open a new Claude Code session to start using it."
