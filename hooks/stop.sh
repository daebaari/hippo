#!/usr/bin/env bash
# Stop hook for Hippo. Reads JSON from stdin, persists capture + embedding.
# On any failure, exits 0 silently — never block the session.
exec ~/code/hippo/bin/stop-capture "$@"
