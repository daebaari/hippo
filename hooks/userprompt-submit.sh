#!/usr/bin/env bash
# UserPromptSubmit hook for Hippo. Reads JSON from stdin, prints memory block to stdout.
# On any failure, exits 0 silently — never block the user's prompt.
exec ~/code/hippo/bin/userprompt-retrieve "$@"
