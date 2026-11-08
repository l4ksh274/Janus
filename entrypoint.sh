#!/bin/sh
set -eu

cd /app
printf '%s\n' "Downloading and installing Python dependencies from uv.lock..."
uv sync --frozen --no-dev

exec "$@"
