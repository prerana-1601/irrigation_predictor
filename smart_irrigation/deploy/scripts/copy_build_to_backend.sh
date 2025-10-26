
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND="$ROOT/frontend"
BACKEND="$ROOT/backend"
DIST="$FRONTEND/dist"

if [ ! -d "$DIST" ]; then
  echo "Build not found. Run: (cd $FRONTEND && npm run build)"
  exit 1
fi

# Copy dist into backend/frontend/dist
mkdir -p "$BACKEND/../frontend/dist"
rsync -a --delete "$DIST/" "$BACKEND/../frontend/dist/"
echo "Copied SPA build to backend/../frontend/dist (Flask will serve it)."
