#!/usr/bin/env bash
set -euo pipefail

# Prefer installer-written env file. If it doesn't exist, fail loudly.
DEFAULT_ENV="$HOME/2d-point-annotator/app.env"
if [[ ! -f "$DEFAULT_ENV" ]]; then
    echo "ERROR: Missing $DEFAULT_ENV (not installed?)"
    exit 1
fi

# shellcheck source=/dev/null
source "$DEFAULT_ENV"

: "${APP_DIR:?APP_DIR missing in .app.env}"

export PATH="$PATH:$HOME/.pixi/bin"

# "$APP_DIR/install_scripts/UnixUpdate.sh" || echo "[warn] update check failed"
pixi run -m "$APP_DIR" "$APP_DIR/install_scripts/update.py"

if [[ ! -f "$APP_DIR/pixi.toml" ]]; then
    echo "ERROR: Could not locate pixi.toml under $APP_DIR"
    read -r -p "Press Enter..."
    exit 1
fi

cd "$APP_DIR"
echo "Starting 2D Point Annotator..."
pixi run annotator || {
    rc=$?
    read -r -p "Annotator exited ($rc). Press Enter..."
    exit "$rc"
}
