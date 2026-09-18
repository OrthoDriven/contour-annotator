#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib.sh"
load_app_env "$SCRIPT_DIR"

cd "$APP_DIR"

STATE_PATH="$INSTALL_ROOT/update_state.json"
TEMP_DIR="${TEMP_DIR:-$SCRIPT_DIR/.update_tmp}"

state_init_if_missing "$STATE_PATH"

state_sha="$(state_get "$STATE_PATH" "sha" "")"
state_etag="$(state_get "$STATE_PATH" "etag" "")"
state_last="$(state_get "$STATE_PATH" "lastCheckUtc" "1970-01-01T00:00:00Z")"

now_iso="$(utc_now_iso)"
delta=$(($(utc_to_epoch "$now_iso") - $(utc_to_epoch "$state_last")))

echo "[debug] lastCheckUtc = $state_last"
echo "[debug] now          = $now_iso"
echo "[debug] delta        = ${delta} seconds"
echo "[debug] min interval = ${MIN_CHECK_INTERVAL_SECONDS} seconds"

if ((delta < 0)); then
    echo "[warn] lastCheckUtc is in the future; resetting lastCheckUtc to now."
    state_touch_lastcheck "$STATE_PATH"
    delta=0
fi

if ((delta < MIN_CHECK_INTERVAL_SECONDS)); then
    remaining=$((MIN_CHECK_INTERVAL_SECONDS - delta))
    echo "[info] Skipping update check (${delta}s since last, ${remaining}s remaining)"
    exit 0
fi

# From here on, always touch lastCheckUtc even if we bail.
trap 'state_touch_lastcheck "$STATE_PATH"' EXIT

rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

cd "$APP_DIR"

resp="$(github_latest_sha_etag "$state_etag" "$TEMP_DIR")"
IFS='|' read -r http_code new_sha new_etag <<<"$resp"

if [[ "$http_code" == "304" ]]; then
    echo "[info] Already up-to-date ($state_sha)"
    exit 0
fi

if [[ "$http_code" != "200" ]]; then
    echo "[warn] Could not query GitHub (HTTP $http_code)"
    exit 0
fi

if [[ -z "$new_sha" ]]; then
    echo "[warn] GitHub response did not include sha"
    exit 0
fi

if [[ "$state_sha" == "$new_sha" ]]; then
    echo "[info] Already up-to-date ($new_sha)"
    exit 0
fi

echo "[info] New version detected ($new_sha). Updating..."

ZIP_PATH="$TEMP_DIR/annotator.zip"
download_repo_zip "$REPO_ZIP_URL" "$ZIP_PATH"
extract_zip "$ZIP_PATH" "$TEMP_DIR"
rm -f "$ZIP_PATH"

cd "$APP_DIR"

NEW_PROJECT_DIR="$(find_project_root_containing_pixi "$TEMP_DIR")"
[[ -n "$NEW_PROJECT_DIR" ]] || die "Cannot find project root in downloaded archive."

swap_app_dir "$NEW_PROJECT_DIR"

cd "$APP_DIR"

state_set_all "$STATE_PATH" "$new_sha" "${new_etag:-$state_etag}"
rm -rf "$TEMP_DIR"

trap - EXIT
echo "[info] Update complete -> $new_sha"
