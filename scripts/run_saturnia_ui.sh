#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit_dir="${AI_TOOLKIT_DIR:-$project_dir/../ai-toolkit}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$project_dir"
exec "$toolkit_dir/.venv/bin/python" saturnia_ui.py --host 127.0.0.1 --port "${SATURNIA_UI_PORT:-7860}" --toolkit-dir "$toolkit_dir"
