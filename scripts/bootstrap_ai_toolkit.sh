#!/usr/bin/env bash
set -euo pipefail

target="${1:-$PWD/vendor/ai-toolkit}"
ref="${2:-main}"

if [[ -e "$target" ]]; then
  echo "Refusing to overwrite existing path: $target" >&2
  exit 2
fi

git clone https://github.com/ostris/ai-toolkit.git "$target"
git -C "$target" checkout "$ref"
git -C "$target" submodule update --init --recursive

revision="$(git -C "$target" rev-parse HEAD)"
echo "AI Toolkit source installed at $target"
echo "Revision: $revision"
echo "Follow the setup instructions in that AI Toolkit checkout, then run Saturnia with:"
echo "  PYTHONPATH=src python -m saturnia_lora run --experiment 01_reference_only --model flux2_klein_4b --ai-toolkit-dir $target"
