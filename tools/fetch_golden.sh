#!/usr/bin/env bash
# Get the golden tensors into artifacts/golden/.
#
# Usage, from the repo root:
#   tools/fetch_golden.sh <url>                       download, then extract
#   tools/fetch_golden.sh ~/Downloads/dllm-golden-v1.tar.gz   extract a file you downloaded yourself
#
# The archive (.tar.gz) holds one folder per case at its root: <case>/{meta.json,tensors.safetensors}.
# The team's link is in the group chat. Or regenerate: uv run python tools/dump_golden.py
set -euo pipefail
src=${1:?Give the team link or the path of a downloaded .tar.gz}
if [[ -f "$src" ]]; then
  archive=$src
else
  archive=$(mktemp)
  trap 'rm -f "$archive"' EXIT
  curl --fail --location --retry 2 "$src" -o "$archive"
fi
if ! gzip -t "$archive" 2>/dev/null; then
  echo "That is not a .tar.gz archive. If the link opened a web page, download the file in your browser and pass its path instead." >&2
  exit 1
fi
mkdir -p artifacts/golden
tar -xzf "$archive" -C artifacts/golden
echo "Extracted $(ls artifacts/golden | wc -l | tr -d ' ') golden cases into artifacts/golden/"
