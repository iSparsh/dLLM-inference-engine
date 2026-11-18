#!/usr/bin/env bash
# Usage: tools/fetch_golden.sh https://github.com/.../releases/download/.../golden.tar.gz
# Archive layout: <case>/{meta.json,tensors.safetensors} relative to its root.
set -euo pipefail
url=${1:?Supply the team release URL for golden.tar.gz}
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
curl --fail --location --retry 2 "$url" -o "$tmp"
python - "$tmp" <<'PY'
import sys, tarfile
from pathlib import Path
root = Path("artifacts/golden")
root.mkdir(parents=True,exist_ok=True)
with tarfile.open(sys.argv[1], "r:gz") as f:
    f.extractall(root,filter="data")
PY
