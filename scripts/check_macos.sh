#!/usr/bin/env bash
# macOS environment check for the dLLM project.
#
# Run from the repo root on the Mac:     bash scripts/check_macos.sh
# Skip the 3 GB model download/test:     bash scripts/check_macos.sh --skip-model
#
# It collects everything we need to know about this Mac in one go and writes it to
# mac-env-report-<timestamp>.txt in the repo root. Send that file to the team.
# What it changes on your machine: installs uv into ~/.local/bin if it is missing,
# creates the project's .venv (uv sync), and (unless --skip-model) downloads the
# dLLM checkpoint (~3.1 GB) into ~/.cache/huggingface. Nothing else.

set -u
SKIP_MODEL=0
for arg in "$@"; do
  case "$arg" in
    --skip-model) SKIP_MODEL=1 ;;
    *) echo "unknown option: $arg (valid: --skip-model)"; exit 2 ;;
  esac
done

[[ -f pyproject.toml && -d dllm ]] || { echo "Run this from the repo root (the folder with pyproject.toml)."; exit 1; }
REPORT="mac-env-report-$(date +%Y%m%d-%H%M%S).txt"
exec > >(tee "$REPORT") 2>&1

section() { echo; echo "===== $* ====="; }
run() { echo "\$ $*"; "$@" || echo "FAILED (rc=$?): $*"; }

echo "dLLM macOS check, $(date)"

section "SYSTEM"
run sw_vers
run uname -m
echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo "memory: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
# 1 means this shell runs under Rosetta (x86_64 emulation): PyTorch would then be the wrong build.
echo "rosetta-translated shell: $(sysctl -in sysctl.proc_translated 2>/dev/null || echo unknown)"
echo "shell: $SHELL"
run df -h .

section "DEVELOPER TOOLS"
run xcode-select -p
run clang --version
run git --version
echo "homebrew: $(command -v brew || echo 'not installed (not required)')"
for p in python3 python3.12; do echo "$p: $(command -v $p || echo -) $($p --version 2>&1 || true)"; done

section "UV"
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  echo "uv not found; installing it into ~/.local/bin with the official installer"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$HOME/.local/bin:$PATH"
run uv --version
run uv python list --only-installed

section "REPO"
run git rev-parse HEAD
run git status --short

section "UV SYNC (locked)"
START=$(date +%s)
run uv sync --locked
echo "uv sync took $(( $(date +%s) - START )) s"
run uv pip list

section "IMPORTS AND REFERENCE OPS (CPU and MPS)"
uv run python - <<'PY' || echo "FAILED: python import/op check"
import platform, time, traceback
print("python", platform.python_version(), "mac", platform.mac_ver()[0], "machine", platform.machine())
for mod in ["torch", "transformers", "huggingface_hub", "einops", "safetensors", "numpy", "pyarrow", "matplotlib", "accelerate"]:
    try:
        m = __import__(mod); print(f"{mod:16s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:16s} IMPORT FAILED: {e!r}")
import torch
print("mps built", torch.backends.mps.is_built(), "mps available", torch.backends.mps.is_available())
print("cuda available", torch.cuda.is_available(), "(expected False on a Mac)")
from dllm import ops
from dllm.ops import reference as R
try:
    from dllm.ops import cuda
    cuda.load()
    print("ours backend: loaded (unexpected on a Mac)")
except Exception as e:
    print("ours backend: clean error as expected ->", str(e)[:120])

def check(device):
    bf = torch.bfloat16
    g = torch.Generator().manual_seed(0)
    t = lambda *s: torch.randn(*s, generator=g).to(bf).to(device)
    out = {}
    st = time.perf_counter()
    out["silu_mul"] = R.silu_mul(t(32, 17920)).shape
    res = t(32, 1536); out["add_rmsnorm"] = R.add_rmsnorm(t(32, 1536), res, t(1536), 1e-6).shape
    L = 512
    kc, vc = t(2, L, 128), t(2, L, 128)
    qs = torch.tensor([96], dtype=torch.int32, device=device)
    q = t(32, 12, 128)
    cos, sin = t(L, 128), t(L, 128)
    R.rope_kv_write(q, t(32, 2, 128), t(32, 2, 128), kc, vc, qs, cos, sin)
    out["block_attention"] = R.block_attention(q, kc, vc, qs, 32).shape
    tokens = torch.full((32,), 151665, dtype=torch.int64, device=device)
    window = torch.tensor([0, 8], dtype=torch.int32, device=device)
    out["select_commit"] = R.select_commit(t(32, 151936), tokens, window, 0.9, 151665).tolist()
    if device == "mps": torch.mps.synchronize()
    print(f"[{device}] reference ops OK in {1000*(time.perf_counter()-st):.0f} ms: {out}")

for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
    try:
        check(dev)
    except Exception:
        print(f"[{dev}] reference ops FAILED"); traceback.print_exc()
PY

section "SMOKE TEST"
run uv run python tools/smoke_test.py

section "PYTEST"
run uv run pytest -q

if [[ $SKIP_MODEL -eq 0 ]]; then
  section "MODEL DOWNLOAD (dLLM, ~3.1 GB)"
  START=$(date +%s)
  run uv run python tools/download_models.py dllm
  echo "download took $(( $(date +%s) - START )) s"

  section "SHORT GENERATION ON CPU AND MPS"
  uv run python - <<'PY' || echo "FAILED: generation check"
import time, traceback, torch
from dllm.models import load, prompt_ids
from dllm.reference_decode import decode
for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
    try:
        st = time.perf_counter()
        model, tok = load("dllm", device=dev, local_only=True)
        print(f"[{dev}] loaded in {time.perf_counter()-st:.1f} s")
        ids = prompt_ids(tok, "Mia buys three bags of four pears. How many pears does she buy?", model.device)
        st = time.perf_counter()
        out = decode(model, ids, 32, 0.9)
        if dev == "mps": torch.mps.synchronize()
        dt = time.perf_counter() - st
        text = tok.decode(out[len(ids):])
        print(f"[{dev}] generated {len(out)-len(ids)} tokens in {dt:.1f} s: {text!r}")
        del model
    except Exception:
        print(f"[{dev}] generation FAILED"); traceback.print_exc()
PY
else
  section "MODEL TEST SKIPPED (--skip-model)"
fi

section "DONE"
echo "Report written to $PWD/$REPORT. Send this file to the team."
