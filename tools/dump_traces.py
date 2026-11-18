"""Record per-pass traces (docs/INTERFACES.md §5) for the first N GSM8K-100 prompts.

Uses the same prompt as the accuracy harness (dllm/prompts.py) and the reference decode loop.
Example: uv run python tools/dump_traces.py --n 30 --tau 0.9 0.7 --out traces/n30.jsonl
"""

import argparse
import json
import uuid
from pathlib import Path

from dllm.models import load
from dllm.prompts import gsm8k_prompt_ids
from dllm.reference_decode import decode
from dllm.trace import TraceWriter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3, help="number of GSM8K-100 prompts (1..100)")
    p.add_argument("--tau", type=float, nargs="+", default=[0.9], help="one or more thresholds")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--output", "--out", type=Path, help="JSONL file (default: traces/<run>.jsonl)")
    p.add_argument("--append", action="store_true", help="append to an existing output file")
    a = p.parse_args()
    if not 1 <= a.n <= 100:
        p.error("--n must be between 1 and 100")
    run = uuid.uuid4().hex
    path = a.output or Path("traces") / f"{run}.jsonl"
    if path.exists() and not a.append:
        p.error(f"{path} exists; choose a new file or pass --append")
    model, tok = load()
    items = [json.loads(l) for l in Path("eval/gsm8k_100.jsonl").read_text().splitlines()[: a.n]]
    for tau in a.tau:
        for item in items:
            writer = TraceWriter(path, run_id=run, prompt_id=item["id"], model="dllm", tau=tau)
            ids = gsm8k_prompt_ids(tok, item["question"], model.device)
            decode(model, ids, a.max_new_tokens, tau, trace=writer)
            print(f"tau={tau} {item['id']} done", flush=True)
    print(path)


if __name__ == "__main__":
    main()
