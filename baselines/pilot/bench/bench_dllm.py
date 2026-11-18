"""Fast-dLLM v2 1.5B reference-implementation timing.
Tests plan claims: tokens/step > 1 at threshold 0.9; ms/step vs bandwidth floor; M per step;
how much of the 2.5x speedup survives at batch 1 on a laptop GPU.
"""
import sys, json, torch
from common import *

PROMPTS = {
    "gsm8k": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Think step by step, then give the final answer after '####'.",
    "intro": "Give me a short introduction to large language models.",
}


def run(model, tok, prompt, max_new, threshold, small_block, use_block_cache):
    text = chat(tok, prompt)
    ids = tok([text], return_tensors="pt").input_ids.cuda()
    fc = ForwardCounter(model)
    t0 = sync_time()
    out = model.generate(ids, tokenizer=tok, max_new_tokens=max_new, small_block_size=small_block,
                         threshold=threshold, use_block_cache=use_block_cache)
    t1 = sync_time()
    fc.restore()
    new = out[0, ids.shape[1]:]
    n_new = int(new.numel())
    calls = fc.calls
    dt = t1 - t0
    return dict(prompt=ids.shape[1], new_tokens=n_new, forwards=len(calls), m_hist={m: calls.count(m) for m in sorted(set(calls))},
                tok_per_fwd=n_new / max(1, len(calls)), ms_per_fwd=1e3 * dt / max(1, len(calls)),
                tok_s=n_new / dt, secs=dt, text=tok.decode(new, skip_special_tokens=True))


def main():
    tok, model = load(DLLM, remote=True)
    max_new = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    # warm up (flex-attention / triton compile)
    run(model, tok, PROMPTS["intro"], 64, 0.9, 8, False)
    run(model, tok, PROMPTS["intro"], 64, 0.9, 8, True)
    results = []
    print(f"{'prompt':6s} {'thr':5s} {'sb':3s} {'bcache':6s} | {'new':4s} {'fwd':4s} {'tok/fwd':7s} {'ms/fwd':7s} {'tok/s':6s} | M histogram")
    for pname, prompt in PROMPTS.items():
        for use_bc in (False, True):
            for thr in (1.0, 0.95, 0.9, 0.7, 0.5):
                r = run(model, tok, prompt, max_new, thr, 8, use_bc)
                r.update(prompt_name=pname, threshold=thr, small_block=8, block_cache=use_bc)
                results.append(r)
                print(f"{pname:6s} {thr:<5} {8:<3} {str(use_bc):6s} | {r['new_tokens']:4d} {r['forwards']:4d} {r['tok_per_fwd']:7.2f} {r['ms_per_fwd']:7.1f} {r['tok_s']:6.1f} | {r['m_hist']}")
        # sub-block size sweep at thr 0.9 with block cache
        for sb in (4, 16, 32):
            r = run(model, tok, prompt, max_new, 0.9, sb, True)
            r.update(prompt_name=pname, threshold=0.9, small_block=sb, block_cache=True)
            results.append(r)
            print(f"{pname:6s} {0.9:<5} {sb:<3} {'True':6s} | {r['new_tokens']:4d} {r['forwards']:4d} {r['tok_per_fwd']:7.2f} {r['ms_per_fwd']:7.1f} {r['tok_s']:6.1f} | {r['m_hist']}")
    print("\n--- sample outputs (thr=0.9, sb=8, block cache) ---")
    for r in results:
        if r["threshold"] == 0.9 and r["small_block"] == 8 and r["block_cache"]:
            print(f"[{r['prompt_name']}] {r['text'][:400]!r}\n")
    print("--- sample output gsm8k thr=0.5 ---")
    for r in results:
        if r["threshold"] == 0.5 and r["prompt_name"] == "gsm8k" and r["block_cache"]:
            print(repr(r["text"][:400]))
    json.dump(results, open("../results/dllm_timing.json", "w"), indent=1)


if __name__ == "__main__":
    main()
