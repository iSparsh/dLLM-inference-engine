"""Qwen2.5-1.5B-Instruct HF (eager + static-cache compiled) AR baseline at batch 1.
Gives the 'HF' cell of the 2x2 and the ms/token to compare with the bandwidth floor (~12-14 ms)."""
import sys, json, torch
from common import *
from bench_dllm import PROMPTS


def run(model, tok, prompt, max_new, **gen_kw):
    text = chat(tok, prompt)
    ids = tok([text], return_tensors="pt").input_ids.cuda()
    t0 = sync_time()
    out = model.generate(ids, max_new_tokens=max_new, min_new_tokens=max_new, do_sample=False, **gen_kw)
    t1 = sync_time()
    n_new = out.shape[1] - ids.shape[1]
    dt = t1 - t0
    # prefill-only time
    t2 = sync_time()
    with torch.no_grad():
        model(ids)
    t3 = sync_time()
    return dict(prompt=ids.shape[1], new_tokens=n_new, secs=dt, tok_s=n_new / dt, ms_per_tok=1e3 * dt / n_new,
                prefill_ms=1e3 * (t3 - t2), text=tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))


def main():
    tok, model = load(AR)
    max_new = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    results = []
    print("mem GB:", torch.cuda.memory_allocated() / 1e9)
    run(model, tok, PROMPTS["intro"], 32)
    for pname, prompt in PROMPTS.items():
        r = run(model, tok, prompt, max_new)
        r.update(prompt_name=pname, mode="eager")
        results.append(r)
        print(f"[eager]   {pname:6s} prompt={r['prompt']} new={r['new_tokens']} {r['tok_s']:.1f} tok/s  {r['ms_per_tok']:.1f} ms/tok  prefill {r['prefill_ms']:.1f} ms")
    # static cache + torch.compile decode (the "fast HF" path)
    try:
        model.generation_config.cache_implementation = "static"
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        run(model, tok, PROMPTS["intro"], 32)
        run(model, tok, PROMPTS["intro"], 32)
        for pname, prompt in PROMPTS.items():
            r = run(model, tok, prompt, max_new)
            r.update(prompt_name=pname, mode="compiled_static")
            results.append(r)
            print(f"[compile] {pname:6s} prompt={r['prompt']} new={r['new_tokens']} {r['tok_s']:.1f} tok/s  {r['ms_per_tok']:.1f} ms/tok")
    except Exception as e:
        print("torch.compile path failed:", repr(e)[:300])
    print("\nsample:", repr(results[0]["text"][:300]))
    json.dump(results, open("../results/ar_timing.json", "w"), indent=1)


if __name__ == "__main__":
    main()
