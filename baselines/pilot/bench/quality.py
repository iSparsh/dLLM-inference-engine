"""Quality sanity probe (plan section 5 'risk check'): BFCL simple_python + GSM8K on both models.
Scores: GSM8K exact match on the number after '####'; BFCL: JSON-valid <tool_call>, name match,
argument match against possible answers (BFCL AST-style, simplified)."""
import sys, json, re, time, torch
from common import *

N_BFCL = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_GSM = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MODEL = sys.argv[3] if len(sys.argv) > 3 else "both"


def load_bfcl():
    qs = [json.loads(l) for l in open("../data/BFCL_v4_simple_python.json")]
    ans = {a["id"]: a["ground_truth"] for a in map(json.loads, open("../data/possible_answer_simple_python.json"))}
    return [(q, ans[q["id"]]) for q in qs[:N_BFCL]]


def load_gsm():
    import pyarrow.parquet as pq
    t = pq.read_table("../data/gsm8k_test.parquet").to_pylist()
    return [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in t[:N_GSM]]


def bfcl_tool_schema(fn):
    # BFCL uses "dict" type; convert to JSON-schema-ish for the chat template
    def fix(p):
        if isinstance(p, dict):
            p = {k: fix(v) for k, v in p.items()}
            if p.get("type") == "dict": p["type"] = "object"
            if p.get("type") == "tuple": p["type"] = "array"
            if p.get("type") == "float": p["type"] = "number"
        return p
    return {"type": "function", "function": {"name": fn["name"], "description": fn["description"], "parameters": fix(fn["parameters"])}}


def score_bfcl(text, gt):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    raw = m.group(1) if m else None
    if raw is None:
        # maybe bare JSON
        m2 = re.search(r"\{.*\}", text, re.S)
        raw = m2.group(0) if m2 else None
    if raw is None:
        return dict(json_ok=False, name_ok=False, ast_ok=False)
    try:
        call = json.loads(raw)
    except Exception:
        return dict(json_ok=False, name_ok=False, ast_ok=False)
    name = call.get("name"); args = call.get("arguments", {})
    if isinstance(args, str):
        try: args = json.loads(args)
        except Exception: return dict(json_ok=True, name_ok=False, ast_ok=False)
    gt_name, gt_args = next(iter(gt[0].items()))
    name_ok = (name == gt_name) or (name and name.replace("_", ".") == gt_name) or (name and gt_name.replace(".", "_") == name)
    if not name_ok or not isinstance(args, dict):
        return dict(json_ok=True, name_ok=bool(name_ok), ast_ok=False)
    ok = True
    for k, allowed in gt_args.items():
        if k not in args:
            if "" in allowed: continue
            ok = False; break
        v = args[k]
        def norm(x):
            if isinstance(x, str): return x.strip().lower()
            if isinstance(x, float) and x.is_integer(): return int(x)
            return x
        if norm(v) not in [norm(a) for a in allowed]:
            ok = False; break
    for k in args:
        if k not in gt_args: ok = False
    return dict(json_ok=True, name_ok=True, ast_ok=ok)


def gen(model, tok, text, max_new, is_dllm, threshold):
    ids = tok([text], return_tensors="pt").input_ids.cuda()
    t0 = sync_time()
    if is_dllm:
        fc = ForwardCounter(model)
        out = model.generate(ids, tokenizer=tok, max_new_tokens=max_new, small_block_size=8, threshold=threshold, use_block_cache=True)
        fc.restore(); nf = len(fc.calls)
    else:
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        nf = out.shape[1] - ids.shape[1]
    dt = sync_time() - t0
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), dt, nf, out.shape[1] - ids.shape[1]


def eval_model(name, is_dllm, thresholds):
    tok, model = load(name, remote=is_dllm)
    bfcl = load_bfcl(); gsm = load_gsm()
    res = {}
    for thr in thresholds:
        tag = f"{name.split('/')[-1]}@thr{thr}" if is_dllm else name.split('/')[-1]
        # BFCL
        js = nm = ast = 0; tt = 0; tf = 0; tn = 0; samples = []
        for q, gt in bfcl:
            tools = [bfcl_tool_schema(f) for f in q["function"]]
            text = chat(tok, q["question"][0][0]["content"], tools=tools)
            out, dt, nf, nn = gen(model, tok, text, 128, is_dllm, thr)
            s = score_bfcl(out, gt)
            js += s["json_ok"]; nm += s["name_ok"]; ast += s["ast_ok"]; tt += dt; tf += nf; tn += nn
            if len(samples) < 3: samples.append(out[:200])
        n = len(bfcl)
        res[tag + "/bfcl"] = dict(n=n, json_valid=js / n, name_match=nm / n, ast_acc=ast / n, s_per_item=tt / n, tok_per_fwd=tn / max(1, tf), samples=samples)
        print(f"{tag:40s} BFCL-simple n={n}: JSON-valid {js/n:.0%}  name {nm/n:.0%}  AST acc {ast/n:.0%}  {tt/n:.2f}s/item  tok/fwd {tn/max(1,tf):.2f}")
        for s in samples: print("    ", repr(s))
        # GSM8K
        em = 0; tt = 0; tf = 0; tn = 0; samples = []
        for q, a in gsm:
            text = chat(tok, q + "\nThink step by step, then give the final numeric answer after '####'.")
            out, dt, nf, nn = gen(model, tok, text, 320, is_dllm, thr)
            m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", out)
            pred = m[-1].replace(",", "") if m else (re.findall(r"-?[\d,]*\.?\d+", out)[-1].replace(",", "") if re.findall(r"-?[\d,]*\.?\d+", out) else None)
            try: ok = pred is not None and abs(float(pred) - float(a)) < 1e-6
            except Exception: ok = False
            em += ok; tt += dt; tf += nf; tn += nn
            if len(samples) < 2: samples.append(out[-200:])
        n = len(gsm)
        res[tag + "/gsm8k"] = dict(n=n, em=em / n, s_per_item=tt / n, tok_per_fwd=tn / max(1, tf), new_tok_per_item=tn / n, samples=samples)
        print(f"{tag:40s} GSM8K n={n}: EM {em/n:.0%}  {tt/n:.2f}s/item  {tn/n:.0f} tok/item  tok/fwd {tn/max(1,tf):.2f}")
        for s in samples: print("    ", repr(s))
    del model; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    all_res = {}
    if MODEL in ("both", "ar"):
        all_res.update(eval_model(AR, False, [None]))
    if MODEL in ("both", "dllm"):
        all_res.update(eval_model(DLLM, True, [1.0, 0.9, 0.7]))
    json.dump(all_res, open("../results/quality.json", "w"), indent=1)
