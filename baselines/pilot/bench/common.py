import time, json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DLLM = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
AR = "Qwen/Qwen2.5-1.5B-Instruct"
MASK_ID = 151665


def load(name, remote=False):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=remote)
    kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=remote)
    model = AutoModelForCausalLM.from_pretrained(name, **kw).cuda().eval()
    return tok, model


class ForwardCounter:
    """Wrap model.forward to count calls and record the number of query tokens (M) per call."""

    def __init__(self, model):
        self.model = model
        self.calls = []
        self._orig = model.forward

        def wrapped(*a, **k):
            ids = k.get("input_ids", a[0] if a else None)
            self.calls.append(int(ids.shape[1]) if ids is not None else -1)
            return self._orig(*a, **k)

        model.forward = wrapped

    def reset(self):
        self.calls = []

    def restore(self):
        self.model.forward = self._orig


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def chat(tok, user, system="You are a helpful assistant.", tools=None):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = {"tools": tools} if tools else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def gpu_power_w():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
    except Exception:
        return float("nan")
