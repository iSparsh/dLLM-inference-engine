"""Stage-1 driver: the reference decode loop on static storage, one CUDA Graph per pass.

`Engine.generate` follows `dllm.reference_decode.decode` (docs/INTERFACES.md §3.6) step
for step, and `Engine.ar_generate` follows `ar_decode`. The differences are where the
state lives and how a pass runs:

- K/V live in `SinglePass`'s static cache. The host tracks the committed length and
  writes it to the device-side `q_start` before each pass.
- The current block's 32 tokens live in a static device buffer that `select_commit`
  updates in place, so a denoise pass needs no host round trip for the tokens.
- Denoise, commit and AR passes are captured as CUDA Graphs, one per attention length
  bucket (and per tau and op backend set). Prefill runs eagerly.
- Each pass ends with one 4-int device-to-host copy, the only synchronization:
  `[n_committed, window_done, block_done, stop]`.

The stock `block_attention` reads every cache slot it is given, so each pass sees only
the first `bucket * ceil((q_start + M) / bucket)` slots. Slots past `q_start + M` are
masked either way, so the bucket changes the cost, not the result.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedModel

from .. import ops
from ..config import BLOCK, EOS, MASK, SUB_BLOCK
from .eos import eos_ready
from .single_pass import SinglePass


@dataclass
class Stats:
    """Counts for the last generation. passes excludes prefill."""

    prompt_tokens: int = 0
    gen_tokens: int = 0
    passes: int = 0
    denoise: int = 0
    commit: int = 0
    captures: int = 0


class Engine:
    """Batch-one decoder for one model. mode is "dllm" (Fast-dLLM v2) or "ar" (Qwen2.5)."""

    @torch.inference_mode()
    def __init__(
        self,
        model: PreTrainedModel,
        mode: str = "dllm",
        capacity: int = 4096,
        bucket: int = 256,
        graphs: bool = True,
    ) -> None:
        if mode not in ("dllm", "ar"):
            raise ValueError("Choose mode='dllm' or 'ar'.")
        self.mode, self.capacity, self.bucket = mode, capacity, bucket
        self.net = SinglePass(model, capacity, BLOCK if mode == "dllm" else 1)
        device = self.net.start.device
        self.graphs = graphs and device.type == "cuda"
        long = dict(device=device, dtype=torch.long)
        # Static inputs the captured passes read; the host only ever copies into them.
        self.ids = torch.full((BLOCK if mode == "dllm" else 1,), MASK, **long)
        self.window = torch.zeros(2, device=device, dtype=torch.int32)
        self.windows = torch.tensor(
            [[lo, lo + SUB_BLOCK] for lo in range(0, BLOCK, SUB_BLOCK)],
            device=device,
            dtype=torch.int32,
        )
        self.gen_start = torch.zeros(1, **long)
        self.seed = torch.zeros(1, **long)
        if mode == "dllm":
            stops = [EOS]  # the reference stops on <|im_end|> only
        else:
            eos = model.generation_config.eos_token_id
            stops = [eos] if isinstance(eos, int) else list(eos)
        self.stop_ids = torch.tensor(stops, **long)
        self.host = torch.zeros(4, dtype=torch.int32, pin_memory=self.graphs)
        self.pool = torch.cuda.graph_pool_handle() if self.graphs else None
        self.captured: dict[tuple, tuple[torch.cuda.CUDAGraph, Tensor]] = {}
        self.stats = Stats()

    # Pass bodies. Each reads the static buffers and returns int32[4] status on the device.

    def _denoise(self, length: int, tau: float) -> Tensor:
        logits = self.net.forward(self.ids, length)
        stats = ops.select_commit(logits, self.ids, self.window, tau, MASK)
        stop = eos_ready(self.ids, self.gen_start, MASK, self.stop_ids)
        return torch.cat((stats, stop.int()))

    def _commit(self, length: int) -> Tensor:
        logits = self.net.forward(self.ids, length)
        self.seed.copy_(logits[-1].argmax().view(1))
        stop = (self.seed == EOS).int()
        return torch.cat((stop.new_zeros(3), stop))

    def _ar_step(self, length: int) -> Tensor:
        logits = self.net.forward(self.ids, length, last_only=True)
        self.ids.copy_(logits[-1].argmax().view(1))
        stop = torch.isin(self.ids, self.stop_ids).int()
        return torch.cat((stop.new_zeros(3), stop))

    def _length(self, q_start: int, rows: int) -> int:
        end = q_start + rows
        if end > self.capacity:
            raise ValueError(
                f"Sequence needs {end} cache slots; raise capacity above {self.capacity}."
            )
        return min(self.capacity, -(-end // self.bucket) * self.bucket)

    def _run(self, key: tuple, body: Callable[[], Tensor]) -> list[int]:
        """Run one pass (replaying its graph if enabled) and return its status on the host."""
        if not self.graphs:
            out = body()
        else:
            graph, out = self._graph(key, body)
            graph.replay()
        self.host.copy_(out, non_blocking=self.graphs)
        if self.graphs:
            torch.cuda.current_stream().synchronize()
        return self.host.tolist()

    def _graph(self, key: tuple, body: Callable[[], Tensor]) -> tuple:
        """Captured graph for key under the current op backends, capturing on first use."""
        key = (*key, tuple(sorted(ops.available().items())))
        if key not in self.captured:
            self.captured[key] = self._capture(body)
        return self.captured[key]

    @torch.inference_mode()
    def prepare(self, total: int, tau: float = 0.9) -> None:
        """Capture every graph a sequence of up to total tokens needs, outside any timing.

        Call before generating (warm-up passes write junk into cache slots that the next
        prefill or commit overwrites before anything reads them). Without it, graphs are
        captured lazily the first time a length bucket is reached.
        """
        if not self.graphs:
            return
        rows = len(self.ids)
        lengths = sorted({self._length(q, rows) for q in range(total - rows + 1)})
        for length in lengths:
            self.net.start.fill_(length - rows)
            if self.mode == "ar":
                self._graph(("ar", length), lambda: self._ar_step(length))
                continue
            self.window.zero_()
            self._graph(("denoise", length, tau), lambda: self._denoise(length, tau))
            self._graph(("commit", length), lambda: self._commit(length))

    def _capture(self, body: Callable[[], Tensor]) -> tuple:
        """Warm up and capture body without changing the visible state.

        Warm-up runs execute for real, so restore the token buffer each body may mutate.
        Their cache writes go to the pass's own slots and the replay rewrites them.
        """
        saved = self.ids.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
                self.ids.copy_(saved)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = body()
        self.ids.copy_(saved)
        self.stats.captures += 1
        return graph, out

    def _prefill(self, ids: Tensor, last_only: bool = False) -> Tensor:
        """Eager prefill with HF's unpacked MLP GEMM shapes (see SinglePass.forward)."""
        self.net.start.fill_(0)
        length = self._length(0, len(ids))
        return self.net.forward(ids, length, last_only, packed=False)

    @torch.inference_mode()
    def generate(self, ids: Tensor, max_new_tokens: int = 64, tau: float = 0.9) -> Tensor:
        """Same contract and output as reference_decode.decode."""
        if self.mode != "dllm":
            raise ValueError("generate() is the dLLM loop; use ar_generate() in AR mode.")
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("Pass one nonempty prompt as a 1-D token tensor.")
        n = len(ids)
        seq = ids.new_full((n + (max_new_tokens // BLOCK + 2) * BLOCK,), MASK)
        seq[:n] = ids
        size, cached, stopped = n, 0, False
        self.stats = Stats(prompt_tokens=n, captures=self.stats.captures)
        if n > BLOCK:
            cached = n // BLOCK * BLOCK
            logits = self._prefill(seq[:cached])
            if n % BLOCK == 0:
                seq[size] = logits[-1].argmax()
                stopped = int(seq[size]) == EOS
                size += 1
        for _ in range(max_new_tokens // BLOCK):
            if stopped:
                break
            start = size
            size += BLOCK - start % BLOCK
            first = size - BLOCK  # block-aligned start of this block in seq
            filled = start - first  # prompt tail or seed tokens ahead of the MASKs
            self.ids.copy_(seq[first:size])
            self.gen_start.fill_(filled)
            self.net.start.fill_(cached)
            length = self._length(cached, BLOCK)
            for w, lo in enumerate(range(0, BLOCK, SUB_BLOCK)):
                if lo + SUB_BLOCK <= filled:
                    continue  # no MASK in this window
                self.window.copy_(self.windows[w])
                while True:
                    _, window_done, _, stopped = self._run(
                        ("denoise", length, tau), lambda: self._denoise(length, tau)
                    )
                    self.stats.denoise += 1
                    if stopped or window_done:
                        break
                if stopped:
                    break
            seq[first:size] = self.ids
            if stopped:
                break
            *_, stopped = self._run(("commit", length), lambda: self._commit(length))
            self.stats.commit += 1
            seq[size] = self.seed[0]
            size += 1
            cached += BLOCK
        self.stats.passes = self.stats.denoise + self.stats.commit
        out = seq[:size]
        hits = (out[n:] == EOS).nonzero()
        out = out[: n + int(hits[0, 0]) + 1] if len(hits) else out
        self.stats.gen_tokens = len(out) - n
        return out

    @torch.inference_mode()
    def ar_generate(self, ids: Tensor, max_new_tokens: int = 64) -> Tensor:
        """Greedy decoding; same output as reference_decode.ar_decode."""
        if self.mode != "ar":
            raise ValueError("ar_generate() needs an Engine built with mode='ar'.")
        n = len(ids)
        seq = ids.new_empty(n + max_new_tokens)
        seq[:n] = ids
        self.stats = Stats(prompt_tokens=n, captures=self.stats.captures)
        self.ids.copy_(self._prefill(ids, last_only=True)[-1].argmax().view(1))
        seq[n] = self.ids[0]
        stopped = bool(torch.isin(self.ids, self.stop_ids))
        pos = n  # index in seq of the newest token, which is also its cache slot
        while not stopped and pos - n + 1 < max_new_tokens:
            self.net.start.fill_(pos)
            length = self._length(pos, 1)
            *_, stopped = self._run(("ar", length), lambda: self._ar_step(length))
            self.stats.passes += 1
            pos += 1
            seq[pos] = self.ids[0]
        self.stats.gen_tokens = pos - n + 1
        return seq[: pos + 1]
