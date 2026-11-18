"""CUDA-event medians and best-effort NVML telemetry."""

import statistics
import time
from collections.abc import Callable

import torch


def gpu_state() -> dict:
    info = {
        "gpu": None,
        "driver": None,
        "clocks": {"sm_mhz": None, "mem_mhz": None, "power_limit_w": None},
    }
    if not torch.cuda.is_available():
        return info
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    info["gpu"] = props.name
    try:
        import pynvml as nv

        nv.nvmlInit()
        try:
            h = nv.nvmlDeviceGetHandleByUUID(str(props.uuid))
            info["driver"] = nv.nvmlSystemGetDriverVersion()
            clocks = info["clocks"]
            clocks["sm_mhz"] = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
            clocks["mem_mhz"] = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)
            try:
                clocks["power_limit_w"] = nv.nvmlDeviceGetPowerManagementLimit(h) / 1000
            except nv.NVMLError_NotSupported:
                clocks["default_power_limit_w"] = (
                    nv.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000
                )
        finally:
            nv.nvmlShutdown()
    except ImportError:
        pass
    except nv.NVMLError:
        pass
    return info


def benchmark(
    fn: Callable,
    repeats: int = 20,
    prepare: Callable | None = None,
    warmup_s: float = 1.0,
) -> float:
    """Return median milliseconds. prepare restores mutable inputs outside the timing interval.

    An idle GPU sits at low clocks (e.g. 210 MHz SM / 405 MHz memory on a laptop) and takes
    a while to ramp up, which makes short microbenchmarks look 10x slower than they are.
    So fn first runs back to back for warmup_s seconds of wall time (at least 5 times).
    """
    if repeats < 20:
        raise ValueError("Use at least 20 measured iterations.")
    start, n = time.perf_counter(), 0
    while n < 5 or time.perf_counter() - start < warmup_s:
        if prepare:
            prepare()
        fn()
        n += 1
        if n % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        if prepare:
            prepare()
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples)
