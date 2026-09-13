"""One auto-batch-size heuristic, shared by every stage that needs one.

Replaces four near-identical copies (EMICA, HaMeR, ViTPose, HMR2-feature) that
all sized the batch against the card's TOTAL memory, so N concurrent pipelines
each claimed the whole GPU. Lives in hmr4d because that is the lowest-level
package the whole pipeline shares.
"""

from __future__ import annotations

import torch

# Classifies the CARD, so it reads total bytes -- but from mem_get_info, never
# from get_device_properties, so this path cannot drift back to sizing on total.
SMALL_GPU_BYTES = 12e9


def auto_batch_size(
    probe,
    *,
    label: str,
    cap: int,
    device=None,
    target_util: float = 0.85,
    safety_factor: float = 1.0,
    small_gpu_divisor: int = 2,
    small_gpu_target_util: float | None = None,
) -> int:
    """Probe free VRAM and return a batch size for `probe`.

    `probe(n)` runs exactly one forward pass at batch size n; its result is
    discarded. Two-point fit: batch 1 gives the fixed cost, batch 4 the
    marginal cost per sample.

        free      = torch.cuda.mem_get_info(device)[0]   # NOT total_memory
        available = free * target_util - peak1
        bs        = clamp(available / (per_sample * safety_factor), 1, cap)

    `free` is what this process can still allocate now, so a co-tenant pipeline
    shrinks our batch instead of being ignored. free <= total always, so this
    cannot OOM anywhere the old total_memory form fit.

    `safety_factor` inflates the measured per-sample cost for stages whose real
    forward exceeds the probe (ViTPose's flip test doubles the true batch).
    `small_gpu_divisor` applies below SMALL_GPU_BYTES.
    """
    if device is None:
        device = torch.device("cuda")
    # mem_get_info needs an INDEX; a bare torch.device("cuda") raises ValueError.
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(index)
    with torch.no_grad():
        probe(1)
    peak1 = torch.cuda.max_memory_allocated(index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(index)

    test_bs = 4
    with torch.no_grad():
        probe(test_bs)
    peak4 = torch.cuda.max_memory_allocated(index)
    per_sample = (peak4 - peak1) / (test_bs - 1)
    torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info(index)
    small = total < SMALL_GPU_BYTES
    util = small_gpu_target_util if (small and small_gpu_target_util is not None) else target_util
    available = free * util - peak1
    optimal = max(1, min(cap, int(available / max(per_sample * safety_factor, 1))))
    if small:
        optimal = max(1, optimal // small_gpu_divisor)

    print(
        f"  [Auto BS] {label}: {free/1e9:.1f}GB free of {total/1e9:.1f}GB, "
        f"fixed={peak1/1e6:.0f}MB, {per_sample/1e6:.0f}MB/sample -> batch_size={optimal}"
    )
    return optimal
