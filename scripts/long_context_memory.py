"""Token overhead and KV-cache memory at long context lengths, per variant.

Usage:  python scripts/long_context_memory.py <windows_train arrow file> [out_dir]

1. Counts <sum> tokens on every training window to get each variant's summary
   rate (sums per raw token): C (scopes), F (scope parts <= W), B/G (fixed K).
2. Projects, for a context of N raw tokens, how many KV entries each layer must
   keep and what that costs in fp16 for Gemma 4 E2B's cache layout:
   12 sliding layers own KV (1 KV head x 256 dims) and 3 full-attention layers
   own KV (1 x 512); the other 20 layers reuse those caches.

Live KV per layer, with S(N) = <sum> tokens so far (never evicted: every later
raw token may attend to them):
  sliding layers, A:            min(N, W)
  sliding layers, B/F/G:        min(N, W) + S(N)   (F's parts are <= W, B/G chunks <= K)
  full layers, standard mask:   N + S(N)           (plain causal: nothing can be dropped)
  full layers, restrict_full:   min(N, W) + S(N)   (same mask as the sliding layers)
Attention reads (compute): keys read per query, summed over all 35 layers
(28 sliding, 7 full), for one query at position N and for the whole prefill.
C is left out of the projection: a C summary must see its whole scope, so a long
scope stays live until it closes (measured peak 1,193 at 2k vs 521 for B).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scopesum.data import segment_window  # noqa: E402
from scopesum.scope_tracker import track_scopes  # noqa: E402
from scopesum.segmenter import n_fixed_sums  # noqa: E402

W = 512
SLIDING_OWN_KV, FULL_OWN_KV = 12, 3
BYTES_SLIDING = 2 * 1 * 256 * 2  # K and V, 1 head, 256 dims, fp16
BYTES_FULL = 2 * 1 * 512 * 2
LENGTHS = [2_048, 8_192, 32_768, 131_072]


def sum_rates(arrow_path: str) -> dict:
    with pa.memory_map(arrow_path) as src:
        rows = pa.ipc.open_stream(src).read_all().select(["text", "offsets"]).to_pylist()
    raw = scope = capped = headers = seps = 0
    lengths = []
    for w in rows:
        offs = [tuple(o) for o in w["offsets"]]
        raw += len(offs)
        lengths.append(len(offs))
        headers += sum(track_scopes(w["text"], offs).token_is_header)
        seps += sum(1 for a, b in offs if w["text"][a:b].strip() in ("", ".", ",", "?", "!", ";", ":"))
        scope += segment_window(w["text"], offs, "scope").n_sums
        capped += segment_window(w["text"], offs, "scope_capped", scope_span=W).n_sums
    fixed = {k: sum(n_fixed_sums(n, k) for n in lengths) for k in (223, 180)}
    return {"raw_tokens": raw, "windows": len(rows), "C": scope / raw, "F": capped / raw,
            "B": fixed[223] / raw, "G": fixed[180] / raw, "signatures": headers / raw, "separators": seps / raw}


N_SLIDING, N_FULL = 28, 7  # attention is computed in all 35 layers (KV sharing saves memory, not reads)


def reads(n: int, rate: float, restrict: bool, with_sums: bool = True) -> tuple[float, float]:
    """Keys read by attention: (one query at position n, whole prefill), summed over all layers.

    A raw query at position t reads min(t, W) recent raw keys + S(t) <sum> keys in a
    sliding layer, and t keys (+ S(t)) in a full layer unless restrict_full.
    """
    t = np.arange(1, n + 1, dtype=np.float64)
    s = rate * t if with_sums else 0 * t
    local = np.minimum(t, W) + s
    full = local if restrict else t + s
    per_query = N_SLIDING * local[-1] + N_FULL * full[-1]
    prefill = N_SLIDING * local.sum() + N_FULL * full.sum()
    return per_query, prefill


def kv_mb(sliding_entries: float, full_entries: float) -> float:
    return (SLIDING_OWN_KV * BYTES_SLIDING * sliding_entries + FULL_OWN_KV * BYTES_FULL * full_entries) / 2**20


def project(rates: dict) -> list[dict]:
    rows = []
    for n in LENGTHS:
        local = min(n, W)
        q, pf = reads(n, 0.0, restrict=False, with_sums=False)
        rows.append({"N": n, "setting": "A: Gemma default (no <sum>)", "tokens": n, "sum_tokens": 0,
                     "kv_sliding": local, "kv_full": n, "kv_mb": kv_mb(local, n), "reads_per_token": q, "reads_prefill": pf})
        for v in ("F", "G"):
            s = rates[v] * n
            q, pf = reads(n, rates[v], restrict=False)
            rows.append({"N": n, "setting": f"{v}: standard mask (full layers see all)", "tokens": n + s, "sum_tokens": s,
                         "kv_sliding": local + s, "kv_full": n + s, "kv_mb": kv_mb(local + s, n + s), "reads_per_token": q, "reads_prefill": pf})
            q, pf = reads(n, rates[v], restrict=True)
            rows.append({"N": n, "setting": f"{v}: restrict_full (every layer: last W + <sum>)", "tokens": n + s, "sum_tokens": s,
                         "kv_sliding": local + s, "kv_full": local + s, "kv_mb": kv_mb(local + s, local + s), "reads_per_token": q, "reads_prefill": pf})
        # raw tokens kept forever on top of the window: def/class lines (F + signatures) or separators (S)
        for name, kept_rate, sum_rate in [("F + signatures: restrict_full", rates["F"] + rates["signatures"], rates["F"]),
                                          ("S: restrict_full (SepLLM-style separators)", rates["separators"], 0.0)]:
            k = kept_rate * n
            q, pf = reads(n, kept_rate, restrict=True)
            rows.append({"N": n, "setting": name, "tokens": n + sum_rate * n, "sum_tokens": sum_rate * n,
                         "kv_sliding": local + k, "kv_full": local + k, "kv_mb": kv_mb(local + k, local + k), "reads_per_token": q, "reads_prefill": pf})
    return rows


def main():
    arrow = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results/long_context"
    os.makedirs(out, exist_ok=True)
    rates = sum_rates(arrow)
    rows = project(rates)
    with open(os.path.join(out, "long_context_memory.json"), "w") as f:
        json.dump({"window": W, "sum_rates": rates, "rows": rows}, f, indent=1)
    print("sum tokens per 1,000 raw tokens:", {k: round(1000 * rates[k], 2) for k in "BCFG"})
    print(f"{'N':>7}  {'setting':48}  {'tokens':>9}  {'KV full layer':>13}  {'KV MB':>8}  {'reads/token':>11}  {'prefill reads':>13}")
    for r in rows:
        print(f"{r['N']:>7}  {r['setting']:48}  {r['tokens']:>9.0f}  {r['kv_full']:>13.0f}  {r['kv_mb']:>8.1f}"
              f"  {r['reads_per_token']:>11.0f}  {r['reads_prefill']:>13.3g}")


if __name__ == "__main__":
    main()
