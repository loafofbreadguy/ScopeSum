"""Do the trained models actually use their ``<sum>`` tokens?

Three diagnostics per variant, on the long-range evaluation windows:

1. **Sum ablation.** Loss with the normal masks vs. with every earlier
   ``<sum>`` hidden from raw-token queries, (a) in the sliding layers only and
   (b) in all layers. If hiding the sums barely changes the loss, the model
   routes around them (e.g. through the global layers, which see every raw
   token).
2. **Attention share.** For raw queries that have out-of-window context
   (raw index >= W), the attention mass on ``<sum>`` keys, on BOS, and on raw
   keys at least W raw tokens back ("far raw", only reachable through the
   global layers in B/C/F), averaged over heads, per layer type. Needs eager
   attention (the weights are taken from each layer's ``self_attn`` output).
3. **Loss by position in scope.** Mean loss bucketed by the token's distance
   (in raw tokens) from the start of its own scope, using the plain scope
   segmentation for every variant so that all variants are scored on the same
   tokens. Compression methods tend to lose most right after a boundary.

The pure helpers (``hide_sums``, ``scope_offsets``, ``bucket_of``) are
unit-tested; ``diagnose_variant`` needs a model.
"""

from __future__ import annotations

import numpy as np

from .sum_insertion import KIND_BOS, KIND_RAW, KIND_SUM

OFFSET_BUCKETS = (0, 16, 64, 256, 10**9)


def bucket_of(offset: int, edges=OFFSET_BUCKETS) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= offset < hi:
            return f"{lo}-{hi - 1}" if hi < 10**9 else f"{lo}+"
    return "unknown"


def hide_sums(mask: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Copy of ``mask[b, q, k]`` where raw queries cannot see any ``<sum>`` key."""
    out = mask.copy()
    rows = kind == KIND_RAW  # [B, L]
    cols = kind == KIND_SUM
    out &= ~(rows[:, :, None] & cols[:, None, :])
    return out


def scope_offsets(window: dict) -> np.ndarray:
    """Per raw token: raw tokens since the first token of its own scope (-1 = unknown)."""
    from .scope_tracker import track_scopes
    from .segmenter import scope_segments

    offs = [tuple(o) for o in window["offsets"]]
    seg = scope_segments(track_scopes(window["text"], offs))
    out = np.full(len(offs), -1, dtype=np.int64)
    for t, s in enumerate(seg.token_segment):
        if s >= 0 and seg.segments[s].first_token >= 0:
            out[t] = t - seg.segments[s].first_token
    return out


class _Acc:
    """Running sums of loss per named group."""

    def __init__(self):
        self.tot, self.n = {}, {}

    def add(self, key: str, loss: float, n: int = 1) -> None:
        self.tot[key] = self.tot.get(key, 0.0) + loss
        self.n[key] = self.n.get(key, 0) + n

    def result(self) -> dict:
        return {k: {"loss": self.tot[k] / max(self.n[k], 1), "n": self.n[k]} for k in sorted(self.tot)}


def _attention_hooks(lm, store: dict):
    """Register hooks that reduce each layer's attention weights on the fly."""
    layer_types = list(lm.text_model.config.layer_types)
    handles = []

    def make(li):
        def hook(_m, _i, out):
            ctx = store.get("ctx")
            if ctx is None or len(ctx["queries"]) == 0:
                return  # not an attention pass
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if w is None:  # sdpa/flash return no weights: load the model with eager attention
                store["missing"] = store.get("missing", 0) + 1
                return
            wq = w[0].float().mean(0)[ctx["queries"]]  # [Q, L], mean over heads
            rec = store.setdefault(li, {"sum": 0.0, "bos": 0.0, "far_raw": 0.0, "n": 0})
            rec["sum"] += float(wq[:, ctx["sum_cols"]].sum())
            rec["bos"] += float(wq[:, ctx["bos_cols"]].sum())
            rec["far_raw"] += float((wq * ctx["far"]).sum())
            rec["n"] += len(ctx["queries"])

        return hook

    for li, layer in enumerate(lm.text_model.layers):
        handles.append(layer.self_attn.register_forward_hook(make(li)))
    return handles, layer_types


def _attention_context(ex, window: int, device):
    import torch

    kind = np.asarray(ex.kind)
    raw_index = np.asarray(ex.raw_index)
    queries = np.where((kind == KIND_RAW) & (raw_index >= window))[0]
    is_raw = kind == KIND_RAW
    far = np.zeros((len(queries), len(kind)), dtype=np.float32)
    for i, q in enumerate(queries):
        far[i] = is_raw & (raw_index <= raw_index[q] - window) & (np.arange(len(kind)) < q)
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return {
        "queries": t(queries),
        "sum_cols": t(np.where(kind == KIND_SUM)[0]),
        "bos_cols": t(np.where(kind == KIND_BOS)[0]),
        "far": t(far),
    }


def diagnose_variant(lm, items: list[dict], tokenizer, chunk_size: int | None, n_attention: int = 16) -> dict:
    """Run all three diagnostics for ``lm.variant`` (see the module docstring)."""
    import torch

    from .data import build_example

    seq_len = lm.cfg.data.seq_len
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    conditions = ["normal"] if lm.variant in ("A", "S") else ["normal", "hide_sliding", "hide_all"]
    loss_acc = {c: _Acc() for c in conditions}
    offset_acc = {c: _Acc() for c in conditions}
    attn_store: dict = {}
    hooks, layer_types = _attention_hooks(lm, attn_store)
    lm.eval()
    try:
        for i, it in enumerate(items):
            ex = build_example(
                it["window"], lm.mode, tokenizer.bos_token_id, lm.sum_id, pad, seq_len, chunk_size=chunk_size, scope_span=lm.scope_span
            )
            offsets = scope_offsets(it["window"])
            targets = set(it["targets"])
            base = lm.make_batch([ex])
            for c in conditions:
                batch = dict(base)
                kind = base["kind"].numpy()
                if c in ("hide_sliding", "hide_all"):
                    batch["sliding_mask"] = torch.from_numpy(hide_sums(base["sliding_mask"].numpy(), kind))
                if c == "hide_all":
                    batch["full_mask"] = torch.from_numpy(hide_sums(base["full_mask"].numpy(), kind))
                if c == "normal" and i < n_attention:
                    attn_store["ctx"] = _attention_context(ex, lm.window, lm.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=lm.dtype, enabled=torch.cuda.is_available()):
                    out = lm(batch)
                attn_store.pop("ctx", None)
                per_tok = out["per_token_loss"][0].float().cpu().numpy()  # index p-1 predicts position p
                labels = base["labels"][0].numpy()
                for p in range(1, len(labels)):
                    if labels[p] == -100:
                        continue
                    r = ex.raw_index[p]
                    loss = float(per_tok[p - 1])
                    loss_acc[c].add("all", loss)
                    if r in targets:
                        loss_acc[c].add("long_range", loss)
                    if 0 <= r < len(offsets) and offsets[r] >= 0:
                        offset_acc[c].add(bucket_of(int(offsets[r])), loss)
    finally:
        for h in hooks:
            h.remove()

    attention = {}
    for lt in sorted(set(layer_types)):
        recs = [attn_store[li] for li, t in enumerate(layer_types) if t == lt and li in attn_store and attn_store[li]["n"]]
        n = sum(r["n"] for r in recs)
        if n:
            attention[lt] = {k: sum(r[k] for r in recs) / n for k in ("sum", "bos", "far_raw")}
    per_layer = [
        {"layer": li, "type": layer_types[li], **{k: attn_store[li][k] / attn_store[li]["n"] for k in ("sum", "bos", "far_raw")}}
        for li in range(len(layer_types))
        if li in attn_store and attn_store[li]["n"]
    ]
    res = {
        "variant": lm.variant,
        "n_windows": len(items),
        "loss": {c: loss_acc[c].result() for c in conditions},
        "by_scope_offset": {c: offset_acc[c].result() for c in conditions},
        "attention": attention,
        "attention_per_layer": per_layer,
        "attention_missing_layers": attn_store.get("missing", 0),
    }
    if "hide_sliding" in res["loss"]:
        n = res["loss"]["normal"]
        res["delta"] = {
            c: {k: res["loss"][c][k]["loss"] - n[k]["loss"] for k in n if k in res["loss"][c]} for c in conditions if c != "normal"
        }
    return res

