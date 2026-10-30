"""Linear probes on ``<sum>`` hidden states (variants B and C, also D/E).

For every function in the probe files we take one ``<sum>`` hidden state:

* scope variants (C/D/E/F): the ``<sum>`` of that function's own segment;
* fixed chunks (B): the ``<sum>`` of the chunk that contains the function's
  last raw token (the chunk summary that "ends" the function).

Controls (the per-function tasks favour scope summaries by construction):

* ``subset``: accuracy on all functions, on functions that lie inside one
  B chunk (``one_chunk``) and on functions that straddle a B chunk boundary
  (``multi_chunk``), using B's ``K`` for every variant;
* ``feature="raw_mean"``: the same probe on the mean hidden state of the
  function's raw tokens (an upper bound on what the layer knows).

Labels (from ``ast``):

* ``returns_value``: the body has ``return <expr>`` (not ``None``) or ``yield``;
* ``n_args_bucket``: number of parameters (excluding a leading self/cls),
  bucketed 0 / 1 / 2 / 3+;
* ``name_reused``: the function's name appears again after the function ends.

Probes are logistic regressions (scikit-learn), split 80/20 by file. We report
accuracy and the majority-class baseline.
"""

from __future__ import annotations

import ast
import hashlib
import re

import numpy as np

from .eval_localize import function_table
from .eval_longrange import line_char_starts

TASKS = ("returns_value", "n_args_bucket", "name_reused")


# ---------------------------------------------------------------------------
# labels (pure, unit-tested)
# ---------------------------------------------------------------------------


def _own_nodes(fn):
    """Nodes of a function body, not descending into nested defs/classes/lambdas."""
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    stack = [n for n in fn.body if not isinstance(n, nested)]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(c for c in ast.iter_child_nodes(n) if not isinstance(c, nested))


def returns_value(fn) -> bool:
    for n in _own_nodes(fn):
        if isinstance(n, ast.Return) and n.value is not None:
            if not (isinstance(n.value, ast.Constant) and n.value.value is None):
                return True
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def n_args(fn) -> int:
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args]
    k = len(names) + len(a.kwonlyargs) + (a.vararg is not None) + (a.kwarg is not None)
    if names and names[0] in ("self", "cls"):
        k -= 1
    return k


def bucket(k: int) -> int:
    return min(k, 3)


def function_labels(text: str) -> dict[str, dict]:
    """qualname -> labels; functions whose qualname is duplicated are dropped."""
    funcs = function_table(text)
    starts = line_char_starts(text)
    counts: dict[str, int] = {}
    for f in funcs:
        counts[f["qualname"]] = counts.get(f["qualname"], 0) + 1
    out = {}
    for f in funcs:
        if counts[f["qualname"]] > 1:
            continue
        fn = f["node"]
        end_char = starts[f["end"]] if f["end"] < len(starts) else len(text)
        later = text[end_char:]
        out[f["qualname"]] = {
            "returns_value": int(returns_value(fn)),
            "n_args_bucket": bucket(n_args(fn)),
            "name_reused": int(re.search(rf"\b{re.escape(fn.name)}\b", later) is not None),
        }
    return out


def file_split(path: str, test_fraction: float = 0.2) -> str:
    return "test" if int(hashlib.md5(path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < test_fraction else "train"


# ---------------------------------------------------------------------------
# feature collection (needs a model)
# ---------------------------------------------------------------------------


def sum_positions_for_functions(window: dict, ex_variant, mode: str, chunk_size: int | None) -> dict[str, int]:
    """qualname (without '<module>.') -> position of the <sum> used as its feature."""
    from .scope_tracker import track_scopes
    from .segmenter import scope_segments

    offs = [tuple(o) for o in window["offsets"]]
    seg = scope_segments(track_scopes(window["text"], offs))
    out = {}
    for s in seg.segments:
        if s.kind != "def" or not s.qualname:
            continue
        q = s.qualname.replace("<module>.", "")
        if mode in ("scope", "scope_capped"):  # capped mode keeps the scope's seg_id
            pos = ex_variant.sum_position[s.seg_id] if s.seg_id < len(ex_variant.sum_position) else -1
        else:  # fixed: chunk holding the function's last raw token
            last_raw = s.insert_before - 1
            if last_raw < 0 or not chunk_size:
                continue
            c = last_raw // chunk_size
            pos = ex_variant.sum_position[c] if c < len(ex_variant.sum_position) else -1
        if pos >= 0:
            out[q] = pos
    return out


def function_spans(window: dict) -> dict[str, tuple[int, int]]:
    """qualname (without '<module>.') -> (first, last) raw token index of the def."""
    from .scope_tracker import track_scopes
    from .segmenter import scope_segments

    offs = [tuple(o) for o in window["offsets"]]
    seg = scope_segments(track_scopes(window["text"], offs))
    return {
        s.qualname.replace("<module>.", ""): (s.first_token, s.insert_before - 1)
        for s in seg.segments
        if s.kind == "def" and s.qualname and s.first_token >= 0
    }


def collect_features(
    lm, tokenizer, files: list[str], chunk_size: int | None, layers=(-1,), max_files=None, split_chunk: int | None = None
) -> dict:
    """Hidden states at <sum> positions + labels, for ``lm.variant``.

    ``split_chunk``: B's ``K``, used to mark functions that straddle a B chunk
    boundary (defaults to ``chunk_size``).
    """
    import torch

    from .data import build_example, tokenize_windows

    mode = lm.mode
    split_chunk = split_chunk or chunk_size
    if lm.variant in ("A", "S"):
        raise ValueError("variant A has no <sum> tokens to probe")
    seq_len = lm.cfg.data.seq_len
    max_raw = seq_len - 1 - lm.cfg.data.sum_reserve
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    feats = {li: [] for li in layers}
    raw_feats = {li: [] for li in layers}
    labels = {t: [] for t in TASKS}
    split, multi_chunk = [], []
    for p in files[: max_files or None]:
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read()
        labs = function_labels(text)
        if not labs:
            continue
        for w in tokenize_windows(text, tokenizer, max_raw, min_raw=64)[:4]:  # cap per file (some files are huge)
            ex = build_example(
                w, mode, tokenizer.bos_token_id, lm.sum_id, pad, seq_len, chunk_size=chunk_size, scope_span=lm.scope_span
            )
            pos = sum_positions_for_functions(w, ex, mode, chunk_size)
            spans = function_spans(w)
            pos_of_raw = np.full(len(w["raw_ids"]), -1)
            for p_, r in enumerate(ex.raw_index):
                if 0 <= r < len(pos_of_raw):
                    pos_of_raw[r] = p_
            pos = {q: p_ for q, p_ in pos.items() if q in labs}
            if not pos:
                continue
            batch = lm.make_batch([ex])
            with torch.no_grad(), torch.autocast("cuda", dtype=lm.dtype, enabled=torch.cuda.is_available()):
                out = lm(batch, output_hidden_states=True, compute_loss=False)
            hs = out["hidden_states"]
            for q, p_ in pos.items():
                first, last = spans.get(q, (-1, -1))
                raw_pos = pos_of_raw[max(first, 0) : last + 1]
                raw_pos = torch.as_tensor(raw_pos[raw_pos >= 0])
                if first < 0 or len(raw_pos) == 0:
                    continue
                for li in layers:
                    feats[li].append(hs[li][0, p_].float().cpu().numpy())
                    raw_feats[li].append(hs[li][0, raw_pos.to(hs[li].device)].float().mean(0).cpu().numpy())
                for t in TASKS:
                    labels[t].append(labs[q][t])
                split.append(file_split(p))
                multi_chunk.append(bool(split_chunk) and first // split_chunk != last // split_chunk)
    stack = lambda d: {li: np.stack(v) if v else np.zeros((0, 1)) for li, v in d.items()}  # noqa: E731
    return {
        "variant": lm.variant,
        "features": stack(feats),
        "raw_features": stack(raw_feats),
        "labels": {t: np.asarray(v) for t, v in labels.items()},
        "split": np.asarray(split),
        "multi_chunk": np.asarray(multi_chunk, dtype=bool),
    }


def train_probes(collected: dict, seed: int = 0) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    tr = collected["split"] == "train"
    te = collected["split"] == "test"
    n = len(collected["split"])
    multi = collected.get("multi_chunk", np.zeros(n, dtype=bool))
    subsets = {"all": np.ones(n, dtype=bool), "one_chunk": ~multi, "multi_chunk": multi}
    feature_sets = {"sum": collected["features"], "raw_mean": collected.get("raw_features", {})}
    for feature, by_layer in feature_sets.items():
        for li, X in by_layer.items():
            for t in TASKS:
                y = collected["labels"][t]
                base = {"variant": collected["variant"], "layer": li, "task": t, "feature": feature}
                if len(y) == 0 or tr.sum() < 10 or te.sum() < 5 or len(set(y[tr])) < 2:
                    rows.append({**base, "subset": "all", "accuracy": None, "majority": None, "n_train": int(tr.sum()), "n_test": int(te.sum())})
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, random_state=seed))
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X)
                vals, cnts = np.unique(y[tr], return_counts=True)
                for subset, sel in subsets.items():
                    m = te & sel
                    if m.sum() < 5:
                        continue
                    rows.append({**base, "subset": subset, "accuracy": float((pred[m] == y[m]).mean()),
                                 "majority": float((y[m] == vals[cnts.argmax()]).mean()),
                                 "n_train": int(tr.sum()), "n_test": int(m.sum())})
    return rows
