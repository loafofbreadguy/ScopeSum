"""Data pipeline: Python files -> windows -> per-variant examples on disk.

Steps (``prepare_dataset`` runs all of them):

1. Read Python files from a Hugging Face dataset (default
   ``bigcode/the-stack-smol-xl``, ``data/python``) or a local glob.
2. Drop files from the evaluation repos (fastapi, requests, rich, httpx) by
   repo name, path components and header/import signatures; drop files that
   do not ``ast.parse``; drop files shorter than ``min_tokens`` tokens.
3. Tokenize once (no special tokens, with offsets) and cut each file into
   *windows* of at most ``seq_len - 1 - sum_reserve`` raw tokens. Cuts are
   placed at top-level scope boundaries when possible, so windows usually
   start at column 0.
4. Build examples for each segmentation mode from the same windows (same
   raw tokens, same order for every variant):
   ``none`` (A), ``fixed`` (B, K from the fairness rule), ``scope`` (C/D/E,
   with call edges stored for E). ``scope_capped`` (F) is not built here:
   :func:`load_or_build_examples` builds it from the saved windows on first use.
5. Save windows, examples and ``stats.json`` (average scope length, <sum>
   counts, K) under ``data.out_dir``.

Heavy libraries (``datasets``, tokenizers) are imported inside functions so
the pure helpers stay importable for unit tests.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass

from .config import Config
from .graph_edges import regex_call_edges
from .scope_tracker import track_scopes
from .segmenter import Segmentation, capped_scope_segments, fairness_stats, fixed_segments, scope_segments
from .sum_insertion import Example, insert_sums

MODES = ("none", "fixed", "scope")  # built by prepare_dataset; "scope_capped" is built on demand

DEFAULT_SIGNATURES = (
    # Header/author strings of the evaluation repos' own source files.
    "Kenneth Reitz",  # requests
    "Will McGugan",  # rich
    "Encode OSS",  # httpx
    "Sebastián Ramírez",  # fastapi
    "tiangolo",  # fastapi
)


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------


def _name_parts(s: str) -> list[str]:
    return [p for p in re.split(r"[^a-z0-9]+", s.lower()) if p]


def exclusion_reason(
    repo: str | None,
    path: str | None,
    text: str,
    excluded_repos,
    signatures=DEFAULT_SIGNATURES,
    excluded_exact=(),
) -> str | None:
    """Why this file must be excluded (``None`` = keep). Deliberately conservative.

    ``excluded_repos`` match repo-name parts, path directories and signatures;
    ``excluded_exact`` match the repo name exactly (e.g. "django" but not "django-foo").
    """
    excluded = {r.lower() for r in excluded_repos}
    if repo:
        base = repo.split("/")[-1].lower()
        if base in excluded or any(p in excluded for p in _name_parts(base)):
            return "repo"
        if base in {r.lower() for r in excluded_exact}:
            return "repo"
    if path:
        dirs = path.lower().replace("\\", "/").split("/")[:-1]
        if any(d in excluded for d in dirs):
            return "path"
    head = text[:4000]
    for sig in signatures:
        if sig in head:
            return f"signature:{sig}"
    for name in excluded:
        # absolute imports of the package's own submodules ("from httpx._models import")
        if re.search(rf"^\s*(from|import)\s+{re.escape(name)}\._", text, flags=re.M):
            return f"import:{name}"
    return None


def parses(text: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(text)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def file_id(repo: str | None, path: str | None, text: str) -> str:
    key = f"{repo}::{path}" if (repo or path) else text[:2000]
    return hashlib.md5(key.encode("utf-8", "replace")).hexdigest()


def is_heldout(fid: str, fraction: float) -> bool:
    return int(fid[:8], 16) / 0xFFFFFFFF < fraction


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


def top_level_cut_points(text: str, offsets) -> list[int]:
    """Raw token indices where a new top-level unit starts (safe window starts)."""
    info = track_scopes(text, offsets)
    cuts = {sc.close_token for sc in info.scopes if sc.parent < 0}
    return sorted(c for c in cuts if 0 < c < len(offsets))


def make_windows(n_tokens: int, cut_points: list[int], max_raw: int, min_raw: int) -> list[tuple[int, int]]:
    """Greedy windows [a, b): prefer the furthest cut point within ``max_raw``."""
    windows = []
    a = 0
    cuts = sorted(cut_points)
    while a < n_tokens:
        limit = a + max_raw
        if limit >= n_tokens:
            b = n_tokens
        else:
            best = None
            for c in cuts:  # cut lists are short (one entry per top-level unit)
                if a < c <= limit:
                    best = c
                elif c > limit:
                    break
            b = best if best is not None else limit
        if b - a >= min_raw:
            windows.append((a, b))
        a = b
    return windows


def window_view(text: str, offsets, a: int, b: int) -> tuple[str, list[tuple[int, int]]]:
    """Text slice and shifted offsets of tokens [a, b)."""
    start = offsets[a][0]
    end = max(e for _, e in offsets[a:b])
    return text[start:end], [(max(s - start, 0), max(e - start, 0)) for s, e in offsets[a:b]]


# ---------------------------------------------------------------------------
# per-variant examples
# ---------------------------------------------------------------------------


def segment_window(
    text: str, offsets, mode: str, chunk_size: int | None = None, scope_span: int | None = None
) -> Segmentation | None:
    if mode == "none":
        return None
    if mode == "fixed":
        if not chunk_size:
            raise ValueError("fixed mode needs chunk_size (variant B's K)")
        return fixed_segments(len(offsets), chunk_size)
    if mode == "scope":
        return scope_segments(track_scopes(text, offsets))
    if mode == "scope_capped":
        if not scope_span:
            raise ValueError("scope_capped mode needs scope_span (variant F's max part span)")
        return capped_scope_segments(track_scopes(text, offsets), scope_span)
    raise ValueError(f"unknown mode {mode!r}")


def build_example(
    window: dict,
    mode: str,
    bos_id: int,
    sum_id: int,
    pad_id: int,
    seq_len: int,
    chunk_size: int | None = None,
    with_edges: bool = True,
    scope_span: int | None = None,
) -> Example:
    offsets = [tuple(o) for o in window["offsets"]]
    seg = segment_window(window["text"], offsets, mode, chunk_size, scope_span)
    edges = regex_call_edges(window["text"], offsets, seg) if (seg is not None and mode == "scope" and with_edges) else None
    return insert_sums(
        list(window["raw_ids"]),
        seg,
        bos_id=bos_id,
        sum_id=sum_id,
        pad_id=pad_id,
        max_len=seq_len,
        pad_to=seq_len,
        seg_edges=edges,
    )


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


@dataclass
class SourceFile:
    repo: str | None
    path: str | None
    text: str


def iter_source_files(cfg: Config):
    d = cfg.data
    if d.local_files_glob:
        for p in sorted(glob.glob(d.local_files_glob, recursive=True)):
            with open(p, encoding="utf-8", errors="replace") as f:
                yield SourceFile(repo=None, path=p, text=f.read())
        return
    from datasets import load_dataset

    kwargs = {"split": d.split, "streaming": d.streaming}
    if d.data_dir:
        kwargs["data_dir"] = d.data_dir
    ds = load_dataset(d.dataset_id, d.dataset_config, **kwargs) if d.dataset_config else load_dataset(d.dataset_id, **kwargs)
    ds = ds.shuffle(seed=d.seed) if not d.streaming else ds.shuffle(seed=d.seed, buffer_size=10_000)
    for row in ds:
        yield SourceFile(repo=row.get(d.repo_field), path=row.get(d.path_field), text=row.get(d.text_field) or "")


def discover_dataset(cfg: Config, n_rows: int = 1, max_chars: int = 300) -> None:
    """Print the dataset's columns and one example (run this first on Kaggle)."""
    from datasets import load_dataset

    d = cfg.data
    kwargs = {"split": d.split, "streaming": True}
    if d.data_dir:
        kwargs["data_dir"] = d.data_dir
    ds = load_dataset(d.dataset_id, **kwargs)
    print(f"[discover] dataset={d.dataset_id} data_dir={d.data_dir} split={d.split}")
    print(f"[discover] features: {getattr(ds, 'features', None)}")
    for i, row in enumerate(ds):
        print(f"[discover] row {i} keys: {sorted(row)}")
        for k, v in row.items():
            print(f"    {k}: {repr(v)[:max_chars]}")
        if i + 1 >= n_rows:
            break
    for name in ("text_field", "repo_field", "path_field"):
        print(f"[discover] config data.{name} = {getattr(d, name)!r}")


def check_offsets(tokenizer, text: str) -> dict:
    """Sanity-check a tokenizer's offset mapping on ``text``."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    n = len(offs)
    empty = sum(1 for s, e in offs if e <= s)
    backwards = sum(1 for i in range(1, n) if offs[i][0] < offs[i - 1][0])
    covered = sum(max(0, e - s) for s, e in offs)
    return {
        "n_tokens": n,
        "empty_spans": empty,
        "non_monotonic": backwards,
        "chars": len(text),
        "chars_covered": covered,
    }


# ---------------------------------------------------------------------------
# the full pipeline
# ---------------------------------------------------------------------------


def _features(kind: str):
    """Explicit Arrow types so shards with empty lists concatenate cleanly."""
    from datasets import Features, List, Value

    if kind == "windows":
        return Features(
            {
                "file_id": Value("string"),
                "repo": Value("string"),
                "path": Value("string"),
                "window": Value("int32"),
                "raw_ids": List(Value("int32")),
                "offsets": List(List(Value("int32"))),
                "text": Value("string"),
            }
        )
    ints = List(Value("int32"))
    return Features(
        {
            "input_ids": ints,
            "labels": ints,
            "kind": List(Value("int8")),
            "segment": ints,
            "is_header": List(Value("int8")),
            "raw_index": ints,
            "seg_parent": ints,
            "sum_position": ints,
            "seg_edges": List(List(Value("int32"))),
            "n_truncated": Value("int32"),
        }
    )


def _save_rows(rows: list[dict], path: str, kind: str = "windows") -> None:
    from datasets import Dataset

    Dataset.from_list(rows, features=_features(kind)).save_to_disk(path)


def _save_generated(gen_fn, path: str, kind: str = "examples", shard_size: int = 500) -> None:
    """Build the dataset shard by shard (examples are large; avoid huge Python lists)."""
    from datasets import Dataset, concatenate_datasets

    feats = _features(kind)
    parts, buf = [], []
    for row in gen_fn():
        buf.append(row)
        if len(buf) >= shard_size:
            parts.append(Dataset.from_list(buf, features=feats))
            buf = []
    if buf or not parts:
        parts.append(Dataset.from_list(buf, features=feats))
    concatenate_datasets(parts).save_to_disk(path)


def collect_windows(cfg: Config, tokenizer, log_every: int = 500) -> tuple[dict, dict]:
    """Return ({"train": [...], "heldout": [...]}, counters)."""
    d = cfg.data
    max_raw = d.seq_len - 1 - d.sum_reserve
    budget = {"train": d.token_budget, "heldout": max(int(d.token_budget * d.heldout_fraction), max_raw)}
    used = {"train": 0, "heldout": 0}
    windows: dict[str, list[dict]] = {"train": [], "heldout": []}
    counts = {"seen": 0, "kept": 0, "too_short": 0, "no_parse": 0, "excluded": {}}
    for f in iter_source_files(cfg):
        if all(used[s] >= budget[s] for s in used):
            break
        if d.max_files is not None and counts["seen"] >= d.max_files:
            break
        counts["seen"] += 1
        reason = exclusion_reason(f.repo, f.path, f.text, d.excluded_repos, excluded_exact=d.excluded_repos_exact)
        if reason:
            key = reason.split(":")[0]
            counts["excluded"][key] = counts["excluded"].get(key, 0) + 1
            continue
        if not parses(f.text):
            counts["no_parse"] += 1
            continue
        fid = file_id(f.repo, f.path, f.text)
        split = "heldout" if is_heldout(fid, d.heldout_fraction) else "train"
        if used[split] >= budget[split]:
            continue
        enc = tokenizer(f.text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
        if len(ids) < d.min_tokens:
            counts["too_short"] += 1
            continue
        counts["kept"] += 1
        cuts = top_level_cut_points(f.text, offs)
        for wi, (a, b) in enumerate(make_windows(len(ids), cuts, max_raw, d.min_window_tokens)):
            wtext, woffs = window_view(f.text, offs, a, b)
            windows[split].append(
                {
                    "file_id": fid,
                    "repo": f.repo or "",
                    "path": f.path or "",
                    "window": wi,
                    "raw_ids": ids[a:b],
                    "offsets": [list(o) for o in woffs],
                    "text": wtext,
                }
            )
            used[split] += b - a
            if used[split] >= budget[split]:
                break
        if counts["seen"] % log_every == 0:
            print(f"[data] seen={counts['seen']} kept={counts['kept']} tokens={used}")
    counts["raw_tokens"] = used
    return windows, counts


def prepare_dataset(cfg: Config, tokenizer, sum_id: int, modes=MODES) -> dict:
    """Run the whole pipeline and save everything under ``cfg.data.out_dir``."""
    d = cfg.data
    os.makedirs(d.out_dir, exist_ok=True)
    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    windows, counts = collect_windows(cfg, tokenizer)
    for split, rows in windows.items():
        _save_rows(rows, os.path.join(d.out_dir, f"windows_{split}"))

    # Fairness: K from the scope-mode <sum> counts on the training windows.
    lengths, scope_counts, truncated = [], [], 0
    for w in windows["train"]:
        seg = segment_window(w["text"], [tuple(o) for o in w["offsets"]], "scope")
        lengths.append(len(w["raw_ids"]))
        scope_counts.append(seg.n_sums)
        truncated += len(w["raw_ids"]) + 1 + seg.n_sums > d.seq_len
    stats = fairness_stats(lengths, scope_counts) if lengths else {}
    stats.update(
        {
            "files": counts,
            "seq_len": d.seq_len,
            "max_raw_per_window": d.seq_len - 1 - d.sum_reserve,
            "scope_examples_truncated": truncated,
            "max_scope_sums_per_example": max(scope_counts) if scope_counts else 0,
            "bos_id": bos_id,
            "sum_id": sum_id,
            "pad_id": pad_id,
            "n_windows": {s: len(r) for s, r in windows.items()},
        }
    )
    k = stats.get("chunk_size")

    for mode in modes:
        for split, rows in windows.items():

            def gen(rows=rows, mode=mode):
                for w in rows:
                    yield build_example(w, mode, bos_id, sum_id, pad_id, d.seq_len, chunk_size=k).to_dict()

            _save_generated(gen, os.path.join(d.out_dir, f"examples_{mode}_{split}"))
            print(f"[data] saved examples_{mode}_{split}: {len(rows)} examples")

    with open(os.path.join(d.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_stats(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "stats.json")) as f:
        return json.load(f)


def load_examples(out_dir: str, mode: str, split: str = "train"):
    """HF dataset of ``Example.to_dict()`` rows (use ``Example.from_dict`` per row)."""
    from datasets import load_from_disk

    return load_from_disk(os.path.join(out_dir, f"examples_{mode}_{split}"))


def load_or_build_examples(
    cfg: Config, mode: str, split: str = "train", scope_span: int | None = None, chunk_size: int | None = None
):
    """Examples of ``mode`` from ``data.out_dir``, else built once from its saved windows.

    Modes added after the data was prepared (``scope_capped``, and ``fixed``
    with a ``chunk_size`` other than stats.json's ``K``) are built into
    ``data.extra_dir`` (the prepared data is a read-only Kaggle input) with the
    token ids recorded in ``stats.json``. Same windows, same order as every
    other mode.
    """
    from datasets import load_from_disk

    stats = load_stats(cfg.data.out_dir)
    k = chunk_size or stats.get("chunk_size")
    name = f"examples_{mode}_{split}"
    if mode == "scope_capped":
        name = f"examples_{mode}{scope_span}_{split}"
    elif mode == "fixed" and k != stats.get("chunk_size"):
        name = f"examples_fixed{k}_{split}"
    for root in (cfg.data.out_dir, cfg.data.extra_dir):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return load_from_disk(p)
    rows = load_windows(cfg.data.out_dir, split)

    def gen():
        for w in rows:
            yield build_example(
                w, mode, stats["bos_id"], stats["sum_id"], stats["pad_id"], cfg.data.seq_len, chunk_size=k, scope_span=scope_span
            ).to_dict()

    p = os.path.join(cfg.data.extra_dir, name)
    os.makedirs(cfg.data.extra_dir, exist_ok=True)
    _save_generated(gen, p)
    print(f"[data] built {name}: {len(rows)} examples -> {p}")
    return load_from_disk(p)


def load_windows(out_dir: str, split: str = "train"):
    from datasets import load_from_disk

    return load_from_disk(os.path.join(out_dir, f"windows_{split}"))


# ---------------------------------------------------------------------------
# tokenizer / model location helpers (shared with model.py)
# ---------------------------------------------------------------------------


def resolve_model_path(cfg: Config) -> str:
    """Where to load the model/tokenizer from, per ``model.source``."""
    m = cfg.model
    if m.source == "hf":
        return m.model_id
    if m.source == "local":
        if not m.local_path:
            raise ValueError("model.source=local needs model.local_path")
        return m.local_path
    if m.source == "kagglehub":
        import kagglehub

        path = kagglehub.model_download(m.kagglehub_handle)
        print(f"[model] kagglehub {m.kagglehub_handle} -> {path}")
        return path
    raise ValueError(f"unknown model.source {m.source!r}")


def load_tokenizer(cfg: Config, path: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path or resolve_model_path(cfg), trust_remote_code=cfg.model.trust_remote_code)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("a fast tokenizer is required for offset mappings")
    return tok


def resolve_sum_token_id(tokenizer, cfg: Config) -> int:
    """Token id used as <sum>.

    ``reuse`` (default): an existing reserved token (``<unused0>`` = id 6 in
    Gemma 4). Its embedding is replaced by a trainable vector in model.py, so
    this behaves exactly like a newly added token without resizing the
    embedding tables (Gemma 4 also has a per-layer embedding table that would
    need resizing). ``add``: append a new ``<sum>`` special token.
    """
    m = cfg.model
    if m.sum_token_mode == "reuse":
        tid = tokenizer.convert_tokens_to_ids(m.sum_token)
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(f"{m.sum_token!r} is not in the vocabulary; pick another reserved token")
        return int(tid)
    if m.sum_token_mode == "add":
        tokenizer.add_special_tokens({"additional_special_tokens": [m.sum_token]})
        return int(tokenizer.convert_tokens_to_ids(m.sum_token))
    raise ValueError(f"unknown model.sum_token_mode {m.sum_token_mode!r}")


def tokenize_windows(text: str, tokenizer, max_raw: int, min_raw: int = 1) -> list[dict]:
    """Tokenize one file and cut it into windows exactly like the training data.

    Each window dict has ``raw_ids``, ``offsets`` (window-relative),
    ``text``, ``start_char`` (offset of the window in ``text``) and ``first_token``.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    if not ids:
        return []
    cuts = top_level_cut_points(text, offs)
    out = []
    for a, b in make_windows(len(ids), cuts, max_raw, min_raw):
        wtext, woffs = window_view(text, offs, a, b)
        out.append(
            {
                "raw_ids": ids[a:b],
                "offsets": [list(o) for o in woffs],
                "text": wtext,
                "start_char": offs[a][0],
                "first_token": a,
            }
        )
    return out
