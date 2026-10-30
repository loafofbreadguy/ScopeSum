"""Long-range identifier loss.

For held-out files of the evaluation repos (fastapi, requests, rich, httpx),
find identifier tokens whose name was defined (``def``, ``class``,
assignment target, ``import``) more than ``D`` raw tokens earlier in the same
window **and** in a different scope than the use. Report mean loss on those
tokens and on all tokens, per variant.

The target set depends only on the raw tokens and the scope tracker, so it is
identical for every variant (distances are measured in raw tokens, ignoring
inserted ``<sum>`` tokens).

The pure-Python helpers (``identifier_events``, ``long_range_targets``) are
unit-tested; ``evaluate_long_range`` needs a model.
"""

from __future__ import annotations

import ast
import bisect
import glob
import hashlib
import json
import os
import re
import warnings

from .scope_tracker import UNKNOWN_SCOPE, physical_lines, track_scopes

# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def line_char_starts(text: str) -> list[int]:
    """Char offset of each line start, numbered like ``ast`` (line 1 = index 0)."""
    starts = [s for s, _ in physical_lines(text)]
    return starts or [0]


def ast_pos_to_char(text: str, starts: list[int], lineno: int, col_byte: int) -> int:
    """``ast`` gives UTF-8 *byte* columns; convert to a char offset."""
    if lineno - 1 >= len(starts):
        return len(text)
    s = starts[lineno - 1]
    e = starts[lineno] if lineno < len(starts) else len(text)
    prefix = text[s:e].encode("utf-8")[: max(col_byte, 0)].decode("utf-8", errors="ignore")
    return s + len(prefix)


def parse_quiet(text: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def definition_name_char(text: str, keyword_char: int, name: str) -> int:
    """Char offset of ``name`` in ``def name`` / ``class name`` starting at ``keyword_char``."""
    m = re.compile(r"(?:async\s+)?(?:def|class)\s+" + re.escape(name) + r"\b").search(text, keyword_char)
    return m.end() - len(name) if m else -1


def identifier_events(text: str) -> tuple[list[tuple[str, int]], list[tuple[str, int, int]]]:
    """(definitions [(name, char)], uses [(name, start_char, end_char)]) of a parseable file."""
    tree = parse_quiet(text)
    if tree is None:
        return [], []
    starts = line_char_starts(text)
    defs, uses = [], []

    def pos(lineno, col):
        return ast_pos_to_char(text, starts, lineno, col)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            k = definition_name_char(text, pos(node.lineno, node.col_offset), node.name)
            if k >= 0:
                defs.append((node.name, k))
        elif isinstance(node, ast.Name):
            c = pos(node.lineno, node.col_offset)
            if isinstance(node.ctx, ast.Store):
                defs.append((node.id, c))
            elif isinstance(node.ctx, ast.Load):
                uses.append((node.id, c, c + len(node.id)))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if node.end_lineno is not None and node.end_col_offset is not None:
                e = pos(node.end_lineno, node.end_col_offset)
                s = e - len(node.attr)
                if text[s:e] == node.attr:
                    uses.append((node.attr, s, e))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name.split(".")[0]
                c = pos(alias.lineno, alias.col_offset) if hasattr(alias, "lineno") else pos(node.lineno, node.col_offset)
                k = text.find(bound, c)
                defs.append((bound, k if k >= 0 else c))
    defs.sort(key=lambda d: d[1])
    uses.sort(key=lambda u: u[1])
    return defs, uses


def _token_at(starts: list[int], ends: list[int], char: int) -> int | None:
    i = bisect.bisect_right(starts, char) - 1
    if i >= 0 and starts[i] <= char < max(ends[i], starts[i] + 1):
        return i
    return None


def long_range_targets(window: dict, defs, uses, min_distance: int) -> list[int]:
    """Raw token indices (within the window) of long-range identifier uses.

    ``defs``/``uses`` are in file char offsets; ``window['start_char']`` maps
    them into the window. A use qualifies when the most recent definition of
    the same name before it lies in the window, more than ``min_distance``
    raw tokens earlier, in a different (known) scope.
    """
    offs = [tuple(o) for o in window["offsets"]]
    if not offs:
        return []
    base = window.get("start_char", 0)
    w_end = base + len(window["text"])
    starts = [s for s, _ in offs]
    ends = [e for _, e in offs]
    info = track_scopes(window["text"], offs)
    scope = info.token_scope

    by_name: dict[str, list[int]] = {}
    for name, c in defs:
        if base <= c < w_end:
            by_name.setdefault(name, []).append(c - base)

    targets = set()
    for name, s, e in uses:
        if not (base <= s < w_end) or name not in by_name:
            continue
        s_rel, e_rel = s - base, e - base
        earlier = [c for c in by_name[name] if c < s_rel]
        if not earlier:
            continue
        d_tok = _token_at(starts, ends, earlier[-1])
        u_first = _token_at(starts, ends, s_rel)
        if d_tok is None or u_first is None:
            continue
        if u_first - d_tok <= min_distance:
            continue
        if scope[d_tok] == UNKNOWN_SCOPE or scope[u_first] == UNKNOWN_SCOPE or scope[d_tok] == scope[u_first]:
            continue
        t = u_first
        while t < len(offs) and starts[t] < e_rel:
            targets.add(t)
            t += 1
    return sorted(targets)


# ---------------------------------------------------------------------------
# file discovery
# ---------------------------------------------------------------------------


def clone_eval_repos(urls, dest: str) -> str:
    """Shallow-clone the evaluation repos into ``dest/<name>`` (skips existing ones)."""
    import subprocess

    os.makedirs(dest, exist_ok=True)
    for url in urls:
        name = url.rstrip("/").split("/")[-1].removesuffix(".git").lower()
        target = os.path.join(dest, name)
        if os.path.isdir(os.path.join(target, ".git")):
            continue
        r = subprocess.run(["git", "clone", "--depth", "1", url, target], capture_output=True, text=True)
        head = subprocess.run(["git", "-C", target, "log", "-1", "--format=%h %cs"], capture_output=True, text=True).stdout.strip()
        print(f"[eval] {url} -> {target} {'OK ' + head if r.returncode == 0 else 'FAILED: ' + r.stderr[-300:]}")
    return dest


def find_eval_repo_files(root: str, repos=("fastapi", "requests", "rich", "httpx"), max_files: int | None = None) -> list[str]:
    """``.py`` files of the evaluation repos, balanced across repos, largest first.

    Each repo contributes up to ``ceil(max_files / len(repos))`` files, taken in
    order of decreasing size: long-range targets need files longer than the
    sliding window, and sorting by path would fill the budget with one repo's
    many tiny docs examples. Duplicate contents are removed.
    """
    import math

    per_repo: dict[str, list[tuple[int, str]]] = {r: [] for r in repos}
    for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        parts = p.replace("\\", "/").lower().split("/")
        repo = next((r for r in repos if r in parts[:-1]), None)
        if repo is None:
            continue
        try:
            per_repo[repo].append((os.path.getsize(p), p))
        except OSError:
            continue
    quota = math.ceil(max_files / len(repos)) if max_files else None
    seen, out = set(), []
    for r in repos:
        taken = 0
        for _, p in sorted(per_repo[r], key=lambda x: (-x[0], x[1])):
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    h = hashlib.md5(f.read().encode("utf-8", "replace")).hexdigest()
            except OSError:
                continue
            if h in seen:
                continue
            seen.add(h)
            out.append(p)
            taken += 1
            if quota and taken >= quota:
                break
    return out[:max_files] if max_files else out


# ---------------------------------------------------------------------------
# model evaluation
# ---------------------------------------------------------------------------


def prepare_long_range(cfg, tokenizer, files: list[str], max_windows_per_file: int = 4) -> list[dict]:
    """Variant-independent windows + target raw indices for a list of files."""
    from .data import tokenize_windows

    d = cfg.data
    max_raw = d.seq_len - 1 - d.sum_reserve
    items = []
    for p in files:
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read()
        defs, uses = identifier_events(text)
        if not uses:
            continue
        for w in tokenize_windows(text, tokenizer, max_raw, min_raw=64)[:max_windows_per_file]:
            targets = long_range_targets(w, defs, uses, cfg.eval.long_range_distance)
            items.append({"file": p, "window": w, "targets": targets})
    return items


def evaluate_long_range(lm, items: list[dict], tokenizer, chunk_size: int | None) -> dict:
    """Mean loss on long-range targets and on all raw tokens for ``lm.variant``."""
    import torch

    from .data import build_example

    mode = lm.mode
    seq_len = lm.cfg.data.seq_len
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    tot_all = tot_lr = 0.0
    n_all = n_lr = 0
    lm.eval()
    for it in items:
        ex = build_example(
            it["window"], mode, tokenizer.bos_token_id, lm.sum_id, pad, seq_len, chunk_size=chunk_size, scope_span=lm.scope_span
        )
        pos_of_raw = {r: p for p, r in enumerate(ex.raw_index) if r >= 0}
        batch = lm.make_batch([ex])
        with torch.no_grad(), torch.autocast("cuda", dtype=lm.dtype, enabled=torch.cuda.is_available()):
            out = lm(batch)
        per_tok = out["per_token_loss"][0].float().cpu()  # index p-1 predicts position p
        lab = batch["labels"][0]
        valid = (lab[1:] != -100).float()
        tot_all += float((per_tok * valid).sum())
        n_all += int(valid.sum())
        for r in it["targets"]:
            p = pos_of_raw.get(r)
            if p is not None and p >= 1:
                tot_lr += float(per_tok[p - 1])
                n_lr += 1
    return {
        "variant": lm.variant,
        "loss_all": tot_all / max(n_all, 1),
        "loss_long_range": tot_lr / max(n_lr, 1),
        "n_tokens_all": n_all,
        "n_tokens_long_range": n_lr,
        "n_windows": len(items),
    }


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
