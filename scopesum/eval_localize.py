"""Bug localization: SWE-bench Lite by default (files fetched from GitHub at each
task's ``base_commit``), or a ``tasks.jsonl`` + ``snapshots/`` folder.

For each task:

1. Ground truth = functions touched by the reference patch: parse the
   unified diff, take the changed old-file lines, and map each to the
   innermost function (``ast``) that contains it.
2. Candidates = all functions in the touched file(s).
3. For each touched file, context = the file's first window (cut at a
   top-level boundary, as in training, with ``<sum>`` tokens per variant)
   followed by a fixed prompt with the issue text. Each candidate is scored
   by the log-likelihood of its (qualified) name after that prompt.
4. Rank all candidates; report top-1 / top-5 accuracy (hit if any ground-truth
   function is in the top k).

Run :func:`discover_tasks` first: it prints the task keys so the
``eval.*_field`` config values can be checked.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import random
import re

from .eval_longrange import parse_quiet

# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def _short(v, n=300):
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[:n] + f"... [{len(s)} chars]"


def discover_competition(cfg, n_tasks: int = 1) -> None:
    """Print the competition folder layout, task keys and one example."""
    e = cfg.eval
    root = e.competition_dir
    print(f"[discover] competition_dir = {root}")
    if not os.path.isdir(root):
        found = glob.glob("/kaggle/input/**/" + e.tasks_file, recursive=True)
        print(f"[discover] not found. {e.tasks_file} files under /kaggle/input: {found[:10]}")
        return
    for name in sorted(os.listdir(root))[:30]:
        p = os.path.join(root, name)
        kind = f"dir ({len(os.listdir(p))} entries)" if os.path.isdir(p) else f"file ({os.path.getsize(p)} bytes)"
        print(f"  {name}: {kind}")
    tasks_path = os.path.join(root, e.tasks_file)
    if os.path.exists(tasks_path):
        with open(tasks_path) as f:
            for i, line in enumerate(f):
                if i >= n_tasks:
                    break
                row = json.loads(line)
                print(f"[discover] task {i} keys: {sorted(row)}")
                for k, v in row.items():
                    print(f"    {k}: {_short(v)}")
    else:
        print(f"[discover] {tasks_path} missing")
    for sub in (e.snapshots_dir, e.graphs_dir):
        p = os.path.join(root, sub)
        if os.path.isdir(p):
            entries = sorted(os.listdir(p))
            print(f"[discover] {sub}/: {len(entries)} entries, first: {entries[:5]}")
            if entries and os.path.isdir(os.path.join(p, entries[0])):
                print(f"    {entries[0]}/: {sorted(os.listdir(os.path.join(p, entries[0])))[:10]}")
    print("[discover] configured fields:", {k: getattr(e, k) for k in ("task_id_field", "issue_field", "patch_field", "repo_field", "snapshot_field", "graph_field")})


def load_swebench_tasks(cfg) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(cfg.eval.swebench_dataset, split=cfg.eval.swebench_split)
    return [dict(r) for r in ds]


def discover_tasks(cfg, n_tasks: int = 1) -> None:
    """Print the task source's keys and one example (either source)."""
    if cfg.eval.localize_source == "jsonl":
        discover_competition(cfg, n_tasks)
        return
    tasks = load_swebench_tasks(cfg)
    print(f"[discover] {cfg.eval.swebench_dataset}:{cfg.eval.swebench_split} -> {len(tasks)} tasks")
    repos: dict[str, int] = {}
    for t in tasks:
        repos[str(t.get(cfg.eval.repo_field))] = repos.get(str(t.get(cfg.eval.repo_field)), 0) + 1
    print(f"[discover] repos: {repos}")
    print(f"[discover] keys: {sorted(tasks[0])}")
    for k, v in tasks[0].items():
        print(f"    {k}: {_short(v)}")
    print("[discover] configured fields:", {k: getattr(cfg.eval, k) for k in ("task_id_field", "issue_field", "patch_field", "repo_field", "commit_field")})


def load_tasks(cfg) -> list[dict]:
    if cfg.eval.localize_source == "swebench":
        tasks = load_swebench_tasks(cfg)
    else:
        path = os.path.join(cfg.eval.competition_dir, cfg.eval.tasks_file)
        with open(path) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
    missing = [k for k in (cfg.eval.issue_field, cfg.eval.patch_field) if tasks and k not in tasks[0]]
    if missing:
        raise KeyError(f"task fields {missing} not found; keys are {sorted(tasks[0])}. Fix eval.*_field in the config.")
    if cfg.eval.max_tasks and len(tasks) > cfg.eval.max_tasks:
        # seeded random subset (identical for every variant); the dataset is sorted by repo
        idx = sorted(random.Random(cfg.data.seed).sample(range(len(tasks)), cfg.eval.max_tasks))
        tasks = [tasks[i] for i in idx]
    return tasks


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(patch: str) -> dict[str, set[int]]:
    """Changed *old-file* line numbers per file from a unified diff.

    Removed/modified lines count directly; a pure insertion counts the old
    line right before it (or line 1), so insertions inside a function body
    still point at that function. New files (``--- /dev/null``) are skipped.
    """
    out: dict[str, set[int]] = {}
    cur = None
    old = 0
    for line in patch.splitlines():
        if line.startswith("--- "):
            p = line[4:].split("\t")[0].strip()
            cur = None if p == "/dev/null" else re.sub(r"^[ab]/", "", p)
            continue
        if line.startswith("+++ "):
            continue
        if line.startswith("diff --git"):
            cur = None
            continue
        m = _HUNK_RE.match(line)
        if m:
            old = int(m.group(1))
            continue
        if cur is None or not line or line.startswith("\\"):
            continue
        tag = line[0]
        if tag == " ":
            old += 1
        elif tag == "-":
            out.setdefault(cur, set()).add(old)
            old += 1
        elif tag == "+":
            out.setdefault(cur, set()).add(max(old - 1, 1))
    return out


def function_table(text: str) -> list[dict]:
    """All functions/methods: qualname, name, start/end line (decorators included)."""
    tree = parse_quiet(text)
    if tree is None:
        return []
    rows = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{child.name}"
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                rows.append({"qualname": q, "name": child.name, "start": start, "end": child.end_lineno or child.lineno, "node": child})
                visit(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return rows


def touched_functions(funcs: list[dict], lines: set[int]) -> set[str]:
    """Innermost function containing each changed line."""
    hit = set()
    for ln in lines:
        inner = None
        for f in funcs:
            if f["start"] <= ln <= f["end"] and (inner is None or f["end"] - f["start"] < inner["end"] - inner["start"]):
                inner = f
        if inner is not None:
            hit.add(inner["qualname"])
    return hit


def topk_hit(ranked: list[str], gold: set[str], k: int) -> bool:
    return any(c in gold for c in ranked[:k])


# ---------------------------------------------------------------------------
# snapshot files
# ---------------------------------------------------------------------------


def fetch_github_file(repo: str, commit: str, rel_path: str, cache_dir: str, retries: int = 3) -> str | None:
    """Download ``rel_path`` of ``repo`` (owner/name) at ``commit`` (cached on disk)."""
    import time
    import urllib.error
    import urllib.request

    dest = os.path.join(cache_dir, repo.replace("/", "__"), commit, rel_path)
    if os.path.exists(dest):
        return dest
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{rel_path}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            return dest
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2 * (attempt + 1))
    return None


def resolve_snapshot_file(cfg, task: dict, rel_path: str) -> str | None:
    """Local path of a patched file: fetched from GitHub (swebench) or found in snapshots/ (jsonl)."""
    e = cfg.eval
    if e.localize_source == "swebench":
        repo, commit = task.get(e.repo_field), task.get(e.commit_field)
        if not repo or not commit:
            return None
        return fetch_github_file(str(repo), str(commit), rel_path, e.file_cache_dir)
    root = os.path.join(e.competition_dir, e.snapshots_dir)
    keys = [task.get(e.snapshot_field), task.get(e.task_id_field), task.get(e.repo_field)]
    for k in keys:
        if k:
            p = os.path.join(root, str(k), rel_path)
            if os.path.exists(p):
                return p
    for k in keys:  # repo name may be "owner/name"
        if k and "/" in str(k):
            p = os.path.join(root, str(k).split("/")[-1], rel_path)
            if os.path.exists(p):
                return p
    hits = glob.glob(os.path.join(root, "**", rel_path), recursive=True)
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# model scoring
# ---------------------------------------------------------------------------


def _edges_for(cfg, task, window, seg_text_offsets, seg):
    from .graph_edges import GraphSchema, call_pairs_from_graph, edges_from_name_pairs, regex_call_edges

    if cfg.eval.localize_edges == "competition":
        gname = task.get(cfg.eval.graph_field)
        if gname:
            cands = glob.glob(os.path.join(cfg.eval.competition_dir, cfg.eval.graphs_dir, "**", f"{gname}*"), recursive=True)
            if cands:
                with open(cands[0]) as f:
                    graph = json.load(f)
                pairs = call_pairs_from_graph(graph, GraphSchema(**cfg.eval.graph_schema), file_suffix=window.get("rel_path"))
                return edges_from_name_pairs(pairs, seg)
    text, offs = seg_text_offsets
    return regex_call_edges(text, offs, seg)


def score_file_candidates(lm, tokenizer, cfg, task, rel_path: str, text: str, candidates: list[str], chunk_size) -> dict[str, float]:
    """Log-likelihood of each candidate name after (file context + issue prompt)."""
    import torch

    from .data import segment_window, tokenize_windows
    from .sum_insertion import insert_sums

    e = cfg.eval
    seq_len = cfg.data.seq_len
    issue_ids = tokenizer(str(task.get(e.issue_field, "")), add_special_tokens=False)["input_ids"][: e.max_issue_tokens]
    issue = tokenizer.decode(issue_ids)
    prompt_ids = tokenizer(e.localize_prompt.format(issue=issue), add_special_tokens=False)["input_ids"]
    name_ids = {c: tokenizer(c + "(", add_special_tokens=False)["input_ids"] for c in candidates}
    longest = max((len(v) for v in name_ids.values()), default=1)
    max_raw = max(64, seq_len - 1 - cfg.data.sum_reserve - len(prompt_ids) - longest)
    windows = tokenize_windows(text, tokenizer, max_raw, min_raw=1)
    if not windows:
        return {}
    w = windows[0]
    w["rel_path"] = rel_path
    mode = lm.mode
    offs = [tuple(o) for o in w["offsets"]]
    seg = segment_window(w["text"], offs, mode, chunk_size, lm.scope_span)
    edges = _edges_for(cfg, task, w, (w["text"], offs), seg) if (seg is not None and lm.variant == "E") else None
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    scores = {}
    items = list(name_ids.items())
    bs = max(1, e.localize_batch_size)
    for s in range(0, len(items), bs):
        chunk = items[s : s + bs]
        exs = [
            insert_sums(w["raw_ids"], seg, tokenizer.bos_token_id, lm.sum_id, pad, max_len=seq_len, pad_to=seq_len,
                        suffix_ids=prompt_ids + ids, seg_edges=edges)
            for _, ids in chunk
        ]
        batch = lm.make_batch(exs)
        with torch.no_grad(), torch.autocast("cuda", dtype=lm.dtype, enabled=torch.cuda.is_available()):
            # Only the candidate-name positions need logits: building the full
            # [L, 262144] logit matrix for every candidate ran out of memory on a T4.
            hidden = lm(batch, compute_loss=False)["hidden"]
            n_before = len(w["raw_ids"]) + len(prompt_ids)
            for b, (cand, ids) in enumerate(chunk):
                positions = [p for p, r in enumerate(exs[b].raw_index) if r >= n_before]
                if len(positions) != len(ids):
                    continue  # truncated: candidate does not fit
                prev = torch.tensor([p - 1 for p in positions], device=hidden.device)
                logits = torch.nn.functional.linear(hidden[b, prev], lm.lm_head.weight).float()
                if lm.softcap:
                    logits = torch.tanh(logits / lm.softcap) * lm.softcap
                target = torch.tensor(ids, device=logits.device)
                lp = float(torch.log_softmax(logits, -1).gather(1, target[:, None]).sum())
                scores[cand] = lp / len(ids) if e.localize_score == "mean" else lp
    return scores


def evaluate_localization(lm, tokenizer, cfg, tasks: list[dict], chunk_size=None, log_path: str | None = None) -> dict:
    e = cfg.eval
    n = hit1 = hit5 = 0
    skipped = {"no_patch_files": 0, "no_gold": 0, "no_candidates": 0}
    log_f = open(log_path, "w") if log_path else None
    for task in tasks:
        changed = parse_patch(str(task.get(e.patch_field, "")))
        changed = {p: ls for p, ls in changed.items() if p.endswith(".py")}
        if not changed:
            skipped["no_patch_files"] += 1
            continue
        gold, scores = set(), {}
        for rel, lines in changed.items():
            path = resolve_snapshot_file(cfg, task, rel)
            if path is None:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            funcs = function_table(text)
            if not funcs:
                continue
            gold |= {f"{rel}::{q}" for q in touched_functions(funcs, lines)}
            cands = sorted({f["qualname"] for f in funcs})
            for c, sc in score_file_candidates(lm, tokenizer, cfg, task, rel, text, cands, chunk_size).items():
                scores[f"{rel}::{c}"] = sc
        if not gold:
            skipped["no_gold"] += 1
            continue
        if not scores:
            skipped["no_candidates"] += 1
            continue
        ranked = sorted(scores, key=scores.get, reverse=True)
        n += 1
        h1, h5 = topk_hit(ranked, gold, 1), topk_hit(ranked, gold, 5)
        hit1 += h1
        hit5 += h5
        if log_f:
            log_f.write(json.dumps({"task": task.get(e.task_id_field), "gold": sorted(gold), "top5": ranked[:5],
                                    "n_candidates": len(ranked), "top1": h1, "top5_hit": h5}) + "\n")
    if log_f:
        log_f.close()
    return {"variant": lm.variant, "top1": hit1 / max(n, 1), "top5": hit5 / max(n, 1), "n_tasks": n, "skipped": skipped}
