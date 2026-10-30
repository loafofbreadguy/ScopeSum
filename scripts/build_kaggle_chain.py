"""Build a Kaggle kernel that trains and diagnoses several runs in one session.

Usage:
  python scripts/build_kaggle_chain.py OWNER DATASET NAME OUT_DIR RUN [RUN ...]
  RUN = TAG:VARIANT:SEED[:key=value,...]   e.g.  A2:A:2   Frf2:F:2:restrict_full=1   G2:G:2:chunk_size=180

Each run trains 130 steps (W=512 unless window=... is given) and then runs the
<sum> diagnostics on its own checkpoint. Runs write to /kaggle/working/runs_<TAG>
and /kaggle/working/diag_<TAG>, so variants and seeds never collide. Three runs
take about 10 hours on a T4, inside Kaggle's 12-hour session limit.
"""

from __future__ import annotations

import copy
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def src(c):
    return "".join(c["source"])


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


def rep(s, a, b):
    assert a in s, a
    return s.replace(a, b)


def parse_run(spec: str) -> tuple[str, str, int, dict]:
    tag, variant, seed, *rest = spec.split(":")
    extra = {"window": 512}
    for kv in (rest[0].split(",") if rest else []):
        k, v = kv.split("=")
        extra[k] = True if v in ("1", "true") else int(v)
    return tag, variant, int(seed), extra


def build(owner: str, dataset: str, name: str, runs: list[str]) -> dict:
    tr = json.load(open(os.path.join(REPO, "notebooks/02_train.ipynb")))
    dg = json.load(open(os.path.join(REPO, "notebooks/05_diagnose.ipynb")))
    setup = rep(src(tr["cells"][1]), 'REPO_URL = "https://github.com/loaferct/scopesum.git"', 'REPO_URL = ""')
    setup = "\n".join("MAX_STEPS = 130" if l.startswith("MAX_STEPS") else "WARMUP_STEPS = 13" if l.startswith("WARMUP_STEPS") else l
                      for l in setup.splitlines())
    cfg_cell = rep(src(tr["cells"][3]), '"train": {"resume_from": RESUME_FROM, "max_steps": MAX_STEPS}',
                   '"variant": VARIANT_EXTRA, "train": {"resume_from": RESUME_FROM, "max_steps": MAX_STEPS, "seed": SEED, "out_dir": f"/kaggle/working/runs_{TAG}"}')
    diag_setup = src(dg["cells"][3])
    diag_setup = rep(diag_setup, '"eval": dict(EVAL_OVERRIDES)})', '"variant": VARIANT_EXTRA, "eval": dict(EVAL_OVERRIDES)})')
    diag_setup = rep(diag_setup, 'if not os.path.exists(os.path.join(DATA_DIR, "stats.json")):', "if False:  # DATA_DIR resolved by the training cell\n   ")
    diag_setup = rep(diag_setup, "if not CHECKPOINTS:", "CHECKPOINTS = {VARIANT: final_dir}\nif False:")
    diag_setup = rep(diag_setup, 'OUT = "/kaggle/working/scopesum_diag"', 'OUT = f"/kaggle/working/diag_{TAG}"')
    diag_setup = 'EVAL_OVERRIDES = {"max_eval_files": 60, "eval_repos_dir": "/tmp/eval_repos"}  # clones outside /kaggle/working\nN_ATTENTION = 16\n' + diag_setup

    cells = [{"cell_type": "markdown", "metadata": {}, "source": [f"# Chained runs: {', '.join(runs)}\n"]}, code(setup), tr["cells"][2]]
    for i, spec in enumerate(runs):
        tag, variant, seed, extra = parse_run(spec)
        cells.append(code(f'import gc, torch\ngc.collect(); torch.cuda.empty_cache()\nTAG, VARIANT, SEED = "{tag}", "{variant}", {seed}\n'
                          f"VARIANT_EXTRA = {extra!r}\nprint('run', TAG, VARIANT, SEED, VARIANT_EXTRA)\n"))
        cells += [code(cfg_cell), copy.deepcopy(tr["cells"][4]), code(diag_setup)]
        if i == 0:
            cells.append(copy.deepcopy(dg["cells"][4]))  # eval files: cloned and tokenized once
        cells += [copy.deepcopy(dg["cells"][5]), copy.deepcopy(dg["cells"][6])]
    nb = copy.deepcopy(tr)
    nb["cells"] = cells
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    meta = {"id": f"{owner}/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
            "dataset_sources": [dataset], "kernel_sources": [], "competition_sources": [], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    return nb, meta


def main():
    owner, dataset, name, out, *runs = sys.argv[1:]
    nb, meta = build(owner, dataset, name, runs)
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, f"{name}.ipynb"), "w"), indent=1)
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    print(d, [parse_run(r) for r in runs])


if __name__ == "__main__":
    main()
