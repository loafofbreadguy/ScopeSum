"""Load and merge YAML configs into dataclasses.

Usage::

    cfg = load_config("configs/base.yaml", "configs/variant_C.yaml",
                      overrides={"train": {"max_steps": 10}})

Later files/overrides win. Unknown keys raise an error so typos are caught.
Values that we could not verify offline (model ids, dataset field names,
competition paths) live here and are printed by the discovery helpers in the
notebooks.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

try:  # PyYAML ships with transformers / Kaggle images
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class ModelConfig:
    model_id: str = "google/gemma-4-E2B"  # base (pretrained, not -it) checkpoint
    source: str = "hf"  # hf | kagglehub | local
    kagglehub_handle: str = "google/gemma-4/transformers/gemma-4-e2b"  # VERIFY on Kaggle
    local_path: str | None = None  # used when source == local (e.g. attached Kaggle model)
    model_class: str = "auto"  # auto | causal_lm | conditional_generation
    dtype: str = "float16"  # float16 (T4) | bfloat16 (L4/TPU) | float32
    load_in_4bit: bool = False
    attn_implementation: str = "sdpa"  # sdpa | eager
    device_map: str | None = None  # None = single cuda:0, "auto" = split over GPUs
    drop_multimodal_towers: bool = True  # free vision/audio encoders (we only feed text)
    sum_token: str = "<unused0>"  # reserved Gemma token reused as <sum>
    sum_token_mode: str = "reuse"  # reuse (recommended) | add
    reinit_ple_row: bool = True  # set the <sum> row of the per-layer embedding to the mean row
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Only text-decoder projections (not the vision/audio towers).
    target_regex: str = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


@dataclass
class VariantConfig:
    id: str = "A"  # A | B | C | D | E | F | G | S (SepLLM-style: separators, no <sum>)
    window: int | None = None  # W for B/C/F/G; None = model config sliding_window
    w_small: int = 64  # D/E recent-token window
    chunk_size: int | None = None  # K for B (None = stats.json, fairness vs C); G must set it (fairness vs F)
    all_descendants: bool = False  # <sum> sees all nested sums instead of direct children only
    scope_span: int | None = None  # F: max raw-token span of one scope part; None = W
    restrict_full: bool = False  # full-attention layers get the local-layer mask too: far context only via <sum>
    keep_signatures: bool = False  # B/C/F/G: raw queries also see every earlier def/class header line

    @property
    def uses_sums(self) -> bool:
        return self.id not in ("A", "S")

    @property
    def segment_mode(self) -> str:
        return {"A": "none", "S": "none", "B": "fixed", "F": "scope_capped", "G": "fixed"}.get(self.id, "scope")


@dataclass
class DataConfig:
    dataset_id: str = "bigcode/the-stack-smol"  # VERIFY: license + availability
    dataset_config: str | None = None
    data_dir: str | None = "data/python"
    split: str = "train"
    streaming: bool = False
    text_field: str = "content"  # VERIFY with discovery
    repo_field: str = "max_stars_repo_name"  # VERIFY with discovery
    path_field: str = "max_stars_repo_path"  # VERIFY with discovery
    local_files_glob: str | None = None  # alternative source: glob of .py files
    excluded_repos: list[str] = field(default_factory=lambda: ["fastapi", "requests", "rich", "httpx"])
    # Exact repo-name matches only (no substring rule): SWE-bench Lite repos used by the localization eval.
    excluded_repos_exact: list[str] = field(
        default_factory=lambda: [
            "astropy", "django", "matplotlib", "seaborn", "flask", "requests", "xarray",
            "pylint", "pytest", "scikit-learn", "sphinx", "sympy",
        ]
    )
    min_tokens: int = 1024
    seq_len: int = 2048
    sum_reserve: int = 256  # positions kept free for <sum> tokens (raw window = seq_len - 1 - this)
    min_window_tokens: int = 256  # drop shorter tail windows
    token_budget: int = 10_000_000  # raw training tokens per variant
    heldout_fraction: float = 0.01
    max_files: int | None = None
    out_dir: str = "/kaggle/working/scopesum_data"
    extra_dir: str = "/tmp/scopesum_data_extra"  # examples of modes built after 01 (e.g. scope_capped)
    seed: int = 1234


@dataclass
class TrainConfig:
    out_dir: str = "/kaggle/working/scopesum_runs"
    resume_from: str | None = None  # checkpoint dir (e.g. attached Kaggle dataset)
    lr: float = 2e-4
    sum_embed_lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 50
    batch_size: int = 1
    grad_accum: int = 16
    max_steps: int | None = None  # None = derived from data.token_budget
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every: int = 100
    keep_last_checkpoints: int = 2
    seed: int = 1234
    loss_chunk: int = 512  # positions per chunk in the memory-saving LM loss
    max_hours: float | None = 11.0  # stop cleanly (and checkpoint) before Kaggle kills the session


@dataclass
class EvalConfig:
    out_dir: str = "/kaggle/working/scopesum_eval"
    checkpoints: dict[str, str] = field(default_factory=dict)  # variant id -> checkpoint dir
    # long-range loss
    long_range_distance: int = 512
    eval_files_glob: str | None = None  # .py files of the 4 evaluation repos (None = clone them, see below)
    max_eval_files: int | None = 200
    eval_repos_dir: str = "/kaggle/working/eval_repos"
    eval_repo_urls: list[str] = field(
        default_factory=lambda: [
            "https://github.com/fastapi/fastapi",
            "https://github.com/psf/requests",
            "https://github.com/Textualize/rich",
            "https://github.com/encode/httpx",
        ]
    )
    # bug localization: "swebench" (HF dataset, files fetched from GitHub at base_commit)
    # or "jsonl" (tasks.jsonl + snapshots/ folder layout)
    localize_source: str = "swebench"
    swebench_dataset: str = "princeton-nlp/SWE-bench_Lite"
    swebench_split: str = "test"
    commit_field: str = "base_commit"
    file_cache_dir: str = "/kaggle/working/localize_files"
    competition_dir: str = "/kaggle/input/localize-tasks"  # jsonl mode only
    tasks_file: str = "tasks.jsonl"
    snapshots_dir: str = "snapshots"
    graphs_dir: str = "graphs"
    task_id_field: str = "instance_id"
    issue_field: str = "problem_statement"
    patch_field: str = "patch"
    repo_field: str = "repo"
    snapshot_field: str = "snapshot"  # jsonl mode: key giving the snapshot folder of a task
    max_tasks: int | None = None
    localize_prompt: str = "\n\n# Bug report:\n{issue}\n# The function that must be edited to fix this bug is named: "
    max_issue_tokens: int = 256
    localize_score: str = "sum"  # sum | mean log-prob of the candidate name tokens
    localize_batch_size: int = 4
    localize_edges: str = "regex"  # variant E edges at eval: regex | competition
    graph_field: str = "graph"  # VERIFY: task key naming its file under graphs_dir (competition edges)
    graph_schema: dict = field(default_factory=dict)  # overrides for graph_edges.GraphSchema field names
    # probes
    probe_layers: list[int] = field(default_factory=lambda: [-1])  # hidden-state layers (-1 = last)
    probe_max_files: int = 300
    # memory
    memory_seq_len: int = 2048


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    variant: VariantConfig = field(default_factory=VariantConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build(cls, values: dict) -> Any:
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(names)
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for k, v in values.items():
        sub = _SECTIONS.get(k) if cls is Config else None
        kwargs[k] = _build(sub, v or {}) if sub is not None else v
    return cls(**kwargs)


_SECTIONS = {
    "model": ModelConfig,
    "lora": LoraConfig,
    "variant": VariantConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def config_from_dict(d: dict) -> Config:
    return _build(Config, d or {})


def read_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required to read config files (pip install pyyaml)")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(*paths: str, overrides: dict | None = None) -> Config:
    merged: dict = {}
    for p in paths:
        merged = deep_merge(merged, read_yaml(p))
    merged = deep_merge(merged, overrides or {})
    return config_from_dict(merged)
