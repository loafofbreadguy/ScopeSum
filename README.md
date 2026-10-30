# scopesum — scope-aligned summary tokens for code LLMs

**Idea.** When a coding model compresses past context into summary tokens, the
summary boundaries should follow code structure (functions/classes), not
fixed-size chunks. We test this on Gemma 4 (E2B/E4B). A causal, indentation-based
scope tracker decides where each Python scope ends; there we insert a `<sum>`
token that attends to the raw tokens of its scope (and to the `<sum>` tokens of
scopes nested inside it). Later tokens in the model's **local (sliding-window)
layers** see the `<sum>` instead of the scope's old raw tokens. **Global layers
keep normal causal attention.** One config flag switches between the variants:

| ID | Name | Local-layer attention of a normal token | `<sum>` tokens |
|---|---|---|---|
| A | baseline | model default (sliding window W) | none |
| B | fixed-chunk summaries | BOS + last W raw tokens + all earlier `<sum>` | one every K tokens |
| C | scope summaries (main) | same as B | one at the end of each scope |
| D | C + scope windows | BOS + own scope's raw tokens + enclosing headers + last 64 tokens + all earlier `<sum>` | same as C |
| E | D + graph edges | same as D | `<sum>` also attends to callee `<sum>` |
| F | C, KV-matched | same as B | like C, but a scope whose tokens span > W raw tokens is cut at line starts into parts, each with its own `<sum>` (a child of the scope's `<sum>`) |
| G | fixed chunks, matched to F | same as B | one every K = 180 tokens |

Fairness: B's K is fitted on the training data so that B inserts the same
total number of `<sum>` tokens as C (`stats.json`: `chunk_size`,
`scope_sums_total`, `fixed_sums_total`). B and C differ only in where
boundaries fall. `<sum>` tokens are never predicted (labels `-100`).

C keeps more KV live than B in the local layers: a long scope's raw tokens
must stay until its `<sum>`, so C's live KV can exceed W. **F/G** is the
pair matched on both `<sum>` count and memory: F caps every part at W raw
tokens (no raw token outlives the window, header tokens excepted), and G's
K = 180 gives the same `<sum>` count as F (57,820 vs 58,536 on the training
windows). On 60 training windows at W = 512, mean/max live KV per local layer
is B 288/516, C 360/843, F 294/535. F and G examples are built from the saved
windows on first use (`data.extra_dir`), so `01_data_prep` need not be rerun.

## Repository layout

```
configs/      base.yaml, variant_A..E.yaml, hw_t4.yaml (float16), hw_l4.yaml (bfloat16)
scopesum/     pure Python/NumPy core: scope_tracker, segmenter, sum_insertion, masks, graph_edges
              torch/transformers: model, train, eval_longrange, eval_localize, eval_memory, probe, diagnose, report
              data (pipeline; heavy imports are lazy), config (YAML -> dataclasses)
tests/        pytest (CPU only; test_model_hook builds a tiny random Gemma 4)
notebooks/    00_smoke_test, 01_data_prep, 02_train, 03_eval, 04_report, 05_diagnose
```

## Getting the code onto Kaggle

Every notebook's first code cell gets the code in one of two ways:

* **Clone (default)**: `REPO_URL = "https://github.com/loaferct/scopesum.git"`.
  The repo is private, so Kaggle needs a token: create a GitHub fine-grained
  token with *Contents: Read-only* on this repo, then in the notebook editor
  open **Add-ons → Secrets**, add it with the label `GITHUB_TOKEN`, and tick
  it for the notebook. Internet must be ON. The token is never printed and is
  removed from `.git/config` after cloning.
* **Attached dataset (fallback)**: if the clone fails, or `REPO_URL = ""`, the
  cell looks for this repo inside an attached Kaggle dataset (e.g.
  `scopesum-code`) and copies it to `/kaggle/working/scopesum`.

## How to run the tests on Kaggle

```
!pip install -r requirements.txt && pytest -q tests
```

(in a notebook cell: `!pip install -q -r requirements.txt` then
`!python -m pytest -q tests -p no:cacheprovider`; `00_smoke_test` and
`01_data_prep` already do this.)

## Kaggle run order

| # | Notebook | Accelerator | Internet | Attach | Produces |
|---|---|---|---|---|---|
| 0 | `00_smoke_test` | GPU T4 x2 or L4 | ON | code | printed checks only |
| 1 | `01_data_prep` | None (CPU) | ON | code | `/kaggle/working/scopesum_data` → save as dataset `scopesum-data` |
| 2 | `02_train` (once per variant: set `VARIANT`) | GPU L4 (preferred) or T4 x2 | ON | code, `scopesum-data`, previous run (to resume) | `/kaggle/working/scopesum_runs/variant_X/{ckpt-N,final}` → save as dataset `scopesum-run-x` |
| 3 | `03_eval` | GPU L4 or T4 x2 | ON | code, `scopesum-data`, all run datasets (SWE-bench Lite and the 4 eval repos are downloaded from the Hub/GitHub) | `/kaggle/working/scopesum_eval/{A..E}.json`, CSVs |
| 4 | `04_report` | None | OFF ok | code, eval output | `/kaggle/working/report/{main_table.tex, *.csv, b_vs_c.png, memory.png}` |
| 5 | `05_diagnose` | GPU T4 x2 or L4 | ON | code, data, run outputs | `/kaggle/working/scopesum_diag/{A..G}.json`, `diagnose_summary.csv`, `diagnose_by_offset.csv` |

Train A, B, C, D (and E, F, G) with the same data and seed: only `VARIANT` changes.

`05_diagnose` checks whether the summaries are used at all: loss with `<sum>`
hidden from raw queries (sliding layers only / all layers), attention share on
`<sum>`, BOS and raw tokens >= W back per layer type (eager attention), and
loss by distance from the start of the token's scope. Global layers see every
raw token, so a model can route around its summaries. The probes in `03` also
report a mean-pooled raw-token baseline and accuracy split by whether a
function crosses a B chunk boundary (the per-function tasks favour scope
summaries by construction).

### Resuming training across sessions

`02_train` saves `ckpt-N/` every `train.save_every` steps and stops cleanly
after `train.max_hours` (default 11 h). Each checkpoint contains the LoRA
adapter, the `<sum>` embedding, optimizer, LR scheduler, GradScaler, RNG
states and the data position. To continue:

1. *Save Version* of the notebook (its output keeps `/kaggle/working/scopesum_runs`).
2. Create a Kaggle dataset from that output (e.g. `scopesum-run-c`).
3. In the next session attach it and set
   `RESUME_FROM = "/kaggle/input/scopesum-run-c/scopesum_runs/variant_C"`
   (a run dir — the newest checkpoint is picked via `LATEST` — or a specific `ckpt-N`).

The training log `train_log.csv` (loss, lr, grad norm, tokens/sec, peak GPU
memory) is appended across sessions.

## Design notes

* **Mask hook** (`scopesum/model.py`, based on `transformers==5.18.0`,
  `models/gemma4/modeling_gemma4.py`): `Gemma4TextModel.forward` accepts
  `attention_mask` as a dict `{"full_attention": ..., "sliding_attention": ...}`
  and gives each layer the mask of its type. Variants B–E pass a dense
  additive `[B, 1, L, L]` mask for sliding layers and the standard causal mask
  for full layers; variant A passes the plain 2D mask (library default path).
  Works with `sdpa` and `eager` (FlashAttention 2 is not needed).
* **`<sum>` token**: Gemma 4's tokenizer reserves 6,229 `<unusedN>` tokens;
  `<unused0>` (id 6) is reused as `<sum>`. A forward hook replaces its
  embedding with a trainable vector initialised to the mean embedding row.
  This is equivalent to adding a new token, but avoids resizing both the
  262k-row embedding and Gemma 4's per-layer-embedding table.
* **Scope tracker** (`scope_tracker.py`): logical-line lexer (brackets,
  backslashes, strings/docstrings, comments) + CPython-like indentation stack.
  Decorators belong to the following def; blank/comment lines never close a
  scope; module-level code forms "module run" segments between top-level
  definitions; broken indentation marks tokens as unknown (never raises).
* **Windows**: files are cut into ≤ `seq_len − 1 − sum_reserve` raw tokens at
  top-level boundaries; all variants use identical windows in identical order.
* **Loss**: chunked, soft-capped (Gemma `final_logit_softcapping`), checkpointed
  per chunk, so the `[L, 262144]` logits are never fully materialised.
* **Data**: default `bigcode/the-stack-smol-xl` (`data/python`, 10k files of
  The Stack, permissively licensed source files; each row carries its repo
  licenses; follow The Stack's terms of use). Files from `fastapi`,
  `requests`, `rich`, `httpx` are excluded by repo name, path components,
  author/header signatures and private-submodule imports. The 12 SWE-bench Lite
  repos (django, sympy, …) are excluded by exact repo name.
* **Evaluation sets**: long-range loss on shallow clones of fastapi, requests,
  rich and httpx (HEAD at eval time); bug localization on SWE-bench Lite
  (`princeton-nlp/SWE-bench_Lite`, 300 issues with reference patches; only the
  patched files are fetched from GitHub at each task's `base_commit`).

## Things to verify on the first Kaggle run

- [ ] **Model ID / source**: `google/gemma-4-E2B` downloads (or set `model.source=kagglehub` + `kagglehub_handle`, or `local` + `local_path`). `00` prints the class (`Gemma4ForConditionalGeneration` expected) and the layer layout: E2B should show 35 layers, `SSSSF` pattern, sliding window 512, 20 KV-shared layers.
- [ ] **`<sum>` token**: `<unused0>` maps to a single id (6) and is not `<unk>` (`00`, `01`).
- [ ] **Mask hook working**: `00` mask sanity check passes (hidden token → no change at the first local layer; visible token → change), and variant A through our wrapper matches the library forward (max |Δlogit| ≈ 0).
- [ ] **LoRA targets**: `[model] LoRA attached to N linear layers` with N > 0 (if 0, fix `lora.target_regex`; module names depend on the loaded class).
- [ ] **float16 NaNs on T4**: `00` prints NaN/inf counts per variant and the max |hidden| per layer. If NaNs appear, use L4 + bfloat16 (`HW="l4"`), or `attn_implementation: eager`.
- [ ] **tokens/sec and memory**: `00` prints fwd+bwd tokens/sec and peak memory; use it to size `data.token_budget`, `train.grad_accum` and `train.max_hours`.
- [ ] **Dataset field names**: `01` discovery prints the row keys (`content`, `max_stars_repo_name`, `max_stars_repo_path` expected) and offset-mapping checks.
- [ ] **K for variant B**: `01` prints `chunk_size` and the B/C `<sum>` totals; `relative_count_gap` should be ≈ 0.
- [ ] **Localization tasks (SWE-bench Lite)**: `03` prints the dataset keys (`instance_id`, `problem_statement`, `patch`, `repo`, `base_commit` expected) and checks that the first task's patched files can be fetched from GitHub at `base_commit`. The 4 long-range repos are cloned from GitHub (the clone log shows each HEAD commit).
- [ ] **Paths of attached datasets** (`DATA_DIR`, `CHECKPOINTS`, `EVAL_DIR`) match what Kaggle mounted under `/kaggle/input`.
