"""Training loop with checkpoint + resume (Kaggle sessions die after 9-12 h).

* Same seed and same data order for every variant: all modes store the same
  windows in the same order, and the shuffling permutation depends only on
  ``train.seed`` and the epoch.
* A checkpoint ``<run_dir>/ckpt-<step>/`` holds: LoRA adapter, ``<sum>``
  embedding, optimizer, scheduler, GradScaler, RNG states (python, numpy,
  torch, cuda) and the data position. ``<run_dir>/LATEST`` names the newest.
* Resume: set ``train.resume_from`` to a checkpoint dir *or* a run dir (the
  newest checkpoint inside is used), e.g. an attached Kaggle dataset.

Usage::

    from scopesum.train import train
    train(cfg)
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch

from .config import Config
from .data import load_or_build_examples, load_stats, load_tokenizer, resolve_model_path
from .model import ScopeSumLM, build_model, load_weights, save_weights, torch_dtype


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(cfg: Config) -> str:
    return os.path.join(cfg.train.out_dir, f"variant_{cfg.variant.id}")


def resolve_variant_settings(cfg: Config, stats: dict) -> None:
    """Fill K for variant B from stats.json (fairness rule) if not set; G must set its own."""
    if cfg.variant.id == "B" and not cfg.variant.chunk_size:
        cfg.variant.chunk_size = int(stats["chunk_size"])
        print(f"[train] variant B chunk size K = {cfg.variant.chunk_size} (from stats.json)")
    if cfg.variant.id == "B" and int(stats.get("chunk_size", -1)) != cfg.variant.chunk_size:
        print("[train] NOTE: variant.chunk_size differs from stats.json; examples are rebuilt with this K")
    if cfg.variant.id == "G" and not cfg.variant.chunk_size:
        raise ValueError("variant G needs variant.chunk_size (K matched to F's <sum> count)")


def data_order(n: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng(seed + 1000 * epoch).permutation(n)


def lr_lambda(warmup: int, total: int, min_ratio: float = 0.0):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def find_checkpoint(path: str | None) -> str | None:
    """Accept a checkpoint dir or a run dir (use LATEST / highest step)."""
    if not path:
        return None
    if os.path.exists(os.path.join(path, "trainer_state.pt")):
        return path
    latest = os.path.join(path, "LATEST")
    if os.path.exists(latest):
        with open(latest) as f:
            cand = os.path.join(path, f.read().strip())
        if os.path.exists(os.path.join(cand, "trainer_state.pt")):
            return cand
    cands = sorted(glob.glob(os.path.join(path, "**", "ckpt-*", "trainer_state.pt"), recursive=True))
    if not cands:
        raise FileNotFoundError(f"no checkpoint under {path}")
    cands.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).split("-")[1]))
    return os.path.dirname(cands[-1])


def save_checkpoint(lm: ScopeSumLM, opt, sched, scaler, step: int, data_pos: int, epoch: int, cfg: Config) -> str:
    rd = run_dir(cfg)
    ck = os.path.join(rd, f"ckpt-{step}")
    tmp = ck + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    save_weights(lm, tmp)
    torch.save(
        {
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "data_pos": data_pos,
            "epoch": epoch,
            "rng": rng_state(),
            "variant": cfg.variant.id,
            "config": cfg.to_dict(),
        },
        os.path.join(tmp, "trainer_state.pt"),
    )
    shutil.rmtree(ck, ignore_errors=True)
    os.replace(tmp, ck)  # atomic: a killed session never leaves a half-written ckpt-N
    with open(os.path.join(rd, "LATEST"), "w") as f:
        f.write(os.path.basename(ck))
    old = sorted(glob.glob(os.path.join(rd, "ckpt-*")), key=lambda p: int(p.rsplit("-", 1)[1]) if p.rsplit("-", 1)[1].isdigit() else -1)
    for p in old[: -cfg.train.keep_last_checkpoints]:
        shutil.rmtree(p, ignore_errors=True)
    print(f"[train] saved {ck}")
    return ck


def build_optimizer(lm: ScopeSumLM, cfg: Config):
    lora, extra = lm.trainable_parameters()
    groups = [{"params": lora, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay}]
    if extra:
        groups.append({"params": extra, "lr": cfg.train.sum_embed_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


@torch.no_grad()
def evaluate_loss(lm: ScopeSumLM, ds, max_examples: int = 64) -> dict:
    lm.eval()
    tot, n = 0.0, 0
    for i in range(min(len(ds), max_examples)):
        batch = lm.make_batch([ds[i]])
        with torch.autocast("cuda", dtype=lm.dtype, enabled=torch.cuda.is_available() and lm.dtype != torch.float32):
            out = lm(batch)
        tot += float(out["per_token_loss"].sum())
        n += int(out["n_targets"])
    lm.train()
    loss = tot / max(n, 1)
    return {"heldout_loss": loss, "heldout_tokens": n}


def train(cfg: Config) -> str:
    """Train one variant; returns the final checkpoint dir."""
    tc = cfg.train
    set_seed(tc.seed)
    stats = load_stats(cfg.data.out_dir)
    resolve_variant_settings(cfg, stats)
    path = resolve_model_path(cfg)
    tok = load_tokenizer(cfg, path)
    sum_id = int(stats["sum_id"])
    if tok.convert_tokens_to_ids(cfg.model.sum_token) != sum_id:
        raise ValueError("tokenizer <sum> id differs from the one used to build the data")

    lm = build_model(cfg, path, sum_id, for_training=True)
    lm.train()

    mode = lm.mode
    ds = load_or_build_examples(cfg, mode, "train", lm.scope_span, cfg.variant.chunk_size)
    try:
        heldout = load_or_build_examples(cfg, mode, "heldout", lm.scope_span, cfg.variant.chunk_size)
    except Exception:  # heldout split may be empty in debug runs
        heldout = None
    n = len(ds)
    per_step = tc.batch_size * tc.grad_accum
    total_steps = tc.max_steps or max(1, n // per_step)
    span = f" scope_span={lm.scope_span}" if mode == "scope_capped" else ""
    print(f"[train] variant={cfg.variant.id} mode={mode}{span} examples={n} steps={total_steps} (x{per_step} examples)")
    opt = build_optimizer(lm, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc.warmup_steps, total_steps))
    use_fp16 = torch_dtype(cfg.model.dtype) == torch.float16 and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    amp_enabled = torch.cuda.is_available() and cfg.model.dtype != "float32"

    step, data_pos, epoch = 0, 0, 0
    ck = find_checkpoint(tc.resume_from)
    if ck:
        load_weights(lm, ck)
        state = torch.load(os.path.join(ck, "trainer_state.pt"), map_location="cpu", weights_only=False)
        if state["variant"] != cfg.variant.id:
            raise ValueError(f"checkpoint is for variant {state['variant']}, config says {cfg.variant.id}")
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        step, data_pos, epoch = state["step"], state["data_pos"], state["epoch"]
        set_rng_state(state["rng"])
        print(f"[train] resumed from {ck}: step={step} data_pos={data_pos} epoch={epoch}")

    rd = run_dir(cfg)
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "config.json"), "w") as f:
        f.write(cfg.to_json())
    log_path = os.path.join(rd, "train_log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "loss", "lr", "grad_norm", "tokens_per_sec", "peak_mem_gb", "elapsed_min", "scale", "n_sum"])

    order = data_order(n, tc.seed, epoch)
    t_start = time.time()
    t_win, tok_win, loss_win, nsum_win = time.time(), 0, [], 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(tc.grad_accum):
            if data_pos + tc.batch_size > n:
                epoch += 1
                data_pos = 0
                order = data_order(n, tc.seed, epoch)
            idx = order[data_pos : data_pos + tc.batch_size]
            data_pos += tc.batch_size
            batch = lm.make_batch([ds[int(i)] for i in idx])
            with torch.autocast("cuda", dtype=lm.dtype, enabled=amp_enabled):
                out = lm(batch)
            loss = out["loss"] / tc.grad_accum
            if not torch.isfinite(loss):
                print(f"[train] WARNING non-finite loss at step {step}; skipping micro-batch")
                continue
            (scaler.scale(loss) if scaler is not None else loss).backward()
            micro_losses.append(float(out["loss"]))
            tok_win += batch["n_raw"]
            nsum_win += batch["n_sum"]

        params = [p for g in opt.param_groups for p in g["params"]]
        if not micro_losses:
            grad_norm = float("nan")  # every micro-batch was non-finite: skip the update
        else:
            if scaler is not None:
                scaler.unscale_(opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, tc.max_grad_norm))
            if scaler is not None:
                scaler.step(opt)  # skips the update itself if grads contain inf/nan
                scaler.update()
            elif math.isfinite(grad_norm):
                opt.step()
        sched.step()
        step += 1
        loss_win += micro_losses

        if step % tc.log_every == 0 or step == total_steps:
            dt = time.time() - t_win
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            row = [
                step,
                float(np.mean(loss_win)) if loss_win else float("nan"),
                sched.get_last_lr()[0],
                grad_norm,
                tok_win / max(dt, 1e-6),
                peak,
                (time.time() - t_start) / 60,
                scaler.get_scale() if scaler is not None else 1.0,
                nsum_win,
            ]
            log.writerow(row)
            log_f.flush()
            print(
                f"[train] step {step}/{total_steps} loss={row[1]:.4f} lr={row[2]:.2e} gnorm={grad_norm:.3f} "
                f"tok/s={row[4]:.0f} peak={peak:.1f}GB"
            )
            t_win, tok_win, loss_win, nsum_win = time.time(), 0, [], 0

        out_of_time = tc.max_hours is not None and (time.time() - t_start) / 3600 > tc.max_hours
        if step % tc.save_every == 0 or step == total_steps or out_of_time:
            save_checkpoint(lm, opt, sched, scaler, step, data_pos, epoch, cfg)
        if out_of_time:
            print(f"[train] stopping after {tc.max_hours} h; resume with train.resume_from={rd}")
            log_f.close()
            return find_checkpoint(rd)

    log_f.close()
    final = os.path.join(rd, "final")
    shutil.rmtree(final, ignore_errors=True)
    save_weights(lm, final)
    summary = {"variant": cfg.variant.id, "steps": step, "examples_seen": step * per_step}
    if heldout is not None and len(heldout) > 0:
        summary.update(evaluate_loss(lm, heldout))
    with open(os.path.join(final, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] done: {summary}")
    return final
