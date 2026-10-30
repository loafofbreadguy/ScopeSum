"""Gemma 4 + <sum> embedding + LoRA + custom sliding-layer masks.

How the mask hook works (based on transformers **v5.18.0**,
``src/transformers/models/gemma4/modeling_gemma4.py``):

* ``Gemma4TextModel.forward`` does::

      if not isinstance(causal_mask_mapping := attention_mask, dict):
          causal_mask_mapping = {
              "full_attention": create_causal_mask(**mask_kwargs),
              "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
          }
      ...
      decoder_layer(..., attention_mask=causal_mask_mapping[self.config.layer_types[i]], ...)

  So passing ``attention_mask`` as a dict with keys ``"full_attention"`` and
  ``"sliding_attention"`` makes every layer use exactly the 4D mask we give
  it for its layer type. The multimodal wrapper ``Gemma4Model.forward`` has
  the same dict check and forwards the dict to ``language_model``.
* ``eager_attention_forward`` adds the mask to the scores (so it must be an
  additive float mask); ``sdpa_attention_forward``
  (``integrations/sdpa_attention.py``) passes it to
  ``F.scaled_dot_product_attention`` and sets ``is_causal=False`` when a mask
  is given. Our masks are additive float tensors ``[B, 1, L, L]`` with ``0``
  for "attend" and ``finfo(dtype).min`` for "blocked" (``-inf`` would give NaN
  in float16 if a row were ever fully blocked; every row keeps its diagonal).
* KV-shared layers (``num_kv_shared_layers``; the last 20 of 35 layers in
  E2B) reuse K/V computed by the last non-shared layer of the same type, and
  still receive the mask of their own layer type, so our sliding mask
  applies to them as well.

We call the text decoder (``Gemma4TextModel``) directly:

* Variant A passes the plain 2D padding mask -> the library builds both
  masks itself (default code path, unchanged).
* Variants B-E pass ``{"full_attention": causal+padding (same as default),
  "sliding_attention": our variant mask}``.

``<sum>`` token: an existing reserved token (``<unused0>``) is reused; a
forward hook on ``embed_tokens`` replaces its embedding with a trainable
vector (initialised to the mean embedding row). This avoids training or
resizing the 262k-row embedding and the per-layer-embedding (PLE) tables.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import Config
from .masks import MaskMeta, build_sliding_mask, full_causal_mask
from .sum_insertion import IGNORE_INDEX, KIND_PAD, KIND_RAW, KIND_SUM, Example

SLIDING = "sliding_attention"
FULL = "full_attention"


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


# ---------------------------------------------------------------------------
# locating submodules
# ---------------------------------------------------------------------------


def find_text_model(model: nn.Module) -> nn.Module:
    """The Gemma 4 text decoder (has ``layers``, ``embed_tokens``, ``config.layer_types``)."""
    for _, mod in model.named_modules():
        if (
            type(mod).__name__.endswith("TextModel")
            and hasattr(mod, "layers")
            and hasattr(mod, "embed_tokens")
            and hasattr(getattr(mod, "config", None), "layer_types")
        ):
            return mod
    raise RuntimeError("could not find the Gemma 4 text decoder (…TextModel) inside the model")


def text_config(model: nn.Module):
    return find_text_model(model).config


def _cfg_get(c, name: str, default=None):
    """Config attribute; Gemma 4 raises for per-layer attributes read globally (e.g. head_dim)."""
    try:
        return getattr(c, name, default)
    except Exception:
        return default


def _attn_shape(attn: nn.Module) -> tuple[int | None, int | None]:
    """(head_dim, num_kv_heads) read from a layer's attention module (LoRA-wrapped or not)."""
    head_dim = getattr(attn, "head_dim", None)
    k = getattr(attn, "k_proj", None)
    out = getattr(k, "out_features", None) or getattr(getattr(k, "base_layer", None), "out_features", None)
    kv = out // head_dim if (out and head_dim) else None
    return head_dim, kv


def describe_attention_layout(model: nn.Module) -> dict:
    """What the smoke test prints: layer types, window, KV sharing, attention impl.

    Head sizes come from the attention modules themselves: in Gemma 4,
    ``head_dim``/``num_key_value_heads`` are per-layer config values
    (``config.per_layer_config[i]``) and differ between sliding and full layers.
    """
    tm = find_text_model(model)
    c = tm.config
    types = list(c.layer_types)
    n_shared = _cfg_get(c, "num_kv_shared_layers", 0) or 0
    first_shared = len(types) - n_shared
    own_kv = [i for i in range(first_shared)]
    sl = next((i for i in own_kv if types[i] == SLIDING), None)
    fl = next((i for i in own_kv if types[i] == FULL), None)
    s_head, s_kv = _attn_shape(tm.layers[sl].self_attn) if sl is not None else (None, None)
    f_head, f_kv = _attn_shape(tm.layers[fl].self_attn) if fl is not None else (None, None)
    return {
        "text_model_class": type(tm).__name__,
        "num_layers": len(types),
        "layer_types": types,
        "n_sliding": types.count(SLIDING),
        "n_full": types.count(FULL),
        "sliding_window": c.sliding_window,
        "num_kv_shared_layers": n_shared,
        "sliding_layers_with_own_kv": sum(1 for i, t in enumerate(types) if t == SLIDING and i < first_shared),
        "num_key_value_heads": s_kv,  # sliding layers (used by the analytic KV memory)
        "head_dim": s_head,  # sliding layers
        "full_num_key_value_heads": f_kv,
        "full_head_dim": f_head,
        "hidden_size": c.hidden_size,
        "final_logit_softcapping": _cfg_get(c, "final_logit_softcapping"),
        "attn_implementation": _cfg_get(c, "_attn_implementation"),
        "vocab_size": c.vocab_size,
    }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_base_model(cfg: Config, path: str) -> nn.Module:
    import transformers

    m = cfg.model
    dtype = torch_dtype(m.dtype)
    kwargs = {
        "dtype": dtype,
        "attn_implementation": m.attn_implementation,
        "trust_remote_code": m.trust_remote_code,
    }
    if m.device_map is not None:
        kwargs["device_map"] = m.device_map
    elif torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    if m.load_in_4bit:
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["lm_head", "vision", "audio", "embed_vision", "embed_audio"],
        )
    if m.model_class == "causal_lm":
        cls = transformers.AutoModelForCausalLM
    elif m.model_class == "conditional_generation":
        cls = transformers.AutoModelForImageTextToText
    elif m.model_class == "auto":
        cls = None
    else:
        raise ValueError(f"unknown model.model_class {m.model_class!r}")
    if cls is not None:
        model = cls.from_pretrained(path, **kwargs)
    else:
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(path, **kwargs)
        except (ValueError, KeyError) as e:
            print(f"[model] AutoModelForCausalLM failed ({e}); trying AutoModelForImageTextToText")
            model = transformers.AutoModelForImageTextToText.from_pretrained(path, **kwargs)
    print(f"[model] loaded {type(model).__name__} from {path}")
    return model


def drop_multimodal_towers(model: nn.Module) -> list[str]:
    """Free vision/audio encoders (only text is ever fed to the model)."""
    targets = []
    for parent_name, parent in model.named_modules():
        if "language_model" in parent_name:
            continue
        for name, child in parent._modules.items():
            if child is not None and name != "language_model" and re.search(r"(vision|audio)", name):
                targets.append((parent_name, parent, name))
    dropped = []
    for parent_name, parent, name in targets:
        parent._modules[name] = None
        dropped.append(f"{parent_name}.{name}".lstrip("."))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dropped


@torch.no_grad()
def mean_row(weight: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Mean of embedding rows in float32, chunked to avoid a full fp32 copy."""
    total = torch.zeros(weight.shape[1], dtype=torch.float32, device=weight.device)
    for s in range(0, weight.shape[0], chunk):
        total += weight[s : s + chunk].float().sum(0)
    return total / weight.shape[0]


class SumEmbedding(nn.Module):
    """Trainable embedding for ``<sum>`` injected by a forward hook on ``embed_tokens``.

    The vector is stored unscaled (like an embedding-table row); the hook
    multiplies it by the module's ``embed_scale`` exactly like
    ``Gemma4TextScaledWordEmbedding.forward`` does for real rows.
    """

    def __init__(self, init: torch.Tensor, sum_id: int):
        super().__init__()
        self.vector = nn.Parameter(init.detach().float().clone())
        self.sum_id = sum_id
        self._handle = None

    def attach(self, embed_tokens: nn.Module) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = embed_tokens.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        ids = inputs[0]
        where = (ids == self.sum_id).unsqueeze(-1)
        scale = getattr(module, "embed_scale", 1.0)
        if torch.is_tensor(scale):
            scale = scale.to(device=output.device, dtype=torch.float32)
        vec = (self.vector.to(output.device) * scale).to(output.dtype)
        return torch.where(where, vec, output)


def apply_lora(cfg: Config, model: nn.Module) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    lc = cfg.lora
    pcfg = LoraConfig(
        r=lc.r,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        target_modules=lc.target_regex,  # a string is matched as a full regex against module names
        bias="none",
    )
    model = get_peft_model(model, pcfg)
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    if n_lora == 0:
        raise RuntimeError(
            "LoRA matched no modules. Print [n for n, _ in model.named_modules()] and fix lora.target_regex"
        )
    print(f"[model] LoRA attached to {n_lora} linear layers")
    return model


# ---------------------------------------------------------------------------
# the wrapper used for training and evaluation
# ---------------------------------------------------------------------------


def additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """bool [B, L, L] (True = attend) -> additive float [B, 1, L, L]."""
    out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    out.masked_fill_(~mask, torch.finfo(dtype).min)
    return out[:, None]


def _chunk_ce(h: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, softcap: float | None) -> torch.Tensor:
    logits = F.linear(h, weight).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
    ).view(y.shape)


def lm_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    softcap: float | None,
    chunk: int = 512,
    use_checkpoint: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-saving causal LM loss.

    Shifts (hidden[t] predicts labels[t+1]), applies Gemma's final logit
    soft-capping, and computes logits chunk by chunk (each chunk is
    recomputed in backward when ``use_checkpoint``) so the full
    ``[L, 262144]`` logit matrix is never materialised.

    Returns (mean loss, per-token loss [B, L-1], number of target tokens).
    """
    h = hidden[:, :-1]
    y = labels[:, 1:]
    weight = lm_head.weight
    parts = []
    for s in range(0, h.shape[1], chunk):
        hs, ys = h[:, s : s + chunk], y[:, s : s + chunk]
        if use_checkpoint and torch.is_grad_enabled():
            parts.append(checkpoint(_chunk_ce, hs, ys, weight, softcap, use_reentrant=False))
        else:
            parts.append(_chunk_ce(hs, ys, weight, softcap))
    per_tok = torch.cat(parts, dim=1)
    n = (y != IGNORE_INDEX).sum()
    return per_tok.sum() / n.clamp(min=1), per_tok, n


class ScopeSumLM(nn.Module):
    """Holds the (LoRA) model, the <sum> embedding and the variant settings."""

    def __init__(self, cfg: Config, model: nn.Module, sum_embedding: SumEmbedding | None, sum_id: int):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.text_model = find_text_model(model)
        self.lm_head = model.get_output_embeddings()
        self.sum_embedding = sum_embedding
        self.sum_id = sum_id
        tc = self.text_model.config
        self.variant = cfg.variant.id
        self.window = cfg.variant.window or tc.sliding_window
        self.mode = cfg.variant.segment_mode
        self.scope_span = cfg.variant.scope_span or self.window  # variant F only
        # A at the model's own window: let the library build its masks (fast path)
        self.library_masks = self.variant == "A" and self.window == tc.sliding_window and not cfg.variant.restrict_full
        self.separator_ids = None  # variant S: set by build_model from the tokenizer
        self.softcap = getattr(tc, "final_logit_softcapping", None)
        self.dtype = torch_dtype(cfg.model.dtype)

    @property
    def device(self) -> torch.device:
        return self.text_model.embed_tokens.weight.device

    # -- batching ------------------------------------------------------------

    def make_batch(self, rows: list[dict | Example]) -> dict:
        """Stack examples and build masks on CPU (NumPy), then move to device."""
        exs = [r if isinstance(r, Example) else Example.from_dict(r) for r in rows]
        L = max(e.length for e in exs)
        if any(e.length != L for e in exs):
            raise ValueError("examples in a batch must have equal (padded) length")
        kind = np.array([e.kind for e in exs], dtype=np.int64)
        batch = {
            "input_ids": torch.tensor([e.input_ids for e in exs], dtype=torch.long),
            "labels": torch.tensor([e.labels for e in exs], dtype=torch.long),
            "kind": torch.from_numpy(kind),
            "pad_mask": torch.from_numpy((kind != KIND_PAD).astype(np.int64)),
            "n_raw": int((kind == KIND_RAW).sum()),
            "n_sum": int((kind == KIND_SUM).sum()),
        }
        if not self.library_masks:
            sliding, full = [], []
            for e in exs:
                meta = MaskMeta.from_example(e)
                if self.variant == "S":
                    meta.is_separator = np.isin(np.asarray(e.input_ids), self.separator_ids)
                sliding.append(
                    build_sliding_mask(
                        self.variant,
                        meta,
                        window=self.window,
                        w_small=self.cfg.variant.w_small,
                        all_descendants=self.cfg.variant.all_descendants,
                        keep_headers=self.cfg.variant.keep_signatures,
                    )
                )
                full.append(sliding[-1] if self.cfg.variant.restrict_full else full_causal_mask(meta))
            batch["sliding_mask"] = torch.from_numpy(np.stack(sliding))
            batch["full_mask"] = torch.from_numpy(np.stack(full))
        return batch

    def attention_masks(self, batch: dict):
        """What we pass as ``attention_mask`` to the text model."""
        dev = self.device
        if self.library_masks:
            return batch["pad_mask"].to(dev)  # default path: the library builds both masks
        return {
            FULL: additive_mask(batch["full_mask"].to(dev), self.dtype),
            SLIDING: additive_mask(batch["sliding_mask"].to(dev), self.dtype),
        }

    # -- forward -------------------------------------------------------------

    def forward(self, batch: dict, output_hidden_states: bool = False, compute_loss: bool = True) -> dict:
        dev = self.device
        input_ids = batch["input_ids"].to(dev)
        B, L = input_ids.shape
        position_ids = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        out = self.text_model(
            input_ids=input_ids,
            attention_mask=self.attention_masks(batch),
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        result = {"hidden": out.last_hidden_state}
        if output_hidden_states:
            result["hidden_states"] = out.hidden_states
        if compute_loss:
            loss, per_tok, n = lm_loss(
                out.last_hidden_state,
                batch["labels"].to(dev),
                self.lm_head,
                self.softcap,
                chunk=self.cfg.train.loss_chunk,
            )
            result.update(loss=loss, per_token_loss=per_tok, n_targets=n)
        return result

    def trainable_parameters(self):
        lora = [p for n, p in self.model.named_parameters() if p.requires_grad]
        extra = [self.sum_embedding.vector] if self.sum_embedding is not None else []
        return lora, extra


def build_model(cfg: Config, path: str, sum_id: int, for_training: bool = True) -> ScopeSumLM:
    """Load the base model, attach <sum> embedding and LoRA, return the wrapper."""
    model = load_base_model(cfg, path)
    if cfg.model.drop_multimodal_towers:
        dropped = drop_multimodal_towers(model)
        print(f"[model] dropped multimodal modules: {dropped}")
    tm = find_text_model(model)
    vocab = tm.embed_tokens.weight.shape[0]
    if cfg.model.sum_token_mode != "reuse" or sum_id >= vocab:
        raise NotImplementedError(
            "Only sum_token_mode=reuse is supported: Gemma 4 has a second (per-layer) embedding table "
            "that would also need resizing. Use a reserved <unusedN> token."
        )
    for p in model.parameters():
        p.requires_grad_(False)

    sum_emb = None
    if cfg.variant.uses_sums:
        sum_emb = SumEmbedding(mean_row(tm.embed_tokens.weight), sum_id)
        sum_emb.attach(tm.embed_tokens)
        ple = getattr(tm, "embed_tokens_per_layer", None)
        if cfg.model.reinit_ple_row and ple is not None and sum_id < ple.weight.shape[0]:
            with torch.no_grad():
                ple.weight[sum_id] = mean_row(ple.weight).to(ple.weight.dtype)

    model = apply_lora(cfg, model)
    # Train LoRA weights in float32 (fp16 master weights would underflow in Adam).
    for _, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    if for_training and cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        model.config.use_cache = False
    lm = ScopeSumLM(cfg, model, sum_emb, sum_id)
    if cfg.variant.id == "S":
        from .data import load_tokenizer
        from .masks import separator_token_ids

        lm.separator_ids = separator_token_ids(load_tokenizer(cfg, path))
        print(f"[model] variant S keeps {len(lm.separator_ids):,} separator token ids")
    if sum_emb is not None:
        sum_emb.to(lm.device)
    n_train = sum(p.numel() for p in lm.trainable_parameters()[0]) + (sum_emb.vector.numel() if sum_emb else 0)
    print(f"[model] variant={cfg.variant.id} window={lm.window} trainable params={n_train:,}")
    return lm


# ---------------------------------------------------------------------------
# checkpoints (weights only; optimizer/RNG state lives in train.py)
# ---------------------------------------------------------------------------


def save_weights(lm: ScopeSumLM, out_dir: str) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)
    lm.model.save_pretrained(os.path.join(out_dir, "adapter"))
    if lm.sum_embedding is not None:
        torch.save({"vector": lm.sum_embedding.vector.detach().cpu(), "sum_id": lm.sum_id}, os.path.join(out_dir, "sum_embedding.pt"))


def load_weights(lm: ScopeSumLM, ckpt_dir: str) -> None:
    import os

    from peft import set_peft_model_state_dict

    adapter_dir = os.path.join(ckpt_dir, "adapter")
    st = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file

        state = load_file(st)
    else:
        state = torch.load(os.path.join(adapter_dir, "adapter_model.bin"), map_location="cpu")
    result = set_peft_model_state_dict(lm.model, state)
    unexpected = getattr(result, "unexpected_keys", None)
    if unexpected:
        print(f"[model] WARNING unexpected adapter keys: {unexpected[:5]} ...")
    se = os.path.join(ckpt_dir, "sum_embedding.pt")
    if lm.sum_embedding is not None:
        if not os.path.exists(se):
            raise FileNotFoundError(f"{se} missing: checkpoint is from a variant without <sum> tokens?")
        d = torch.load(se, map_location="cpu")
        if int(d["sum_id"]) != lm.sum_id:
            raise ValueError(f"checkpoint sum_id {d['sum_id']} != current {lm.sum_id}")
        with torch.no_grad():
            lm.sum_embedding.vector.copy_(d["vector"].to(lm.sum_embedding.vector.device))
    print(f"[model] loaded weights from {ckpt_dir}")


def load_for_eval(cfg: Config, path: str, sum_id: int, ckpt_dir: str | None) -> ScopeSumLM:
    lm = build_model(cfg, path, sum_id, for_training=False)
    if ckpt_dir:
        load_weights(lm, ckpt_dir)
    lm.eval()
    return lm


# ---------------------------------------------------------------------------
# helpers for the smoke test
# ---------------------------------------------------------------------------


@torch.no_grad()
def capture_layer_attention(lm: ScopeSumLM, batch: dict, layer_idx: int) -> torch.Tensor:
    """Output of ``layers[layer_idx].self_attn`` (before the residual add)."""
    store = {}
    attn = lm.text_model.layers[layer_idx].self_attn

    def hook(_m, _i, out):
        store["out"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()

    h = attn.register_forward_hook(hook)
    try:
        lm(batch, compute_loss=False)
    finally:
        h.remove()
    return store["out"]


def first_layer_of_type(lm: ScopeSumLM, layer_type: str) -> int:
    return list(lm.text_model.config.layer_types).index(layer_type)


def isfinite_report(t: torch.Tensor) -> dict:
    t = t.float()
    return {"nan": int(torch.isnan(t).sum()), "inf": int(torch.isinf(t).sum()), "max_abs": float(t.abs().max())}


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 50.0))
