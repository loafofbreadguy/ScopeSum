"""End-to-end check of the mask hook on a tiny random Gemma 4 (CPU, float32).

Skipped when torch/transformers are missing or the tiny config cannot be
built (e.g. a transformers version with different Gemma 4 config fields).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scopesum.masks import MaskMeta, build_sliding_mask, full_causal_mask  # noqa: E402
from scopesum.segmenter import fixed_segments  # noqa: E402
from scopesum.sum_insertion import insert_sums  # noqa: E402

WINDOW = 4


def tiny_text_model():
    try:
        from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig

        cfg = Gemma4TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            sliding_window=WINDOW,
            layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=128,
            max_position_embeddings=256,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
        )
        cfg._attn_implementation = "eager"
        torch.manual_seed(0)
        model = Gemma4ForCausalLM(cfg).eval()
    except Exception as e:  # pragma: no cover - depends on library version
        pytest.skip(f"cannot build a tiny Gemma 4 model: {e!r}")
    from scopesum.model import find_text_model

    return find_text_model(model)


def as_dict(meta, variant):
    from scopesum.model import additive_mask

    sliding = torch.from_numpy(build_sliding_mask(variant, meta, window=WINDOW, w_small=2))[None]
    full = torch.from_numpy(full_causal_mask(meta))[None]
    return {"full_attention": additive_mask(full, torch.float32), "sliding_attention": additive_mask(sliding, torch.float32)}


def run(tm, ids, mask):
    ids_t = torch.tensor([ids])
    pos = torch.arange(len(ids))[None]
    with torch.no_grad():
        return tm(input_ids=ids_t, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state


def test_explicit_baseline_masks_match_library_default():
    tm = tiny_text_model()
    ex = insert_sums(list(range(10, 22)), None, bos_id=2, sum_id=6)
    meta = MaskMeta.from_example(ex)
    default = run(tm, ex.input_ids, torch.ones(1, ex.length, dtype=torch.long))
    ours = run(tm, ex.input_ids, as_dict(meta, "A"))
    assert torch.allclose(default, ours, atol=1e-5), (default - ours).abs().max()


def test_hidden_token_does_not_reach_first_local_layer():
    tm = tiny_text_model()
    ex = insert_sums(list(range(10, 26)), fixed_segments(16, 3), bos_id=2, sum_id=6)
    meta = MaskMeta.from_example(ex)
    mask_np = build_sliding_mask("B", meta, window=WINDOW)
    kind = np.array(ex.kind)
    q = int(np.where(kind == 1)[0][-1])
    hidden = [j for j in range(1, q) if kind[j] == 1 and not mask_np[q, j]]
    visible = [j for j in range(1, q) if kind[j] == 1 and mask_np[q, j] and j != q]
    assert hidden and visible

    store = {}
    h = tm.layers[0].self_attn.register_forward_hook(lambda m, i, o: store.__setitem__("o", o[0][0, q].clone()))
    try:
        mask = as_dict(meta, "B")
        run(tm, ex.input_ids, mask)
        ref = store["o"]
        ids = list(ex.input_ids)
        ids[hidden[0]] = 99
        run(tm, ids, mask)
        assert torch.allclose(store["o"], ref, atol=1e-6)
        ids = list(ex.input_ids)
        ids[visible[0]] = 99
        run(tm, ids, mask)
        assert not torch.allclose(store["o"], ref, atol=1e-6)
    finally:
        h.remove()
