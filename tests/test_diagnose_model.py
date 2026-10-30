"""Diagnostics and probes end-to-end on a tiny random Gemma 4 (CPU, float32, eager)."""

import numpy as np
import pytest
from conftest import toy_tokenize
from test_capped import LONG
from test_model_hook import WINDOW, tiny_text_model

torch = pytest.importorskip("torch")


def tiny_lm(variant: str, **variant_kw):
    from scopesum.config import config_from_dict
    from scopesum.model import ScopeSumLM

    tm = tiny_text_model()
    cfg = config_from_dict(
        {"variant": {"id": variant, **variant_kw}, "model": {"dtype": "float32"}, "data": {"seq_len": 640}, "train": {"loss_chunk": 64}}
    )

    class Wrap(torch.nn.Module):  # the tiny model is a text decoder; give it an LM head like the real one
        def __init__(self):
            super().__init__()
            self.model = tm
            self.lm_head = torch.nn.Linear(tm.config.hidden_size, tm.config.vocab_size, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

    return ScopeSumLM(cfg, Wrap(), None, sum_id=6)


class Tok:
    bos_token_id, pad_token_id = 2, 0


def items():
    offs = toy_tokenize(LONG)
    rng = np.random.default_rng(0)
    w = {"text": LONG, "offsets": [list(o) for o in offs], "raw_ids": rng.integers(10, 120, len(offs)).tolist()}
    return [{"window": w, "targets": list(range(len(offs) // 2, len(offs), 7))}]


@pytest.mark.parametrize("variant,kw,chunk", [("A", {}, None), ("C", {}, None), ("F", {}, None), ("G", {"chunk_size": 24}, 24), ("D", {"w_small": 2}, None)])
def test_diagnose_variant_runs(variant, kw, chunk):
    from scopesum.diagnose import diagnose_variant

    lm = tiny_lm(variant, **kw)
    assert lm.window == WINDOW and lm.scope_span == WINDOW
    res = diagnose_variant(lm, items(), Tok(), chunk, n_attention=1)
    assert res["loss"]["normal"]["all"]["n"] > 0 and res["loss"]["normal"]["long_range"]["n"] > 0
    assert res["attention_missing_layers"] == 0, "eager attention must return weights"
    assert set(res["attention"]) == {"sliding_attention", "full_attention"}
    for a in res["attention"].values():
        assert 0.0 <= a["sum"] <= 1.0 + 1e-5 and 0.0 <= a["bos"] <= 1.0 + 1e-5
    if variant == "A":
        assert list(res["loss"]) == ["normal"]
        return
    assert set(res["delta"]) == {"hide_sliding", "hide_all"}
    assert res["attention"]["sliding_attention"]["far_raw"] < 1e-6 or variant == "D", "B/C/F/G local layers never see far raw tokens"
    assert res["attention"]["sliding_attention"]["sum"] > 0
    assert res["delta"]["hide_all"]["all"] != 0.0, "hiding sums changes the output of a random model"


def test_probe_controls():
    pytest.importorskip("sklearn")
    import os
    import tempfile

    from scopesum.probe import collect_features, train_probes

    class FileTok(Tok):
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            offs = toy_tokenize(text)
            return {"input_ids": [10 + (hash(text[a:b]) % 100) for a, b in offs], "offset_mapping": offs}

    lm = tiny_lm("F")
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i in range(12):
            p = os.path.join(d, f"f{i}.py")
            with open(p, "w") as f:
                f.write(LONG.replace("Big", f"Big{i}"))
            files.append(p)
        col = collect_features(lm, FileTok(), files, None, max_files=None, split_chunk=24)
    n = len(col["split"])
    assert n > 0 and col["features"][-1].shape[0] == n == col["raw_features"][-1].shape[0]
    assert col["multi_chunk"].any(), "LONG has functions crossing a 24-token boundary"
    col["split"] = np.array(["train"] * (n - n // 3) + ["test"] * (n // 3))
    rows = train_probes(col)
    assert {r["feature"] for r in rows} == {"sum", "raw_mean"}
    assert {r["subset"] for r in rows} >= {"all"}


def test_baseline_honours_window_override():
    from scopesum.sum_insertion import insert_sums

    ex = insert_sums(list(range(10, 22)), None, bos_id=2, sum_id=6)
    default, narrow = tiny_lm("A"), tiny_lm("A", window=2)
    narrow.load_state_dict(default.state_dict())
    assert default.library_masks and not narrow.library_masks
    b_default, b_narrow = default.make_batch([ex]), narrow.make_batch([ex])
    assert "sliding_mask" not in b_default and b_narrow["sliding_mask"][0, -1].sum() == 2
    with torch.no_grad():
        assert not torch.allclose(default(b_default, compute_loss=False)["hidden"], narrow(b_narrow, compute_loss=False)["hidden"])


@pytest.mark.parametrize("variant,kw", [("A", {}), ("F", {}), ("G", {"chunk_size": 24})])
def test_restrict_full_gives_full_layers_the_local_mask(variant, kw):
    lm = tiny_lm(variant, restrict_full=True, **kw)
    assert not lm.library_masks
    ex = lm.make_batch([build_item(lm)])
    assert torch.equal(ex["full_mask"], ex["sliding_mask"])
    if variant != "A":
        res = __import__("scopesum.diagnose", fromlist=["x"]).diagnose_variant(lm, items(), Tok(), kw.get("chunk_size"), n_attention=1)
        assert res["attention"]["full_attention"]["far_raw"] < 1e-6, "full layers no longer see far raw tokens"
        assert res["attention"]["full_attention"]["sum"] > 0


def build_item(lm):
    from scopesum.data import build_example

    it = items()[0]
    return build_example(it["window"], lm.mode, Tok.bos_token_id, lm.sum_id, Tok.pad_token_id, 640, chunk_size=24, scope_span=lm.scope_span)


def test_keep_signatures_with_restrict_full():
    lm = tiny_lm("F", restrict_full=True, keep_signatures=True)
    batch = lm.make_batch([build_item(lm)])
    assert torch.equal(batch["full_mask"], batch["sliding_mask"])
    res = __import__("scopesum.diagnose", fromlist=["x"]).diagnose_variant(lm, items(), Tok(), None, n_attention=1)
    assert res["attention"]["full_attention"]["far_raw"] > 0, "far signature tokens are attended in full layers"
    assert set(res["delta"]) == {"hide_sliding", "hide_all"}


def test_separator_variant_runs():
    lm = tiny_lm("S", restrict_full=True)
    assert not lm.cfg.variant.uses_sums and lm.mode == "none"
    lm.separator_ids = np.arange(10, 40)  # a band of ids in the toy vocabulary
    batch = lm.make_batch([build_item(lm)])
    assert torch.equal(batch["full_mask"], batch["sliding_mask"])
    res = __import__("scopesum.diagnose", fromlist=["x"]).diagnose_variant(lm, items(), Tok(), None, n_attention=1)
    assert list(res["loss"]) == ["normal"], "no <sum> to hide"
    assert res["attention"]["full_attention"]["far_raw"] > 0, "far separators are attended"
