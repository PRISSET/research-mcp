import json

import pytest

from kb.packs import build, client
from kb.packs import format as fmt
from kb.packs import recipes as rcp
from kb.store import Database

KOHYA = {
    "ss_sd_scripts_commit_hash": "7db4222", "ss_network_module": "networks.lora_flux", "ss_network_dim": "16",
    "ss_network_alpha": "8.0", "ss_learning_rate": "0.0004", "ss_unet_lr": "None", "ss_text_encoder_lr": "0.0",
    "ss_optimizer": "bitsandbytes.optim.adamw.AdamW8bit(weight_decay=0.01)", "ss_lr_scheduler": "cosine",
    "ss_max_train_steps": "1600", "ss_num_epochs": "16", "ss_gradient_accumulation_steps": "2",
    "ss_datasets": json.dumps([{"batch_size_per_device": 2, "resolution": [1024, 1024],
                                "subsets": [{"img_count": 20, "num_repeats": 5}, {"img_count": 10, "num_repeats": 2}]}]),
    "ss_mixed_precision": "bf16", "ss_noise_offset": "0.05", "ss_timestep_sampling": "sigmoid",
    "ss_training_started_at": "1735689600.0", "modelspec.architecture": "flux-1-dev/lora",
}


def test_kohya_header_gives_the_whole_recipe():
    head = {"__metadata__": KOHYA, "lora_unet_x.lora_down.weight": {"shape": [16, 3072], "dtype": "BF16"}}
    r = rcp.from_header(head)
    assert (r["trainer"], r["method"], r["rank"], r["alpha"], r["lr"]) == ("sd-scripts", "lora", 16, 8.0, 0.0004)
    assert (r["optimizer"], r["scheduler"], r["steps"], r["epochs"]) == ("AdamW8bit", "cosine", 1600, 16)
    assert (r["batch"], r["images"], r["repeats"], r["resolution"]) == (4, 30, 4.0, 1024)
    assert r["extra"]["text_encoder"] == "frozen" and r["extra"]["noise_offset"] == 0.05
    assert r["date"] == "2025-01-01" and rcp.family(r["extra"]["base_hint"]) == "flux.1-dev"


def test_other_trainers_and_rank_from_shapes():
    ai = {"__metadata__": {"software": '{"name": "ai-toolkit", "version": "0.13.21"}',
                           "training_info": '{"step": 3000, "epoch": 6}'},
          "a.lora_A.weight": {"shape": [32, 3072]}, "b.lora_A.weight": {"shape": [32, 3072]}}
    r = rcp.from_header(ai)
    assert (r["trainer"], r["steps"], r["epochs"], r["rank"]) == ("ai-toolkit 0.13.21", 3000, 6, 32)
    assert rcp.useful(r) and not rcp.useful(rcp.from_header({"x.lora_A.weight": {"shape": [8, 64]}}))


def test_trainer_card_and_axolotl_config():
    card = """### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999)
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3

### Framework versions
- PEFT 0.15.2
- Transformers 4.57.0
- TRL 0.24.0
"""
    r = rcp.from_card(card)
    assert (r["lr"], r["batch"], r["optimizer"], r["scheduler"], r["epochs"]) == (2e-05, 32, "ADAMW_TORCH", "cosine", 3)
    assert r["method"] == "lora" and r["trainer"] == "trl" and r["extra"]["warmup_ratio"] == 0.1
    ax = "axolotl version: 0.12\nbase_model: Qwen/Qwen3-8B\nadapter: qlora\nlora_r: 32\nlora_alpha: 64\n" \
         "learning_rate: 0.0002\nmicro_batch_size: 2\ngradient_accumulation_steps: 4\nsequence_len: 4096\n"
    r = rcp.from_card(ax)
    assert (r["method"], r["rank"], r["alpha"], r["batch"], r["trainer"]) == ("qlora", 32, 64.0, 8, "axolotl")
    assert r["extra"]["seq_len"] == 4096 and rcp.from_card("no settings here") == {}


def test_families_and_tasks():
    assert rcp.family("black-forest-labs/FLUX.1-dev") == "flux.1-dev"
    assert rcp.family("Flux.1 D") == "flux.1-dev" and rcp.family("SDXL 1.0") == "sdxl"
    assert rcp.family("Wan Video 2.2 T2V-A14B") == "wan2.2" and rcp.family("Qwen/Qwen-Image-2.1") == "qwen-image"
    assert rcp.family("Qwen/Qwen3.5-4B-Base") == "qwen3.5" and rcp.family("meta-llama/Llama-3.1-8B") == "llama-3.1"
    assert rcp.task_of(["character", "woman"], "Jane LoRA") == "character"
    assert rcp.task_of([], "Hyper-FLUX 8steps") == "speed"
    assert rcp.task_of(["dpo"], "x", llm=True) == "dpo"


def _recipe(i, **kw):
    base = {"type": "recipe", "id": f"recipe:hf:a/{i}", "family": "flux.1-dev", "author": f"u{i % 4}",
            "lr": 1e-4 * (1 + i % 3), "rank": 16, "alpha": 16.0, "steps": 2000, "batch": 1, "images": 25,
            "optimizer": "AdamW8bit", "popularity": 1000 * i, "extra": {"noise_offset": 0.1}}
    return {**base, **kw}


def test_recipes_summary(tmp_path):
    db = Database(tmp_path / "kb.sqlite")
    for i in range(1, 9):
        rcp.store(db, _recipe(i))
    rcp.store(db, _recipe(99, family="sdxl", lr=1e-3))
    db.commit()
    out = rcp.recipes(db, "FLUX.1-dev", images=30)
    fam = out["families"][0]
    assert out["matched_family"] == "flux.1-dev" and fam["family"] == "flux.1-dev" and fam["recipes"] == 8
    assert fam["authors"] == 4 and fam["settings"]["rank"]["median"] == 16
    assert fam["settings"]["passes_per_image"]["median"] == 80
    assert fam["optimizer"][0] == {"value": "AdamW8bit", "share": 1.0, "n": 8}
    assert fam["tricks"]["noise_offset"]["share"] == 1.0 and fam["examples"][0]["popularity"] == 8000
    assert rcp.recipes(db, "nothing-like-this")["families_known"]["sdxl"] == 1
    db.close()


def _doc(i, text="Use a learning rate of 1e-4 for rank 16."):
    return {"type": "doc", "id": f"gh:org/trainer/docs/{i}.md", "src": "docs:org/trainer", "rev": f"sha{i}",
            "paper": {"title": f"org/trainer/docs/{i}.md", "abstract": "", "authors": [], "url": f"https://x/{i}",
                      "source": "github", "kind": "web", "extra": {"repo": "org/trainer"}},
            "chunks": [["Training", "text", f"{text} Page {i}."]]}


@pytest.fixture
def local(tmp_path, monkeypatch):
    """A published folder and a user base, without network or the user's notebook."""
    pub = tmp_path / "pub"
    monkeypatch.setattr(client, "Database", lambda: Database(tmp_path / "user.sqlite"))
    monkeypatch.setattr(client, "_cited", lambda: {"gh:org/trainer/docs/2.md"})
    monkeypatch.setattr(client, "_postprocess", lambda db, ids: f"{len(ids)} docs")
    monkeypatch.setattr(build, "load_def", lambda name: {"name": name, "title": "Test"})
    state: dict = {}
    monkeypatch.setattr(build, "collect", lambda spec, prev: ({k: dict(v) for k, v in state.items()}, set()))

    def publish(records: dict) -> dict:
        state.clear()
        state.update(records)
        return build.build(["t"], pub, str(pub))

    return pub, publish, tmp_path


def test_install_then_update_downloads_only_changes(local):
    pub, publish, tmp = local
    recs = {r["id"]: r for r in [_doc(1), _doc(2), _doc(3), _recipe(1)]}
    assert publish(recs)["t"]["version"] == 1
    r = client.update(["t"], base=str(pub), install=True)["t"]
    assert (r["downloaded"], r["written"], r["to"]) == ("snapshot", 4, 1)

    recs2 = dict(recs)
    recs2[_doc(1)["id"]] = _doc(1, "Use 2e-4 instead.")    # changed
    del recs2[_doc(3)["id"]]                               # removed
    recs2[_doc(4)["id"]] = _doc(4)                         # new
    rep = publish(recs2)["t"]
    assert rep["version"] == 2 and rep["change"]["puts"] == 2 and rep["change"]["dels"] == 1
    r = client.update(base=str(pub))["t"]
    assert (r["downloaded"], r["written"], r["deleted"]) == ("changes", 2, 1)
    assert r["files"] == ["t.ch-00002.jsonl.xz"]

    db = Database(tmp / "user.sqlite")
    texts = [x[0] for x in db.conn.execute("SELECT text FROM chunks WHERE paper_id = 'gh:org/trainer/docs/1.md'")]
    assert texts == ["Use 2e-4 instead. Page 1."]
    assert not db.conn.execute("SELECT 1 FROM papers WHERE id = 'gh:org/trainer/docs/3.md'").fetchone()
    assert db.conn.execute("SELECT version FROM packs WHERE name = 't'").fetchone()[0] == 2
    db.close()
    assert publish(recs2)["t"]["changed"] is False          # nothing new: no version
    assert client.update(base=str(pub))["t"]["status"] == "up to date"


def test_far_behind_gets_snapshot_and_stale_items_go(local, monkeypatch):
    pub, publish, tmp = local
    publish({r["id"]: r for r in [_doc(i) for i in range(1, 5)]})
    client.update(["t"], base=str(pub), install=True)
    monkeypatch.setattr(build, "SNAPSHOT_EVERY", 2)
    publish({r["id"]: r for r in [_doc(1, "v2"), *(_doc(i) for i in range(2, 5))]})
    publish({r["id"]: r for r in [_doc(1, "v3"), _doc(2), _doc(3), _doc(5)]})   # snapshot at v3, 4 removed
    entry = fmt.load_manifest(str(pub))["packs"]["t"]
    assert entry["snapshot"]["version"] == 3
    assert fmt.files_to_apply(entry, 0)[0] == "snapshot"
    big = {**entry, "snapshot": {**entry["snapshot"], "bytes": 10**9}}
    assert fmt.files_to_apply(big, 1) == ("changes", entry["changes"])     # fewer bytes than the snapshot
    assert fmt.files_to_apply({**big, "changes": entry["changes"][1:]}, 1)[0] == "snapshot"   # v2 not kept

    pub_entry = json.loads((pub / "manifest.json").read_text())
    pub_entry["packs"]["t"]["changes"] = pub_entry["packs"]["t"]["changes"][1:]
    (pub / "manifest.json").write_text(json.dumps(pub_entry))
    r = client.update(base=str(pub))["t"]
    assert r["downloaded"] == "snapshot" and r["deleted"] == 1              # doc 4 is gone from the snapshot
    db = Database(tmp / "user.sqlite")
    ids = sorted(x[0] for x in db.conn.execute("SELECT item_id FROM pack_items"))
    text = db.conn.execute("SELECT text FROM chunks WHERE paper_id = 'gh:org/trainer/docs/1.md'").fetchone()[0]
    db.close()
    assert ids == [f"gh:org/trainer/docs/{i}.md" for i in (1, 2, 3, 5)] and text.startswith("v3")


def test_removed_pack_keeps_cited_docs_and_build_refuses_mass_delete(local):
    pub, publish, tmp = local
    publish({r["id"]: r for r in [_doc(i) for i in range(1, 7)]})
    client.update(["t"], base=str(pub), install=True)
    assert client.remove("t")["removed"] == 6
    db = Database(tmp / "user.sqlite")
    left = [r[0] for r in db.conn.execute("SELECT id FROM papers")]
    db.close()
    assert left == ["gh:org/trainer/docs/2.md"]                             # the notebook cites it
    with pytest.raises(RuntimeError, match="would delete"):
        publish({_doc(1)["id"]: _doc(1)})


def test_failed_source_keeps_its_records(monkeypatch):
    prev = {"gh:a/b/docs/x.md": {**_doc(1), "src": "docs:a/b"}, "recipe:hf:z": {**_recipe(1), "src": "recipes:hf"}}

    def fake_recipes(spec, cache, prev, out, failed):
        failed.add("recipes:hf")
        return []

    monkeypatch.setattr(build, "collect_recipes", fake_recipes)
    for name in ("collect_docs", "collect_issues"):
        monkeypatch.setattr(build, name, lambda spec, prev, db, out, failed: None)
    monkeypatch.setattr(build, "collect_cards", lambda *a: None)
    monkeypatch.setattr(build, "_cache_path", lambda name: __import__("pathlib").Path(__import__("os").devnull))
    out, failed = build.collect({"name": "t"}, prev)
    assert "recipe:hf:z" in out and "gh:a/b/docs/x.md" not in out and failed == {"recipes:hf"}


def test_new_downloads_are_budgeted_and_rate_limits_keep_old_records():
    models = [{"id": f"m{i}"} for i in range(6)]
    prev = {"r:m4": {"id": "r:m4", "old": True}, "r:m1": {"id": "r:m1", "old": True}}
    known = {"m0"}

    def recipe(m):
        if m["id"] == "m1":
            raise rcp.Retry("429")
        return {"id": f"r:{m['id']}"}

    out: dict = {}
    got = build._read_new(models, recipe, lambda m: f"r:{m['id']}", lambda m: m["id"] in known, 2, 1, prev, out, "s")
    # m0 from cache, m1/m2 are the budget (m1 rate-limited keeps its old record), m3-m5 wait; m4 keeps its old one
    assert sorted(out) == ["r:m0", "r:m1", "r:m2", "r:m4"] and out["r:m1"]["old"] and out["r:m4"]["old"]
    assert [m["id"] for m in got] == ["m0", "m2"]
