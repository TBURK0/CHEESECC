import numpy as np

from cheesestack.ml.annotation import Annotation, BoxLabel
from cheesestack.ml.dataset import SampleRecord
from cheesestack.ml.versioning import (
    DatasetRegistry, SplitConfig, content_hash, split_for, split_records,
)


def records(n, seed=0):
    rng = np.random.default_rng(seed)
    return [SampleRecord(f"s{i}", float(i), image=rng.random((4, 4)), sensors={"drive": f"d{i // 10}"})
            for i in range(n)]


def test_content_hash_sensitive_to_labels_and_pixels():
    r = records(1)[0]
    h0 = content_hash(r)
    r.annotation = Annotation("s0", 4, 4, boxes=[BoxLabel("car", 0, 0, 2, 2)])
    h1 = content_hash(r)
    r.image = r.image + 0.1
    assert len({h0, h1, content_hash(r)}) == 3
    r.triggers = ["periodic"]
    assert content_hash(r) == content_hash(SampleRecord(r.sample_id, r.stamp, r.image, None, r.sensors,
                                                        r.annotation))


def test_registry_commit_load_diff(tmp_path):
    reg = DatasetRegistry(tmp_path / "reg")
    recs = records(10)
    v1 = reg.commit(recs, "v1")
    assert reg.commit(recs, "v1-again").snapshot_id == v1.snapshot_id  # content addressed
    recs2 = recs[1:] + [SampleRecord("new", 99.0, image=np.ones((2, 2)))]
    recs2[0].tags.append("relabelled")
    v2 = reg.commit(recs2, "v2", parent=v1.snapshot_id)
    d = reg.diff("v1", "v2")
    assert d.added == ["new"] and d.removed == ["s0"] and d.modified == ["s1"] and d.unchanged == 8
    assert reg.load("v2").parent == v1.snapshot_id
    assert len(reg.records("v2")) == 10
    assert reg.diff("v1", "v1").is_empty
    assert set(reg.list_versions()) == {v1.snapshot_id, v2.snapshot_id}


def test_splits_deterministic_and_proportional():
    ids = [f"sample_{i}" for i in range(5000)]
    a = [split_for(i) for i in ids]
    assert a == [split_for(i) for i in ids]
    frac = a.count("train") / len(a)
    assert 0.77 < frac < 0.83
    assert split_for("x", SplitConfig(salt="a")) in ("train", "val", "test")


def test_group_split_keeps_drives_together():
    cfg = SplitConfig(group_key="drive")
    out = split_records(records(100), cfg)
    for name, rs in out.items():
        for r in rs:
            assert all(split_for(o, cfg) == name for o in records(100) if o.sensors["drive"] == r.sensors["drive"])
