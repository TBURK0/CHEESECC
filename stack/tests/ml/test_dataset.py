import numpy as np

from cheesestack.ml.annotation import Annotation, BoxLabel
from cheesestack.ml.dataset import (
    CollectorConfig, DatasetCollector, DisagreementTrigger, LowConfidenceTrigger, PeriodicTrigger,
    RareClassTrigger, SampleRecord, TriggerContext, read_manifest, write_manifest,
)


def rec(i, t=None, **kw):
    return SampleRecord(f"s{i}", float(i if t is None else t), **kw)


def test_manifest_roundtrip(tmp_path):
    ann = Annotation("s0", 4, 4, boxes=[BoxLabel("car", 0, 0, 2, 2)])
    r = rec(0, sensors={"speed_mps": 3.2}, annotation=ann, tags=["night"])
    p = tmp_path / "m.jsonl"
    write_manifest(p, [r, rec(1)])
    out = read_manifest(p)
    assert len(out) == 2 and out[0].sensors["speed_mps"] == 3.2
    assert out[0].annotation.boxes[0].cls == "car"


def test_low_confidence_trigger():
    t = LowConfidenceTrigger(0.3, 0.6)
    assert t(rec(0), TriggerContext([BoxLabel("car", 0, 0, 5, 5, 0.45)]))
    assert not t(rec(0), TriggerContext([BoxLabel("car", 0, 0, 5, 5, 0.95)]))


def test_disagreement_trigger():
    t = DisagreementTrigger(iou_threshold=0.5)
    a = [BoxLabel("car", 0, 0, 10, 10, 0.9)]
    assert not t(rec(0), TriggerContext(a, [BoxLabel("car", 1, 1, 10, 10, 0.9)]))
    assert t(rec(0), TriggerContext(a, [BoxLabel("truck", 1, 1, 10, 10, 0.9)]))
    assert t(rec(0), TriggerContext(a, []))
    assert not t(rec(0), TriggerContext(a, None))


def test_rare_class_trigger():
    t = RareClassTrigger(max_frequency=0.05, min_observations=20, always_rare=("animal",))
    car = TriggerContext([BoxLabel("car", 0, 0, 5, 5)])
    fired = [t(rec(i), car) for i in range(30)]
    assert not any(fired)
    assert t(rec(31), TriggerContext([BoxLabel("bus", 0, 0, 5, 5)]))
    assert t(rec(32), TriggerContext([BoxLabel("animal", 0, 0, 5, 5)]))


def test_collector_rate_limit_budget_and_images(tmp_path):
    cfg = CollectorConfig(output_dir=str(tmp_path / "ds"), min_interval_s=1.0, max_samples=3)
    col = DatasetCollector([PeriodicTrigger(0.0)], cfg)
    img = np.zeros((4, 4, 3), np.uint8)
    kept = [col.offer(rec(i, t=i * 0.5, image=img)) for i in range(20)]
    assert sum(kept) == 3
    assert kept[:3] == [True, False, True]
    records = col.records()
    assert [r.sample_id for r in records] == ["s0", "s2", "s4"]
    assert records[0].triggers == ["periodic"]
    assert records[0].load_image(cfg.output_dir).shape == (4, 4, 3)
