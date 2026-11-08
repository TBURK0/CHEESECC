import json

import numpy as np

from cheesestack.ml.annotation import (
    Annotation, BoxLabel, ColorBlobPreLabeler, LabelSchema, MaskLabel, PolylineLabel,
    auto_label, from_coco, from_yolo, mask_to_rle, rle_to_mask, to_coco, to_yolo,
)


def make_ann():
    mask = np.zeros((20, 30), bool)
    mask[10:, 5:25] = True
    return Annotation(
        "img0", 30, 20,
        boxes=[BoxLabel("car", 2, 3, 10, 8, 0.9), BoxLabel("traffic_light", 20, 1, 4, 6)],
        polylines=[PolylineLabel("lane_solid", [[5, 19], [10, 10], [12, 5]])],
        masks=[MaskLabel("drivable", mask)], tags=["day"],
    )


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(5):
        m = rng.random((7, 9)) > 0.5
        assert np.array_equal(rle_to_mask(mask_to_rle(m)), m)
    m = np.ones((3, 3), bool)
    assert mask_to_rle(m)["counts"][0] == 0


def test_schema_validation():
    schema = LabelSchema()
    assert schema.validate(make_ann()) == []
    bad = make_ann()
    bad.boxes.append(BoxLabel("dragon", 25, 15, 10, 10, 1.5))
    bad.polylines.append(PolylineLabel("lane_solid", [[1, 1]]))
    errs = schema.validate(bad)
    assert any("unknown class" in e for e in errs)
    assert any("outside" in e for e in errs)
    assert any("score" in e for e in errs)
    assert any("points" in e for e in errs)


def test_coco_roundtrip():
    ann = make_ann()
    coco = json.loads(json.dumps(to_coco([ann], LabelSchema())))
    back = from_coco(coco)[0]
    assert back.image_id == "img0" and back.tags == ["day"]
    assert [(b.cls, b.u, b.w) for b in back.boxes] == [(b.cls, b.u, b.w) for b in ann.boxes]
    assert np.allclose(back.polylines[0].points, ann.polylines[0].points)
    assert np.array_equal(back.masks[0].mask, ann.masks[0].mask)


def test_yolo_roundtrip():
    ann = make_ann()
    classes = ["car", "traffic_light"]
    txt = to_yolo(ann, classes)
    assert len(txt.strip().splitlines()) == 2
    back = from_yolo(txt, classes, 30, 20, "img0")
    for a, b in zip(ann.boxes, back.boxes):
        assert a.cls == b.cls
        assert np.allclose([a.u, a.v, a.w, a.h], [b.u, b.v, b.w, b.h], atol=1e-4)


def test_color_blob_prelabeler():
    img = np.zeros((30, 30, 3), np.float32)
    img[5:9, 5:8] = [1.0, 0.05, 0.05]  # red blob
    img[20:25, 18:22] = [0.05, 0.9, 0.1]  # green blob
    ann = auto_label({"x": img}, ColorBlobPreLabeler())[0]
    states = sorted(b.attributes["state"] for b in ann.boxes)
    assert states == ["green", "red"]
    red = next(b for b in ann.boxes if b.attributes["state"] == "red")
    assert (red.u, red.v, red.w, red.h) == (5, 5, 3, 4)
    assert LabelSchema().validate(ann) == []
