import numpy as np

from cheesestack.ml.annotation import Annotation, BoxLabel, MaskLabel, PolylineLabel
from cheesestack.ml.augmentation import (
    BrightnessContrast, Compose, Cutout, GaussianBlur, GaussianNoise, HorizontalFlip, RandomCrop,
    gaussian_blur,
)


def scene():
    img = np.zeros((40, 60, 3), np.float32)
    img[10:20, 5:15] = 1.0  # object pixels match the box
    mask = np.zeros((40, 60), bool)
    mask[10:20, 5:15] = True
    ann = Annotation("a", 60, 40, boxes=[BoxLabel("turn_left", 5, 10, 10, 10)],
                     polylines=[PolylineLabel("lane_solid", [[10, 39], [20, 0]])],
                     masks=[MaskLabel("drivable", mask)])
    return img, ann


def test_hflip_moves_pixels_and_labels_together():
    img, ann = scene()
    out, a = HorizontalFlip(p=1.0, class_swap={"turn_left": "turn_right"})(img, ann, np.random.default_rng(0))
    b = a.boxes[0]
    assert (b.u, b.w, b.cls) == (45, 10, "turn_right")
    assert out[10:20, 45:55].min() == 1.0 and out[10:20, 5:15].max() == 0.0
    assert np.array_equal(a.masks[0].mask, out[..., 0] > 0.5)
    assert np.allclose(a.polylines[0].points[:, 0], [50, 40])
    assert ann.boxes[0].u == 5  # input not mutated


def test_random_crop_keeps_consistency():
    img, ann = scene()
    rng = np.random.default_rng(3)
    for _ in range(20):
        out, a = RandomCrop(30, 30)(img, ann, rng)
        assert out.shape == (30, 30, 3) and (a.width, a.height) == (30, 30)
        for b in a.boxes:
            assert b.u >= 0 and b.v >= 0 and b.u + b.w <= 30 and b.v + b.h <= 30
            u, v = int(b.u), int(b.v)
            assert out[v:v + int(b.h), u:u + int(b.w)].min() == 1.0
        assert a.masks[0].mask.shape == (30, 30)


def test_photometric_preserve_dtype_and_range():
    rng = np.random.default_rng(0)
    img8 = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
    for t in (BrightnessContrast(), GaussianNoise(0.1), GaussianBlur(p=1.0)):
        out, _ = t(img8, None, rng)
        assert out.dtype == np.uint8 and out.shape == img8.shape
    f = rng.random((16, 16)).astype(np.float32)
    out, _ = GaussianNoise(0.5)(f, None, rng)
    assert out.min() >= 0 and out.max() <= 1


def test_blur_preserves_mean_and_smooths():
    img = np.zeros((21, 21), np.float32)
    img[10, 10] = 1.0
    b = gaussian_blur(img, 1.0)
    assert abs(b.sum() - 1.0) < 1e-5 and b[10, 10] < 0.5 and b[10, 10] == b.max()


def test_cutout_drops_occluded_boxes():
    img, ann = scene()
    out, a = Cutout(n_holes=1, size=200, p=1.0)(img, ann, np.random.default_rng(0))
    assert out.max() == 0.0 and a.boxes == []


def test_compose_seeded_reproducible():
    img, ann = scene()
    pipe = lambda: Compose([HorizontalFlip(), RandomCrop(32, 32), BrightnessContrast(), GaussianNoise()], seed=7)
    o1, a1 = pipe()(img, ann)
    o2, a2 = pipe()(img, ann)
    assert np.array_equal(o1, o2) and a1.to_dict() == a2.to_dict()
