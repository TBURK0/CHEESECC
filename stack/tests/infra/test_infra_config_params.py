import sys
import threading
from dataclasses import dataclass, field

import pytest

from cheesestack.infra import ConfigError, ConfigLoader, ParamServer, from_dict, load_config, to_dict
from cheesestack.infra.config import coerce, save


@dataclass
class LateralCfg:
    kp: float = 0.5
    kd: float = 0.1


@dataclass
class StackCfg:
    name: str = "cheese"
    rate_hz: int = 20
    enabled: bool = True
    gains: list[float] = field(default_factory=lambda: [1.0, 2.0])
    lateral: LateralCfg = field(default_factory=LateralCfg)

    def validate(self):
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")


def test_layering_order(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"rate_hz": 30, "lateral": {"kp": 0.7}}')
    env = {"CHEESE_RATE_HZ": "40", "CHEESE_LATERAL__KD": "0.3", "CHEESE_ENABLED": "false", "OTHER": "x"}
    loader = ConfigLoader(StackCfg)
    cfg = loader.load(f, environ=env, overrides={"lateral.kp": 0.9})
    assert cfg.rate_hz == 40 and cfg.enabled is False
    assert cfg.lateral == LateralCfg(kp=0.9, kd=0.3)
    assert cfg.name == "cheese"
    assert loader.sources == ["defaults", f"file:{f}", "env", "overrides"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib needs 3.11")
def test_toml_and_roundtrip(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('name = "x"\ngains = [3.0]\n[lateral]\nkp = 1.5\n')
    cfg = load_config(StackCfg, f, environ={})
    assert cfg.name == "x" and cfg.gains == [3.0] and cfg.lateral.kp == 1.5
    out = tmp_path / "o.json"
    save(cfg, out)
    assert load_config(StackCfg, out, environ={}) == cfg
    assert from_dict(StackCfg, to_dict(cfg)) == cfg


def test_validation_and_unknown_keys(tmp_path):
    with pytest.raises(ConfigError):
        load_config(StackCfg, environ={}, overrides={"rate_hz": 0})
    with pytest.raises(ConfigError):
        load_config(StackCfg, environ={}, overrides={"bogus": 1})
    with pytest.raises(ConfigError):
        load_config(StackCfg, environ={"CHEESE_RATE_HZ": "abc"})


def test_coerce():
    assert coerce("yes", bool) is True
    assert coerce("[1, 2]", list[int]) == [1, 2]
    with pytest.raises(ConfigError):
        coerce(1.5, int)


def test_params_typed_namespaced_callbacks():
    ps = ParamServer()
    ps.declare("/control/lateral/kp", 0.5, min=0.0, max=5.0)
    ps.declare("control/lateral/kd", 0.1)
    ps.declare("/planning/horizon", 50)
    seen = []
    ps.on_change("/control", lambda p, o, n: seen.append((p, o, n)))
    assert ps.set("/control/lateral/kp", 1) is True  # int -> float ok
    assert ps.get("/control/lateral/kp") == 1.0
    assert ps.set("/control/lateral/kp", 1.0) is False  # unchanged
    ps.set("/planning/horizon", 60)  # outside namespace
    assert seen == [("/control/lateral/kp", 0.5, 1.0)]
    with pytest.raises(TypeError):
        ps.set("/control/lateral/kp", "high")
    with pytest.raises(ValueError):
        ps.set("/control/lateral/kp", 9.0)
    assert ps.tree("/control") == {"lateral": {"kp": 1.0, "kd": 0.1}}
    assert list(ps.list("/control")) == ["/control/lateral/kd", "/control/lateral/kp"]
    ns = ps.namespace("/control").namespace("lateral")
    assert ns.get("kd") == 0.1


def test_params_snapshot_restore_and_errors():
    ps = ParamServer()
    ps.declare("/a", 1)
    snap = ps.snapshot()
    ps.set("/a", 2)
    ps.set("/b", "new")
    changes = []
    ps.on_change("/", lambda p, o, n: changes.append(p))
    ps.on_change("/a", lambda p, o, n: 1 / 0)
    assert ps.restore(snap) == ["/a", "/b"]
    assert ps.snapshot() == {"/a": 1}
    assert sorted(changes) == ["/a", "/b"]
    assert len(ps.callback_errors) == 1


def test_params_thread_safe():
    ps = ParamServer()
    ps.declare("/n", 0)
    lock = threading.Lock()

    def work():
        for _ in range(500):
            with lock:  # read-modify-write needs external lock; individual ops are atomic
                ps.set("/n", ps.get("/n") + 1)
            ps.list("/")
    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert ps.get("/n") == 2000
