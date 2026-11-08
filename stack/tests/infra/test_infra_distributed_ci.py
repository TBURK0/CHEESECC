import pytest

from cheesestack.infra import MapReduceError, TaskQueue, TaskQueueConfig, check_imports, map_reduce
from cheesestack.infra.ci import CIConfig, discover_modules, main


class Flaky:
    def __init__(self, fails):
        self.fails = fails
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        if self.calls <= self.fails:
            raise IOError("transient")
        return x * 2


def test_task_queue_retries():
    sleeps = []
    with TaskQueue(TaskQueueConfig(max_workers=1, max_retries=2, backoff_s=0.1), sleep=sleeps.append) as q:
        ok = q.submit(Flaky(2), 5).result()
        bad = q.submit(Flaky(5), 5).result()
        st = q.stats()
    assert ok.ok and ok.value == 10 and ok.attempts == 3
    assert not bad.ok and isinstance(bad.error, IOError) and bad.attempts == 3
    assert sleeps == [0.1, 0.2, 0.1, 0.2]
    assert st == {"submitted": 2, "done": 2, "ok": 1, "failed": 1, "retries": 4}


def test_non_retryable():
    cfg = TaskQueueConfig(max_retries=3, retry_on=(IOError,), backoff_s=0)
    with TaskQueue(cfg) as q:
        r = q.submit(lambda: 1 / 0).result()
    assert not r.ok and r.attempts == 1


def test_map_reduce_deterministic_order():
    shards = [[1, 2], [3], [4, 5, 6]]
    out = map_reduce(shards, lambda s: [sum(s)], lambda a, b: a + b, [], TaskQueueConfig(max_workers=3))
    assert out == [3, 3, 15]

    def mapper(s):
        if s == "bad":
            raise ValueError("x")
        return len(s)
    cfg = TaskQueueConfig(max_retries=0)
    with pytest.raises(MapReduceError) as e:
        map_reduce(["ab", "bad", "c"], mapper, lambda a, b: a + b, 0, cfg)
    assert list(e.value.failures) == [1]
    assert map_reduce(["ab", "bad", "c"], mapper, lambda a, b: a + b, 0, cfg, skip_failed=True) == 3


def test_ci_import_check():
    mods = discover_modules("cheesestack.infra")
    assert "cheesestack.infra.config" in mods and "cheesestack.infra.ci" in mods
    rep = check_imports(CIConfig(package="cheesestack.infra"))
    assert rep.passed and len(rep.ok) >= 12
    assert check_imports(CIConfig(package="cheesestack_does_not_exist")).failed
    assert main(["--imports", "--exclude", "cheesestack.pipeline"]) in (0, 1)
