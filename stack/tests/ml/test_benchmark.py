import numpy as np

from cheesestack.ml.benchmark import BenchmarkConfig, benchmark, compare, model_size


class FakeTimer:
    def __init__(self, durations):
        self.d = list(durations)
        self.t = 0.0
        self.started = False

    def __call__(self):
        if self.started:
            self.t += self.d.pop(0)
        self.started = not self.started
        return self.t


def test_benchmark_percentiles_and_warmup():
    calls = []
    durs = [0.001] * 98 + [0.010, 0.020]
    res = benchmark(lambda x: calls.append(x), [1, 2, 3], BenchmarkConfig(warmup=3, iterations=100, batch_size=4,
                                                                          track_memory=False),
                    timer=FakeTimer(durs))
    assert len(calls) == 103
    assert np.isclose(res.p50_ms, 1.0) and np.isclose(res.max_ms, 20.0)
    assert res.p99_ms > 9.0 and not res.meets(5.0)
    assert np.isclose(res.throughput_per_s, 400 / sum(durs))


def test_benchmark_real_timer_and_memory():
    W = np.random.default_rng(0).normal(size=(64, 64))
    res = benchmark(lambda x: x @ W, np.ones((8, 64)), BenchmarkConfig(warmup=2, iterations=10), name="mm")
    assert res.mean_ms >= 0 and res.peak_mem_bytes > 0
    assert "mm" in compare([res])


def test_model_size():
    rep = model_size({"W": np.zeros((10, 5)), "b": np.zeros(5)})
    assert rep.num_params == 55 and rep.size_bytes == 55 * 8
    assert model_size({"W": np.zeros((10, 5))}, np.float16).size_bytes == 100
