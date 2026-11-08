import numpy as np

from cheesestack.ml.training import SoftmaxRegression, Trainer, TrainerConfig
from cheesestack.ml.tuning import Choice, IntUniform, LogUniform, grid_search, random_search, successive_halving


def quad(p):
    return -((p["x"] - 0.3) ** 2) - (p["y"] - 2) ** 2


def test_grid_search_finds_optimum():
    res = grid_search(quad, {"x": [0.0, 0.3, 0.6], "y": [1, 2, 3]})
    assert res.best_params == {"x": 0.3, "y": 2} and len(res.trials) == 9
    res_min = grid_search(lambda p: -quad(p), {"x": [0.0, 0.3], "y": [2]}, mode="min")
    assert res_min.best_params["x"] == 0.3


def test_random_search_seeded():
    space = {"lr": LogUniform(1e-4, 1.0), "n": IntUniform(1, 5), "act": Choice(["relu", "tanh"]), "k": 3}
    a = random_search(lambda p: -abs(np.log10(p["lr"]) + 2), space, n_trials=30, seed=1)
    b = random_search(lambda p: -abs(np.log10(p["lr"]) + 2), space, n_trials=30, seed=1)
    assert a.best_params == b.best_params
    assert all(1e-4 <= t.params["lr"] <= 1 and 1 <= t.params["n"] <= 5 and t.params["k"] == 3 for t in a.trials)
    assert abs(np.log10(a.best_params["lr"]) + 2) < 0.3


def test_successive_halving_budget_schedule():
    calls = []

    def obj(p, budget):
        calls.append(budget)
        return -abs(p["x"] - 0.5) * (1 + 1 / budget)

    res = successive_halving(obj, {"x": [i / 26 for i in range(27)]}, n_configs=27, min_budget=1,
                             max_budget=9, eta=3, seed=0)
    assert calls.count(1) == 27 and calls.count(3) == 9 and calls.count(9) == 3
    assert res.final_budget == 9 and abs(res.best_params["x"] - 0.5) < 0.1


def test_tune_real_training_lr():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2)); y = (X[:, 0] + X[:, 1] > 0).astype(int)

    def obj(p, budget=5):
        m = SoftmaxRegression(2, 2)
        Trainer(m, TrainerConfig(epochs=int(budget), lr=p["lr"])).fit(X, y)
        return float(np.mean(m.predict(X) == y))

    res = grid_search(obj, {"lr": [1e-5, 0.5]})
    assert res.best_params["lr"] == 0.5
