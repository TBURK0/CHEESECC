import numpy as np

from cheesestack.ml.training import (
    EarlyStopping, MLPClassifier, ModelCheckpoint, SoftmaxRegression, Trainer, TrainerConfig, load_checkpoint,
)


def blobs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [3, 3], [-3, 3]])
    y = rng.integers(0, 3, n)
    return centers[y] + rng.normal(0, 0.7, (n, 2)), y


def xor(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)


def test_softmax_regression_learns_blobs():
    X, y = blobs()
    Xv, yv = blobs(seed=1)
    m = SoftmaxRegression(2, 3)
    hist = Trainer(m, TrainerConfig(epochs=20, lr=0.2)).fit(X, y, Xv, yv)
    assert hist["loss"][-1] < hist["loss"][0]
    assert hist["val_acc"][-1] > 0.9


def test_numeric_gradient_softmax():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(5, 3)), np.array([0, 1, 2, 1, 0])
    m = SoftmaxRegression(3, 3, weight_decay=0.0)
    W0 = m.params["W"].copy()
    m.train_step(X, y, lr=1.0)  # W1 = W0 - grad
    grad = W0 - m.params["W"]
    m.params["W"] = W0.copy()
    num = np.zeros_like(W0)
    eps = 1e-5
    for idx in np.ndindex(W0.shape):
        m.params["W"] = W0.copy(); m.params["W"][idx] += eps; lp = m.loss(X, y)
        m.params["W"] = W0.copy(); m.params["W"][idx] -= eps; lm = m.loss(X, y)
        num[idx] = (lp - lm) / (2 * eps)
    assert np.allclose(grad, num, atol=1e-6)


def test_mlp_solves_xor_where_linear_fails():
    X, y = xor()
    lin = SoftmaxRegression(2, 2)
    Trainer(lin, TrainerConfig(epochs=30, lr=0.5)).fit(X, y)
    mlp = MLPClassifier(2, 2, hidden=16)
    Trainer(mlp, TrainerConfig(epochs=60, lr=0.1)).fit(X, y)
    assert np.mean(lin.predict(X) == y) < 0.75
    assert np.mean(mlp.predict(X) == y) > 0.9
    assert mlp.embed(X).shape == (len(X), 16)


def test_early_stopping_and_checkpoint(tmp_path):
    X, y = blobs()
    Xv, yv = blobs(seed=1)
    m = SoftmaxRegression(2, 3)
    es = EarlyStopping(patience=2, min_delta=1e-2)
    ck = ModelCheckpoint(str(tmp_path / "ckpt"))
    hist = Trainer(m, TrainerConfig(epochs=200, lr=0.2), [es, ck]).fit(X, y, Xv, yv)
    assert len(hist["loss"]) < 200
    m2 = SoftmaxRegression(2, 3, seed=5)
    meta = load_checkpoint(ck.saved[-1], m2)
    assert np.allclose(m2.loss(Xv, yv), meta["val_loss"])
    assert np.isclose(m.loss(Xv, yv), es.best)  # restore_best put the best weights back
    assert meta["val_loss"] <= es.best + 1e-12


def test_training_deterministic():
    X, y = blobs()
    runs = []
    for _ in range(2):
        m = MLPClassifier(2, 3, seed=1)
        Trainer(m, TrainerConfig(epochs=3, seed=4)).fit(X, y)
        runs.append(m.state_dict()["W1"])
    assert np.array_equal(*runs)
