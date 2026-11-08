"""Training pipeline: Model interface, numpy reference models, Trainer + callbacks.

``Model`` is the seam where a real framework plugs in (a torch/JAX wrapper would
implement the same methods). Two working numpy models are provided so the
pipeline trains end to end in tests:

- ``SoftmaxRegression``  multinomial logistic regression, L2 weight decay
- ``MLPClassifier``      one hidden ReLU layer, He init, softmax cross-entropy

``Trainer`` runs mini-batch SGD (optional momentum, step LR decay), evaluates on
a validation set each epoch, and drives callbacks (``EarlyStopping``,
``ModelCheckpoint``, ``History``).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))


class Model(ABC):
    """Minimal trainable-classifier interface.

    ``train_step`` does forward + backward + parameter update for one batch and
    returns the batch loss. ``state_dict`` / ``load_state_dict`` exchange plain
    numpy arrays so checkpointing is framework-agnostic.
    """

    num_classes: int

    @abstractmethod
    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return cross_entropy(self.predict_proba(X), y)

    @abstractmethod
    def state_dict(self) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, np.ndarray]) -> None: ...


class _SGDMixin:
    momentum: float = 0.0
    _vel: dict

    def _apply(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float) -> None:
        for k, g in grads.items():
            v = self._vel.get(k)
            v = g if v is None or self.momentum == 0 else self.momentum * v + g
            self._vel[k] = v
            params[k] -= lr * v


class SoftmaxRegression(_SGDMixin, Model):
    def __init__(self, num_features: int, num_classes: int, weight_decay: float = 1e-4,
                 momentum: float = 0.0, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.params = {"W": rng.normal(0, 0.01, (num_features, num_classes)), "b": np.zeros(num_classes)}
        self._vel = {}

    def predict_proba(self, X):
        return softmax(X @ self.params["W"] + self.params["b"])

    def train_step(self, X, y, lr):
        P = self.predict_proba(X)
        loss = cross_entropy(P, y)
        d = P.copy()
        d[np.arange(len(y)), y] -= 1.0
        d /= len(y)
        grads = {"W": X.T @ d + self.weight_decay * self.params["W"], "b": d.sum(axis=0)}
        self._apply(self.params, grads, lr)
        return loss

    def state_dict(self):
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state):
        self.params = {k: np.array(v, dtype=float) for k, v in state.items()}


class MLPClassifier(_SGDMixin, Model):
    def __init__(self, num_features: int, num_classes: int, hidden: int = 32, weight_decay: float = 1e-4,
                 momentum: float = 0.9, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / num_features), (num_features, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, np.sqrt(2.0 / hidden), (hidden, num_classes)), "b2": np.zeros(num_classes),
        }
        self._vel = {}

    def _forward(self, X):
        h_pre = X @ self.params["W1"] + self.params["b1"]
        h = np.maximum(h_pre, 0.0)
        return h_pre, h, softmax(h @ self.params["W2"] + self.params["b2"])

    def predict_proba(self, X):
        return self._forward(X)[2]

    def embed(self, X):
        """Hidden-layer activations; used as embeddings for drift monitoring."""
        return self._forward(X)[1]

    def train_step(self, X, y, lr):
        h_pre, h, P = self._forward(X)
        loss = cross_entropy(P, y)
        d2 = P.copy()
        d2[np.arange(len(y)), y] -= 1.0
        d2 /= len(y)
        dh = (d2 @ self.params["W2"].T) * (h_pre > 0)
        wd = self.weight_decay
        grads = {"W2": h.T @ d2 + wd * self.params["W2"], "b2": d2.sum(0),
                 "W1": X.T @ dh + wd * self.params["W1"], "b1": dh.sum(0)}
        self._apply(self.params, grads, lr)
        return loss

    def state_dict(self):
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state):
        self.params = {k: np.array(v, dtype=float) for k, v in state.items()}


# --------------------------------------------------------------------------- callbacks


class Callback:
    def on_train_begin(self, trainer: "Trainer") -> None: ...

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None: ...

    def on_train_end(self, trainer: "Trainer") -> None: ...


@dataclass
class History(Callback):
    history: dict = field(default_factory=dict)

    def on_epoch_end(self, trainer, epoch, logs):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


@dataclass
class EarlyStopping(Callback):
    monitor: str = "val_loss"
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"
    restore_best: bool = True
    best: float = field(default=float("nan"), init=False)
    best_epoch: int = field(default=-1, init=False)
    wait: int = field(default=0, init=False)
    _best_state: Optional[dict] = field(default=None, init=False, repr=False)

    def _better(self, v: float) -> bool:
        if np.isnan(self.best):
            return True
        return v < self.best - self.min_delta if self.mode == "min" else v > self.best + self.min_delta

    def on_train_begin(self, trainer):
        self.best, self.best_epoch, self.wait, self._best_state = float("nan"), -1, 0, None

    def on_epoch_end(self, trainer, epoch, logs):
        v = logs.get(self.monitor)
        if v is None:
            return
        if self._better(v):
            self.best, self.best_epoch, self.wait = v, epoch, 0
            self._best_state = trainer.model.state_dict()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.stop_training = True

    def on_train_end(self, trainer):
        if self.restore_best and self._best_state is not None:
            trainer.model.load_state_dict(self._best_state)


@dataclass
class ModelCheckpoint(Callback):
    directory: str
    monitor: str = "val_loss"
    mode: str = "min"
    save_best_only: bool = True
    best: float = field(default=float("nan"), init=False)
    saved: list = field(default_factory=list, init=False)

    def on_epoch_end(self, trainer, epoch, logs):
        v = logs.get(self.monitor, float("nan"))
        improved = np.isnan(self.best) or (v < self.best if self.mode == "min" else v > self.best)
        if self.save_best_only and not improved:
            return
        if improved:
            self.best = v
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        name = "best.npz" if self.save_best_only else f"epoch_{epoch:04d}.npz"
        path = os.path.join(self.directory, name)
        save_checkpoint(path, trainer.model, {"epoch": epoch, **logs})
        self.saved.append(path)


def save_checkpoint(path: os.PathLike, model: Model, meta: Optional[dict] = None) -> None:
    arrays = {f"param__{k}": v for k, v in model.state_dict().items()}
    for k, v in (meta or {}).items():
        arrays[f"meta__{k}"] = np.asarray(v)
    np.savez(path, **arrays)


def load_checkpoint(path: os.PathLike, model: Model) -> dict:
    with np.load(path) as z:
        model.load_state_dict({k[7:]: z[k] for k in z.files if k.startswith("param__")})
        return {k[6:]: z[k].item() for k in z.files if k.startswith("meta__")}


# --------------------------------------------------------------------------- trainer


@dataclass
class TrainerConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.1
    lr_decay: float = 1.0  # multiply lr by this ...
    lr_decay_every: int = 10  # ... every N epochs
    shuffle: bool = True
    seed: int = 0


class Trainer:
    """Generic mini-batch loop over any ``Model``.

    # TODO(ml): data loaders streaming from the manifest with augmentation workers,
    # mixed precision / device placement (in the framework backend), gradient
    # clipping, distributed training, experiment tracking.
    """

    def __init__(self, model: Model, config: Optional[TrainerConfig] = None,
                 callbacks: Sequence[Callback] = ()) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        self.history = History()
        self.callbacks = [self.history, *callbacks]
        self.stop_training = False
        self.epoch = 0

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> dict:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.stop_training = False
        for cb in self.callbacks:
            cb.on_train_begin(self)
        lr = cfg.lr
        for epoch in range(cfg.epochs):
            self.epoch = epoch
            idx = rng.permutation(len(X)) if cfg.shuffle else np.arange(len(X))
            losses, counts = [], []
            for s in range(0, len(X), cfg.batch_size):
                b = idx[s:s + cfg.batch_size]
                losses.append(self.model.train_step(X[b], y[b], lr))
                counts.append(len(b))
            logs = {"loss": float(np.average(losses, weights=counts)), "lr": lr}
            logs["acc"] = float(np.mean(self.model.predict(X) == y))
            if X_val is not None and y_val is not None:
                logs["val_loss"] = self.model.loss(np.asarray(X_val, float), np.asarray(y_val, int))
                logs["val_acc"] = float(np.mean(self.model.predict(np.asarray(X_val, float)) == y_val))
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)
            if (epoch + 1) % cfg.lr_decay_every == 0:
                lr *= cfg.lr_decay
            if self.stop_training:
                break
        for cb in self.callbacks:
            cb.on_train_end(self)
        return self.history.history
