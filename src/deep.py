"""Version 0.5: neural networks for spectra (docs/v0.5_deep_learning_plan.md).

The networks are wrapped as scikit-learn estimators, so the Version 0.4 machinery (patient-grouped
search, sigmoid calibration on out-of-fold predictions, cut-off from validation, caching) applies
unchanged and the numbers stay comparable.

Two architectures, both deliberately small for 2,977 training spectra:
- `MLP`: fully connected, the standard neural baseline for fixed-length vectors.
- `SpectrumCNN`: 1-D convolutions along m/z. Neighbouring bins belong to the same peak, so sharing
  weights along that axis needs far fewer parameters than a fully connected layer.

Training uses early stopping on an inner split of the data it is given (never validation or test rows),
restores the best epoch's weights, and records the loss curve.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import check_is_fitted
from torch import nn

DEEP_KINDS = ("mlp", "cnn")


class DeepError(ValueError):
    pass


def set_threads(n_threads: int) -> None:
    """PyTorch CPU threads (this project has no GPU)."""
    torch.set_num_threads(max(1, int(n_threads)))


@contextmanager
def single_thread():
    """Predict on one thread: several threads can sum a layer in different orders, and the validation
    results of the development run must be reproduced bit for bit by the test run."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def parse_sizes(value: Any) -> tuple[int, ...]:
    """'256,64' or [256, 64] -> (256, 64)."""
    def one(part: Any) -> int:
        if isinstance(part, bool) or (isinstance(part, float) and not part.is_integer()):
            raise DeepError(f"Layer sizes must be whole numbers, got {part!r}")
        return int(part)

    try:
        parts = [p.strip() for p in value.split(",") if p.strip()] if isinstance(value, str) else list(value)
        sizes = tuple(one(p) for p in parts)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DeepError(f"Cannot read layer sizes from {value!r}") from exc
    if not sizes or any(s < 1 for s in sizes):
        raise DeepError(f"Layer sizes must be positive integers, got {value!r}")
    return sizes


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_features
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)]
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class SpectrumCNN(nn.Module):
    """Convolutions along m/z, then global average pooling, so the length of the spectrum is free."""

    def __init__(self, channels: tuple[int, ...], kernel_size: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        previous = 1
        for size in channels:
            blocks += [nn.Conv1d(previous, size, kernel_size, padding=kernel_size // 2),
                       nn.BatchNorm1d(size), nn.ReLU(), nn.MaxPool1d(4)]
            previous = size
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(previous, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.unsqueeze(1))).squeeze(1)


@dataclass
class TrainingHistory:
    """Per-epoch loss and inner-validation AUROC; `best_epoch` is the restored one."""
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_roc_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"train_loss": self.train_loss, "validation_loss": self.validation_loss,
                "validation_roc_auc": self.validation_roc_auc, "best_epoch": self.best_epoch,
                "epochs": len(self.train_loss), "stopped_early": self.stopped_early}


class TorchClassifier(ClassifierMixin, BaseEstimator):
    """A small neural network with early stopping, as a scikit-learn classifier.

    `fit` splits off `inner_validation_fraction` of the rows it receives to decide when to stop; those
    rows are part of the training data it was given, never the validation or test parts of a split.
    """

    def __init__(self, kind: str = "mlp", hidden: Any = "256,64", channels: Any = "16,32",
                 kernel_size: int = 9, dropout: float = 0.3, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, batch_size: int = 64, max_epochs: int = 60, patience: int = 8,
                 inner_validation_fraction: float = 0.15, positive_class_weight: bool = False,
                 random_state: int = 42):
        self.kind = kind
        self.hidden = hidden
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.inner_validation_fraction = inner_validation_fraction
        self.positive_class_weight = positive_class_weight
        self.random_state = random_state

    # ---------------------------------------------------------------- building

    def _build(self, n_features: int) -> nn.Module:
        if self.kind == "mlp":
            return MLP(n_features, parse_sizes(self.hidden), float(self.dropout))
        if self.kind == "cnn":
            if int(self.kernel_size) < 1:
                raise DeepError(f"kernel_size must be >= 1, got {self.kernel_size}")
            return SpectrumCNN(parse_sizes(self.channels), int(self.kernel_size), float(self.dropout))
        raise DeepError(f"Unknown network kind {self.kind!r} (known: {', '.join(DEEP_KINDS)})")

    def _tensors(self, X: np.ndarray, y: np.ndarray | None = None):
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        if y is None:
            return x
        return x, torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    # ---------------------------------------------------------------- fitting

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchClassifier:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        if X.ndim != 2 or len(X) != len(y):
            raise DeepError(f"Expected a matrix and one label per row, got {X.shape} and {y.shape}")
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise DeepError("Both classes are needed to train a network.")
        if not 0 < float(self.inner_validation_fraction) < 1:
            raise DeepError(f"inner_validation_fraction must be in (0, 1), got {self.inner_validation_fraction}")
        torch.manual_seed(int(self.random_state))
        self.n_features_in_ = X.shape[1]

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(self.inner_validation_fraction),
                                          random_state=int(self.random_state))
        inner_train, inner_val = next(splitter.split(X, y))
        x_train, y_train = self._tensors(X[inner_train], y[inner_train])
        x_val, y_val = self._tensors(X[inner_val], y[inner_val])

        model = self._build(X.shape[1])
        optimiser = torch.optim.AdamW(model.parameters(), lr=float(self.learning_rate),
                                      weight_decay=float(self.weight_decay))
        weight = None
        if self.positive_class_weight:
            positives = max(int((y[inner_train] == 1).sum()), 1)
            weight = torch.tensor(float((y[inner_train] == 0).sum()) / positives)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        plain_loss = nn.BCEWithLogitsLoss()

        history = TrainingHistory()
        best_score, best_state, waited = -np.inf, copy.deepcopy(model.state_dict()), 0
        generator = torch.Generator().manual_seed(int(self.random_state))
        for epoch in range(int(self.max_epochs)):
            model.train()
            order = torch.randperm(len(x_train), generator=generator)
            total, seen = 0.0, 0
            for start in range(0, len(order), int(self.batch_size)):
                batch = order[start:start + int(self.batch_size)]
                if len(batch) == 1 and start > 0:
                    break            # batch normalisation cannot standardise a single row; drop the remainder
                optimiser.zero_grad()
                loss = loss_fn(model(x_train[batch]), y_train[batch])
                loss.backward()
                optimiser.step()
                total += float(loss.detach()) * len(batch)
                seen += len(batch)
            model.eval()
            with torch.no_grad():
                logits = model(x_val)
                validation_loss = float(plain_loss(logits, y_val))
                scores = torch.sigmoid(logits).numpy()
            score = float(roc_auc_score(y[inner_val], scores)) if np.unique(y[inner_val]).size == 2 else float("nan")
            history.train_loss.append(total / seen)
            history.validation_loss.append(validation_loss)
            history.validation_roc_auc.append(score)
            if np.isfinite(score) and score > best_score:
                best_score, best_state, waited = score, copy.deepcopy(model.state_dict()), 0
                history.best_epoch = epoch
            else:
                waited += 1
                if waited >= int(self.patience):
                    history.stopped_early = True
                    break
        model.load_state_dict(best_state)                 # checkpointing: keep the best epoch
        model.eval()
        self.model_ = model
        self.history_ = history
        self.inner_validation_roc_auc_ = best_score
        return self

    # ---------------------------------------------------------------- predicting

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise DeepError(f"Expected {self.n_features_in_} features per row, got {X.shape}")
        with torch.no_grad(), single_thread():
            return self.model_(self._tensors(X)).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probability = expit(self.decision_function(X))     # expit, not 1/(1+exp(-x)): no overflow warning
        return np.column_stack([1 - probability, probability])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    @property
    def history(self) -> dict[str, Any]:
        if not hasattr(self, "history_"):
            raise NotFittedError("The network has not been trained yet.")
        return self.history_.to_dict()

    def n_parameters(self) -> int:
        check_is_fitted(self, "model_")
        return int(sum(p.numel() for p in self.model_.parameters()))
