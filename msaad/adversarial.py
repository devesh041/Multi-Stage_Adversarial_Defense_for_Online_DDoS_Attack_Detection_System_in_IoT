"""
adversarial.py — Phase 2: Adversarial attack simulation using ART.

Uses a PyTorch MLP surrogate model (required for gradient-based attacks).
Implements FGSM, BIM, PGD (gradient-based via PyTorchClassifier) and
C&W (L2, also requires gradients). All four are wrapped via ART.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ART imports — PyTorch wrapper required for gradient attacks
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    BasicIterativeMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
)

from msaad.config import (
    ADV_EPSILON,
    BIM_EPS_STEP, BIM_MAX_ITER,
    PGD_EPS_STEP, PGD_MAX_ITER,
    CW_CONFIDENCE, CW_MAX_ITER,
    ART_BUFFER_SIZE, MINI_BATCH_SIZE,
)

# Suppress noisy library-specific deprecation warnings while retaining system warnings
warnings.filterwarnings("ignore", category=UserWarning, module="art")
warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────
#  Surrogate PyTorch MLP
# ─────────────────────────────────────────────────────────────

class _SurrogateMLP(nn.Module):
    """Lightweight MLP for white-box surrogate attacks."""

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 2),   # 2 classes: Normal (0), DDoS (1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
#  Surrogate model manager
# ─────────────────────────────────────────────────────────────

class SurrogateModel:
    """
    Manages a PyTorch MLP surrogate trained on a sliding window of recent
    preprocessed samples.  Wraps the model in ART's PyTorchClassifier so
    gradient-based attacks (FGSM, BIM, PGD, C&W) work correctly.
    """

    def __init__(self, buffer_size: int = ART_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self._X_buf: deque = deque(maxlen=buffer_size)
        self._y_buf: deque = deque(maxlen=buffer_size)
        self._mlp: Optional[_SurrogateMLP] = None
        self._art_classifier: Optional[PyTorchClassifier] = None
        self._fitted = False
        self._n_features: Optional[int] = None
        self._device = torch.device("cpu")

    def update(self, x_arr: np.ndarray, y: int) -> None:
        """Add sample to buffer; refit periodically."""
        self._X_buf.append(x_arr.astype(np.float32))
        self._y_buf.append(int(y))
        n = len(self._X_buf)
        if n >= min(100, self.buffer_size) and n % 50 == 0:
            self._refit()

    def _refit(self) -> None:
        X = np.stack(list(self._X_buf)).astype(np.float32)
        y = np.array(list(self._y_buf), dtype=np.int64)
        n_features = X.shape[1]

        if len(np.unique(y)) < 2:
            return  # need both classes

        # Build/rebuild model when feature dim changes
        if self._mlp is None or self._n_features != n_features:
            self._mlp = _SurrogateMLP(input_dim=n_features).to(self._device)
            self._n_features = n_features

        X_t = torch.tensor(np.clip(X, 0.0, 1.0), dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=min(64, len(X_t)), shuffle=True
        )

        optimizer = optim.Adam(self._mlp.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        self._mlp.train()
        for _ in range(5):          # 5 quick epochs online
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._mlp(xb), yb)
                loss.backward()
                optimizer.step()
        self._mlp.eval()

        # Wrap in ART PyTorchClassifier
        self._art_classifier = PyTorchClassifier(
            model=self._mlp,
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(self._mlp.parameters(), lr=1e-3),
            input_shape=(n_features,),
            nb_classes=2,
            clip_values=(0.0, 1.0),
            device_type="cpu",
        )
        self._fitted = True

    @property
    def is_ready(self) -> bool:
        return self._fitted and self._art_classifier is not None

    @property
    def art_classifier(self) -> Optional[PyTorchClassifier]:
        return self._art_classifier

    @property
    def n_features(self) -> Optional[int]:
        return self._n_features


# ─────────────────────────────────────────────────────────────
#  Adversarial attack generator
# ─────────────────────────────────────────────────────────────

class AdversarialAttackGenerator:
    """
    Generates adversarial perturbations for streaming data using ART.

    Usage:
      gen = AdversarialAttackGenerator()
      gen.update(x_array, y)              # feed clean samples to train surrogate
      adv = gen.generate(batch, "fgsm")  # generate adversarial perturbations
    """

    def __init__(self):
        self.surrogate = SurrogateModel()
        self._attacks: dict = {}
        self._surrogate_version: int = 0

    def update(self, x_arr: np.ndarray, y: int) -> None:
        """Update surrogate model buffer with a new clean sample."""
        self.surrogate.update(x_arr, y)
        # Invalidate cached attacks so they are rebuilt on next generate()
        self._attacks = {}

    def _get_attacks(self) -> dict:
        """Lazily build ART attack objects around the current surrogate."""
        if self._attacks:
            return self._attacks
        clf = self.surrogate.art_classifier
        self._attacks = {
            "fgsm": FastGradientMethod(
                estimator=clf,
                eps=ADV_EPSILON,
                targeted=False,
                batch_size=MINI_BATCH_SIZE,
            ),
            "bim": BasicIterativeMethod(
                estimator=clf,
                eps=ADV_EPSILON,
                eps_step=BIM_EPS_STEP,
                max_iter=BIM_MAX_ITER,
                targeted=False,
                batch_size=MINI_BATCH_SIZE,
            ),
            "pgd": ProjectedGradientDescent(
                estimator=clf,
                eps=ADV_EPSILON,
                eps_step=PGD_EPS_STEP,
                max_iter=PGD_MAX_ITER,
                targeted=False,
                batch_size=MINI_BATCH_SIZE,
            ),
            "cw": CarliniL2Method(
                classifier=clf,
                confidence=CW_CONFIDENCE,
                max_iter=CW_MAX_ITER,
                batch_size=MINI_BATCH_SIZE,
            ),
        }
        return self._attacks

    def generate(
        self,
        X: np.ndarray,
        method: str = "fgsm",
    ) -> np.ndarray:
        """
        Generate adversarial versions of input batch X.

        Args:
            X: numpy array of shape (n_samples, n_features), values in [0,1]
            method: one of "fgsm", "bim", "pgd", "cw"

        Returns:
            Adversarial X of same shape, clipped to [0, 1].
        """
        if not self.surrogate.is_ready:
            noise = np.random.uniform(
                -ADV_EPSILON, ADV_EPSILON, X.shape
            ).astype(np.float32)
            return np.clip(X + noise, 0.0, 1.0)

        X = np.clip(X.astype(np.float32), 0.0, 1.0)
        attacks = self._get_attacks()
        attack = attacks.get(method)
        if attack is None:
            raise ValueError(
                f"Unknown attack method: {method!r}. "
                f"Choose from {list(attacks)}"
            )

        try:
            X_adv = attack.generate(x=X)
        except Exception as exc:
            print(f"[WARNING] ART '{method}' failed: {exc}. Using noise fallback.")
            noise = np.random.uniform(
                -ADV_EPSILON, ADV_EPSILON, X.shape
            ).astype(np.float32)
            X_adv = X + noise

        return np.clip(X_adv, 0.0, 1.0).astype(np.float32)

    def generate_msapg(self, X: np.ndarray) -> np.ndarray:
        """
        Multi-Source Adversarial Perturbation Generator (MSAPG):
        Average perturbation across all four attack methods.
        Used to build diverse adversarial training data for the GAN.
        """
        results = []
        for method in ["fgsm", "bim", "pgd", "cw"]:
            try:
                results.append(self.generate(X, method=method))
            except Exception:
                results.append(X.copy())
        stacked = np.stack(results, axis=0)  # (4, n_samples, n_features)
        return stacked.mean(axis=0).astype(np.float32)

    @property
    def is_ready(self) -> bool:
        return self.surrogate.is_ready
