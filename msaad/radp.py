"""
radp.py — Phase 3 Layer 1: Resilient Adversarial Detection and Purification (RADP)

Components:
  1. GAN (Generator + Discriminator) to learn adversarial sample distribution
  2. MSAPG: Multi-Source Adversarial Perturbation Generator (uses adversarial.py)
  3. RAD: Resilient Adversarial Detector (online logistic regression)
  4. Pruner: filters adversarial samples before passing to base classifiers
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional

from msaad.config import (
    GAN_NOISE_DIM, GAN_HIDDEN_DIM, GAN_LR,
    GAN_WARMUP_EPOCHS, MINI_BATCH_SIZE, RAD_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────
#  GAN: Generator
# ─────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Maps random noise to synthetic adversarial feature vectors.
    Output dimension matches the preprocessed feature space.
    """

    def __init__(self, noise_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.Sigmoid(),  # outputs in [0,1] to match normalized features
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─────────────────────────────────────────────────────────────
#  GAN: Discriminator
# ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Classifies feature vectors as real adversarial (1) or synthetic/clean (0).
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
#  Adversarial GAN wrapper
# ─────────────────────────────────────────────────────────────

class AdversarialGAN:
    """
    Online GAN that learns to mimic real adversarial samples.

    The Generator produces synthetic adversarial examples which, combined
    with MSAPG-generated real ART perturbations, form the training set for
    the RAD detector.
    """

    def __init__(self, feature_dim: int,
                 noise_dim: int = GAN_NOISE_DIM,
                 hidden_dim: int = GAN_HIDDEN_DIM,
                 lr: float = GAN_LR):
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.device = torch.device("cpu")  # CPU for streaming compatibility

        self.G = Generator(noise_dim, feature_dim, hidden_dim).to(self.device)
        self.D = Discriminator(feature_dim, hidden_dim).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        # Buffer of real adversarial samples for discriminator training
        self._adv_buffer: deque = deque(maxlen=512)
        self._clean_buffer: deque = deque(maxlen=512)
        self._step = 0

    def add_adversarial(self, X_adv: np.ndarray) -> None:
        """Add real adversarial samples (from MSAPG/ART) to training buffer."""
        for row in X_adv:
            self._adv_buffer.append(row.astype(np.float32))

    def add_clean(self, X_clean: np.ndarray) -> None:
        """Add clean samples to the training buffer."""
        for row in X_clean:
            self._clean_buffer.append(row.astype(np.float32))

    def train_step(self, batch_size: int = MINI_BATCH_SIZE) -> dict[str, float]:
        """
        Run one GAN training step (D update then G update).
        Returns dict with d_loss and g_loss.
        """
        n_adv = len(self._adv_buffer)
        n_clean = len(self._clean_buffer)
        if n_adv < batch_size or n_clean < batch_size:
            return {"d_loss": 0.0, "g_loss": 0.0}

        # Sample real adversarial batch
        idx_adv = np.random.choice(n_adv, batch_size, replace=False)
        real_adv = torch.tensor(
            [self._adv_buffer[i] for i in idx_adv], dtype=torch.float32
        ).to(self.device)

        # ── Train Discriminator ──────────────────────────────
        self.opt_D.zero_grad()

        # Real adversarial → label 1
        real_labels = torch.ones(batch_size, 1, device=self.device)
        d_real = self.D(real_adv)
        d_loss_real = self.criterion(d_real, real_labels)

        # Fake (generated) adversarial → label 0
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_adv = self.G(z).detach()
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        d_fake = self.D(fake_adv)
        d_loss_fake = self.criterion(d_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.opt_D.step()

        # ── Train Generator ──────────────────────────────────
        self.opt_G.zero_grad()
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_adv = self.G(z)
        # Generator wants D to output 1 for its fakes
        g_loss = self.criterion(self.D(fake_adv), real_labels)
        g_loss.backward()
        self.opt_G.step()

        self._step += 1
        return {"d_loss": float(d_loss.item()), "g_loss": float(g_loss.item())}

    def generate_synthetic_adversarial(self, n: int) -> np.ndarray:
        """Generate n synthetic adversarial feature vectors."""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n, self.noise_dim, device=self.device)
            samples = self.G(z).cpu().numpy()
        self.G.train()
        return samples.astype(np.float32)

    def warmup(self, X_adv: np.ndarray, X_clean: np.ndarray,
               epochs: int = GAN_WARMUP_EPOCHS) -> None:
        """Pre-train GAN for a fixed number of epochs on buffered data."""
        self.add_adversarial(X_adv)
        self.add_clean(X_clean)
        steps_per_epoch = max(10, len(self._adv_buffer) // MINI_BATCH_SIZE)
        for _ in range(epochs * steps_per_epoch):
            self.train_step()


# ─────────────────────────────────────────────────────────────
#  RAD: Resilient Adversarial Detector
# ─────────────────────────────────────────────────────────────

class ResilientAdversarialDetector:
    """
    Online binary classifier distinguishing clean samples from adversarial ones.
    Uses a simple online logistic regression (SGD) for streaming compatibility.
    """

    def __init__(self, feature_dim: int, threshold: float = RAD_THRESHOLD):
        self.threshold = threshold
        self.feature_dim = feature_dim
        self._w: Optional[np.ndarray] = None   # weight vector
        self._b: float = 0.0                   # bias
        self._lr = 0.01
        self._n = 0
        self._initialized = False

    def _init_weights(self, n_features: int) -> None:
        self._w = np.zeros(n_features, dtype=np.float64)
        self._b = 0.0
        self._initialized = True

    @staticmethod
    def _sigmoid(z: float | np.ndarray) -> float | np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict_proba(self, x: np.ndarray) -> float:
        """Return P(adversarial | x)."""
        if not self._initialized:
            return 0.0
        x = x.astype(np.float64)
        if len(x) != len(self._w):
            # Feature dimension mismatch (pipeline still stabilising)
            return 0.0
        logit = float(np.dot(self._w, x)) + self._b
        return float(self._sigmoid(logit))

    def predict(self, x: np.ndarray) -> int:
        """Return 1 if adversarial, 0 if clean."""
        return int(self.predict_proba(x) >= self.threshold)

    def learn_one(self, x: np.ndarray, is_adversarial: int) -> None:
        """Online SGD update for logistic regression."""
        x = x.astype(np.float64)
        if not self._initialized:
            self._init_weights(len(x))
        if len(x) != len(self._w):
            return  # skip dimension mismatch
        y_pred = float(self._sigmoid(np.dot(self._w, x) + self._b))
        error = y_pred - float(is_adversarial)
        # L2 regularized SGD
        lr = self._lr / (1 + 0.001 * self._n)
        self._w -= lr * (error * x + 0.001 * self._w)
        self._b -= lr * error
        self._n += 1

    def train_batch(self, X: np.ndarray, labels: np.ndarray) -> None:
        """Train on a batch of (features, is_adversarial) pairs."""
        for x, y in zip(X, labels):
            self.learn_one(x, int(y))


# ─────────────────────────────────────────────────────────────
#  RADP: Full Layer 1 Orchestrator
# ─────────────────────────────────────────────────────────────

class RADPLayer:
    """
    Resilient Adversarial Detection and Purification (RADP) — Layer 1.

    Responsibilities:
      - Maintain the GAN and use it + MSAPG to build the RAD training set
      - Detect adversarial samples online (RAD)
      - Prune (filter) detected adversarial samples
      - Return only clean-flagged samples to Layer 2
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.gan = AdversarialGAN(feature_dim)
        self.rad = ResilientAdversarialDetector(feature_dim)
        self._warmup_done = False
        self._prune_count = 0
        self._pass_count = 0

    def warmup(
        self,
        X_clean: np.ndarray,
        X_adv: np.ndarray,
    ) -> None:
        """
        Pre-train GAN and RAD using warmup buffers.
        X_clean: (N, F) clean preprocessed samples
        X_adv:   (N, F) adversarial samples (from MSAPG)
        """
        self.gan.warmup(X_adv, X_clean)

        # Train RAD: clean=0, adversarial=1
        n = min(len(X_clean), len(X_adv))
        if n > 0:
            X_all = np.vstack([X_clean[:n], X_adv[:n]])
            y_all = np.array([0] * n + [1] * n, dtype=np.int32)
            self.rad.train_batch(X_all, y_all)

        # Also use GAN-generated synthetic adversarial for additional training
        syn_adv = self.gan.generate_synthetic_adversarial(n)
        self.rad.train_batch(
            np.vstack([X_clean[:n], syn_adv[:n]]),
            np.array([0] * n + [1] * n, dtype=np.int32),
        )
        self._warmup_done = True

    def process(
        self,
        x_arr: np.ndarray,
        is_adversarial_truth: Optional[int] = None,
    ) -> tuple[bool, float]:
        """
        Evaluate one sample.

        Args:
            x_arr: preprocessed feature array
            is_adversarial_truth: ground-truth label for RAD training (optional)

        Returns:
            (is_pruned: bool, adv_proba: float)
            is_pruned=True means the sample is flagged as adversarial and removed.
        """
        adv_proba = self.rad.predict_proba(x_arr)
        is_detected = adv_proba >= self.rad.threshold

        # Online RAD update if truth label is available
        if is_adversarial_truth is not None:
            self.rad.learn_one(x_arr, is_adversarial_truth)

        if is_detected:
            self._prune_count += 1
            return True, adv_proba
        else:
            self._pass_count += 1
            return False, adv_proba

    def online_update_gan(
        self,
        X_clean: np.ndarray,
        X_adv: np.ndarray,
    ) -> None:
        """Update GAN buffers and run a training step."""
        self.gan.add_clean(X_clean)
        self.gan.add_adversarial(X_adv)
        self.gan.train_step()

    @property
    def stats(self) -> dict:
        return {
            "pruned": self._prune_count,
            "passed": self._pass_count,
            "prune_rate": (
                self._prune_count / max(1, self._prune_count + self._pass_count)
            ),
        }
