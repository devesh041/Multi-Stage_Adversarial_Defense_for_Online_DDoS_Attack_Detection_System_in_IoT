"""
mabtse.py — Phase 3 Layer 3: Multi-Armed Bandit Thompson Sampling with Ensemble (MABTSE)

Implements:
  - Thompson Sampling MAB to dynamically select the best classifier arm
  - Arms: ARF-ADWIN (0), SRP-DDM (1), KNN-ADWIN (2), Ensemble (3)
  - AUWPAE: Accuracy Update Weighted Probability Averaging Ensemble
    (used when the Ensemble arm is selected)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional

from msaad.classifiers import Layer2Classifiers
from msaad.config import (
    MAB_ALPHA_INIT, MAB_BETA_INIT,
    MAB_ARM_NAMES, AUWPAE_WINDOW_SIZE,
)


# ─────────────────────────────────────────────────────────────
#  Thompson Sampling MAB
# ─────────────────────────────────────────────────────────────

class ThompsonSamplingMAB:
    """
    Multi-Armed Bandit with Beta-Bernoulli Thompson Sampling.

    Arms:
      0 → ARF-ADWIN
      1 → SRP-DDM
      2 → KNN-ADWIN
      3 → Ensemble (AUWPAE)

    Reward: 1 if the arm's prediction is correct, 0 otherwise.
    """

    def __init__(
        self,
        n_arms: int = 4,
        alpha_init: float = MAB_ALPHA_INIT,
        beta_init: float = MAB_BETA_INIT,
    ):
        self.n_arms = n_arms
        # Beta distribution parameters per arm
        self._alpha = np.full(n_arms, alpha_init)
        self._beta  = np.full(n_arms, beta_init)
        self._pull_counts = np.zeros(n_arms, dtype=np.int64)
        self._reward_sums = np.zeros(n_arms, dtype=np.float64)

    def select_arm(self) -> int:
        """Sample θ_i ~ Beta(α_i, β_i) for each arm; return argmax."""
        samples = np.array([
            np.random.beta(self._alpha[i], self._beta[i])
            for i in range(self.n_arms)
        ])
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """Update Beta posterior with Bernoulli reward."""
        self._alpha[arm] += reward
        self._beta[arm]  += (1.0 - reward)
        self._pull_counts[arm] += 1
        self._reward_sums[arm] += reward

    @property
    def arm_success_rates(self) -> np.ndarray:
        pulls = np.maximum(self._pull_counts, 1)
        return self._reward_sums / pulls

    def arm_summary(self) -> dict:
        return {
            name: {
                "pulls": int(self._pull_counts[i]),
                "success_rate": float(self._reward_sums[i] / max(1, self._pull_counts[i])),
            }
            for i, name in enumerate(MAB_ARM_NAMES)
        }


# ─────────────────────────────────────────────────────────────
#  AUWPAE: Accuracy Update Weighted Probability Averaging Ensemble
# ─────────────────────────────────────────────────────────────

class AUWPAE:
    """
    Ensemble aggregation using sliding-window accuracy as weights.

    Weight of classifier i = accuracy_i over the last W samples.
    Final prediction = Σ(w_i * p_i) / Σ(w_i), thresholded at 0.5.
    """

    def __init__(self, n_classifiers: int, window_size: int = AUWPAE_WINDOW_SIZE):
        self.n_classifiers = n_classifiers
        self.window_size = window_size
        # Per-classifier sliding window of correct (1) / wrong (0)
        self._windows: list[deque] = [
            deque(maxlen=window_size) for _ in range(n_classifiers)
        ]

    def update(self, classifier_preds: list[int], y_true: int) -> None:
        """Update accuracy windows after observing true label."""
        for i, pred in enumerate(classifier_preds):
            correct = int(pred == y_true)
            self._windows[i].append(correct)

    def _weight(self, i: int) -> float:
        window = self._windows[i]
        if len(window) == 0:
            return 1.0  # uniform prior when no data
        return sum(window) / len(window)

    def predict(self, probas: list[float]) -> int:
        """
        Weighted probability average prediction.

        Args:
            probas: P(DDoS=1) from each classifier

        Returns:
            Binary prediction (0 or 1)
        """
        weights = [self._weight(i) for i in range(self.n_classifiers)]
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0
        weighted_avg = sum(w * p for w, p in zip(weights, probas)) / total_weight
        return int(weighted_avg >= 0.5)

    def predict_proba(self, probas: list[float]) -> float:
        """Return weighted average probability."""
        weights = [self._weight(i) for i in range(self.n_classifiers)]
        total_weight = sum(weights) or 1.0
        return sum(w * p for w, p in zip(weights, probas)) / total_weight


# ─────────────────────────────────────────────────────────────
#  MABTSE: Full Layer 3 Orchestrator
# ─────────────────────────────────────────────────────────────

class MABTSERouter:
    """
    Multi-Armed Bandit Thompson Sampling Ensemble (MABTSE) — Layer 3.

    Per incoming sample:
      1. MAB selects an arm (0-2: individual classifier, 3: ensemble)
      2. Prediction is made:
         - If arm 0-2: use the corresponding base classifier directly
         - If arm 3: use AUWPAE over all three classifiers
      3. After observing the true label, MAB receives reward and
         AUWPAE accuracy windows are updated.
    """

    ENSEMBLE_ARM = 3

    def __init__(self, layer2: Layer2Classifiers):
        self.layer2 = layer2
        self.n_classifiers = len(layer2)
        self.mab = ThompsonSamplingMAB(n_arms=4)
        self.auwpae = AUWPAE(n_classifiers=self.n_classifiers)

    def predict_one(self, x: dict) -> tuple[int, int, float]:
        """
        Make a prediction using Thompson Sampling arm selection.

        Returns:
            (prediction, arm_selected, confidence_proba)
        """
        arm = self.mab.select_arm()

        probas = self.layer2.predict_probas_one(x)

        if arm == self.ENSEMBLE_ARM:
            proba = self.auwpae.predict_proba(probas)
            pred = int(proba >= 0.5)
        else:
            proba = probas[arm]
            pred = int(proba >= 0.5)

        return pred, arm, proba

    def learn_one(self, x: dict, y: int) -> tuple[int, int]:
        """
        Make prediction, update base classifiers, update MAB and AUWPAE.

        Returns:
            (prediction, arm_selected)
        """
        arm = self.mab.select_arm()
        probas = self.layer2.predict_probas_one(x)

        if arm == self.ENSEMBLE_ARM:
            proba = self.auwpae.predict_proba(probas)
            pred = int(proba >= 0.5)
        else:
            proba = probas[arm]
            pred = int(proba >= 0.5)

        # Update base classifiers
        self.layer2.learn_one(x, y)

        # Reward = 1 if correct prediction
        reward = float(pred == y)
        self.mab.update(arm, reward)

        # Update AUWPAE accuracy windows
        clf_preds = [int(p >= 0.5) for p in probas]
        self.auwpae.update(clf_preds, y)

        return pred, arm

    @property
    def arm_summary(self) -> dict:
        return self.mab.arm_summary()
