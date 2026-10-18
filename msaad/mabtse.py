"""
mabtse.py — Phase 3 Layer 3: Multi-Armed Bandit LinUCB with Ensemble (MABLUCBE)

Implements:
  - LinUCB (Linear Upper Confidence Bound) contextual bandit to dynamically
    select the best classifier arm based on input feature context
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
    LINUCB_ALPHA,
    MAB_ARM_NAMES, AUWPAE_WINDOW_SIZE,
    LINUCB_CONTEXT_DIM,
)


# ─────────────────────────────────────────────────────────────
#  LinUCB Contextual Bandit
# ─────────────────────────────────────────────────────────────

class LinUCBMAB:
    """
    Contextual Multi-Armed Bandit using the LinUCB (Disjoint) algorithm.

    Each arm maintains its own ridge-regression model that maps a
    d-dimensional context vector to an expected reward.  Arm selection
    maximises  θ_a^T x  +  α √(x^T A_a^{-1} x),  balancing exploitation
    (predicted reward) against exploration (uncertainty).

    Arms:
      0 → ARF-ADWIN
      1 → SRP-DDM
      2 → HT-ADWIN
      3 → KNN-ADWIN
      4 → OLR
      5 → Ensemble (AUWPAE)

    Reward: 1 if the arm's prediction is correct, 0 otherwise.

    Reference:
      Li et al., "A Contextual-Bandit Approach to Personalized News
      Article Recommendation", WWW 2010.
    """

    def __init__(
        self,
        n_arms: int = 6,
        d: int = LINUCB_CONTEXT_DIM,
        alpha: float = LINUCB_ALPHA,
    ):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha

        # Per-arm parameters: A_a (d×d), b_a (d,)
        self._A = [np.eye(d) for _ in range(n_arms)]       # A_a = I_d
        self._b = [np.zeros(d) for _ in range(n_arms)]     # b_a = 0
        self._A_inv = [np.eye(d) for _ in range(n_arms)]   # cached inverse

        self._pull_counts = np.zeros(n_arms, dtype=np.int64)
        self._reward_sums = np.zeros(n_arms, dtype=np.float64)

    def _build_context(self, x: dict) -> np.ndarray:
        """
        Convert a feature dictionary to a fixed-size context vector.

        Uses a deterministic hash-based projection to map variable-length
        feature dicts into a fixed d-dimensional vector, followed by
        L2 normalisation for numerical stability.
        """
        raw = np.array(list(x.values()), dtype=np.float64)

        if len(raw) >= self.d:
            # Truncate to d dimensions
            ctx = raw[:self.d].copy()
        else:
            # Pad with zeros
            ctx = np.zeros(self.d, dtype=np.float64)
            ctx[:len(raw)] = raw

        # L2-normalise to keep UCB numerically stable
        norm = np.linalg.norm(ctx)
        if norm > 0:
            ctx /= norm

        return ctx

    def select_arm(self, x: dict) -> int:
        """
        Select the arm with the highest LinUCB upper confidence bound.

        UCB_a = θ_a^T context  +  α √(context^T A_a^{-1} context)

        Args:
            x: Feature dictionary for the current sample.

        Returns:
            Index of the selected arm.
        """
        ctx = self._build_context(x)
        ucb_values = np.empty(self.n_arms)

        for a in range(self.n_arms):
            A_inv = self._A_inv[a]
            theta_a = A_inv @ self._b[a]                 # θ_a = A_a^{-1} b_a
            pred = float(theta_a @ ctx)                   # exploitation
            var = float(ctx @ A_inv @ ctx)                # uncertainty
            ucb_values[a] = pred + self.alpha * np.sqrt(max(var, 0.0))

        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float, x: dict) -> None:
        """
        Update the arm's ridge-regression model after observing a reward.

        A_a ← A_a + x x^T
        b_a ← b_a + r x
        """
        ctx = self._build_context(x)

        self._A[arm] += np.outer(ctx, ctx)
        self._b[arm] += reward * ctx

        # Incrementally update the inverse via Sherman-Morrison
        A_inv = self._A_inv[arm]
        u = A_inv @ ctx
        denom = 1.0 + float(ctx @ u)
        self._A_inv[arm] = A_inv - np.outer(u, u) / denom

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
#  MABLUCBE: Full Layer 3 Orchestrator
# ─────────────────────────────────────────────────────────────

class MABTSERouter:
    """
    Multi-Armed Bandit LinUCB Ensemble (MABLUCBE) — Layer 3.

    Per incoming sample:
      1. LinUCB selects an arm using the sample's feature context
         (0-4: individual classifier, 5: ensemble)
      2. Prediction is made:
         - If arm 0-4: use the corresponding base classifier directly
         - If arm 5: use AUWPAE over all five classifiers
      3. After observing the true label, LinUCB receives a reward and
         AUWPAE accuracy windows are updated.
    """

    ENSEMBLE_ARM = 5

    def __init__(self, layer2: Layer2Classifiers):
        self.layer2 = layer2
        self.n_classifiers = len(layer2)
        self.mab = LinUCBMAB(n_arms=6)
        self.auwpae = AUWPAE(n_classifiers=self.n_classifiers)

    def predict_one(self, x: dict) -> tuple[int, int, float]:
        """
        Make a prediction using LinUCB contextual arm selection.

        Returns:
            (prediction, arm_selected, confidence_proba)
        """
        arm = self.mab.select_arm(x)

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
        Make prediction, update base classifiers, update LinUCB and AUWPAE.

        Returns:
            (prediction, arm_selected)
        """
        arm = self.mab.select_arm(x)
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
        self.mab.update(arm, reward, x)

        # Update AUWPAE accuracy windows
        clf_preds = [int(p >= 0.5) for p in probas]
        self.auwpae.update(clf_preds, y)

        return pred, arm

    @property
    def arm_summary(self) -> dict:
        return self.mab.arm_summary()
