"""
classifiers.py — Phase 3 Layer 2: Online base classifiers with concept drift detection.

Three streaming classifiers:
  1. ARF-ADWIN  — Adaptive Random Forest (ADWIN Bagging) with drift detection
  2. SRP-DDM    — Streaming Random Patches with DDM drift detector
  3. KNN-ADWIN  — K-Nearest Neighbours with ADWIN drift detection

Each exposes a unified interface:
  .learn_one(x: dict, y: int) -> None
  .predict_proba_one(x: dict) -> float   # P(DDoS=1)
  .predict_one(x: dict) -> int
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# river 0.26.x API
from river.ensemble import ADWINBaggingClassifier, SRPClassifier
from river.tree import HoeffdingTreeClassifier
from river.neighbors import KNNClassifier
from river.drift import ADWIN
from river.drift.binary import DDM

from msaad.config import (
    ARF_N_MODELS, SRP_N_MODELS,
    KNN_N_NEIGHBORS, KNN_WINDOW_SIZE,
)


# ─────────────────────────────────────────────────────────────
#  Base classifier wrapper interface
# ─────────────────────────────────────────────────────────────

class BaseOnlineClassifier:
    """Abstract interface for all Layer 2 classifiers."""

    name: str = "base"

    def learn_one(self, x: dict, y: int) -> None:
        raise NotImplementedError

    def predict_proba_one(self, x: dict) -> float:
        raise NotImplementedError

    def predict_one(self, x: dict) -> int:
        proba = self.predict_proba_one(x)
        return int(proba >= 0.5)


# ─────────────────────────────────────────────────────────────
#  Classifier 1: ARF-ADWIN
# ─────────────────────────────────────────────────────────────

class ARFADWINClassifier(BaseOnlineClassifier):
    """
    Adaptive Random Forest equivalent using ADWINBaggingClassifier with
    HoeffdingTreeClassifier base learners (river 0.26.x API).
    ADWIN handles drift detection automatically within the ensemble.
    """

    name = "ARF-ADWIN"

    def __init__(self, n_models: int = ARF_N_MODELS):
        self._clf = ADWINBaggingClassifier(
            model=HoeffdingTreeClassifier(),
            n_models=n_models,
            seed=42,
        )

    def learn_one(self, x: dict, y: int) -> None:
        self._clf.learn_one(x, y)

    def predict_proba_one(self, x: dict) -> float:
        proba_dict = self._clf.predict_proba_one(x)
        if proba_dict is None:
            return 0.5
        return float(proba_dict.get(1, 0.0))

    def predict_one(self, x: dict) -> int:
        return int(self._clf.predict_one(x) or 0)


# ─────────────────────────────────────────────────────────────
#  Classifier 2: SRP-DDM
# ─────────────────────────────────────────────────────────────

class SRPDDMClassifier(BaseOnlineClassifier):
    """
    Streaming Random Patches ensemble with DDM (Drift Detection Method).
    Uses river.drift.binary.DDM for drift and warning detection.
    """

    name = "SRP-DDM"

    def __init__(self, n_models: int = SRP_N_MODELS):
        self._clf = SRPClassifier(
            n_models=n_models,
            drift_detector=DDM(),
            warning_detector=DDM(),
            seed=42,
        )

    def learn_one(self, x: dict, y: int) -> None:
        self._clf.learn_one(x, y)

    def predict_proba_one(self, x: dict) -> float:
        proba_dict = self._clf.predict_proba_one(x)
        if proba_dict is None:
            return 0.5
        return float(proba_dict.get(1, 0.0))

    def predict_one(self, x: dict) -> int:
        result = self._clf.predict_one(x)
        return int(result or 0)


# ─────────────────────────────────────────────────────────────
#  Classifier 3: KNN-ADWIN
# ─────────────────────────────────────────────────────────────

class KNNADWINClassifier(BaseOnlineClassifier):
    """
    K-Nearest Neighbours with an external ADWIN drift detector.

    When ADWIN detects drift, the KNN classifier is reset to discard
    stale samples (river 0.26.x: KNNClassifier has no window_size param;
    drift-driven reset achieves the same effect).
    """

    name = "KNN-ADWIN"

    def __init__(self, n_neighbors: int = KNN_N_NEIGHBORS,
                 window_size: int = KNN_WINDOW_SIZE):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self._clf = KNNClassifier(n_neighbors=n_neighbors)
        self._adwin = ADWIN()
        self._last_pred: int = 0

    def learn_one(self, x: dict, y: int) -> None:
        # Update ADWIN with prediction error signal
        pred = self.predict_one(x)
        error = int(pred != y)
        self._adwin.update(error)

        if self._adwin.drift_detected:
            # Reset KNN on drift to discard stale concept
            self._clf = KNNClassifier(n_neighbors=self.n_neighbors)

        self._clf.learn_one(x, y)

    def predict_proba_one(self, x: dict) -> float:
        proba_dict = self._clf.predict_proba_one(x)
        if proba_dict is None:
            return 0.5
        return float(proba_dict.get(1, 0.0))

    def predict_one(self, x: dict) -> int:
        result = self._clf.predict_one(x)
        if result is None:
            return 0
        return int(result)



# ─────────────────────────────────────────────────────────────
#  Layer 2 container
# ─────────────────────────────────────────────────────────────

class Layer2Classifiers:
    """
    Container for all three base classifiers.
    Provides unified learn/predict interface used by Layer 3.
    """

    def __init__(self):
        self.classifiers: list[BaseOnlineClassifier] = [
            ARFADWINClassifier(),
            SRPDDMClassifier(),
            KNNADWINClassifier(),
        ]
        self.names = [c.name for c in self.classifiers]

    def learn_one(self, x: dict, y: int) -> None:
        for clf in self.classifiers:
            clf.learn_one(x, y)

    def predict_probas_one(self, x: dict) -> list[float]:
        """Return P(DDoS=1) from each classifier."""
        return [clf.predict_proba_one(x) for clf in self.classifiers]

    def predict_one(self, idx: int, x: dict) -> int:
        return self.classifiers[idx].predict_one(x)

    def predict_proba_one(self, idx: int, x: dict) -> float:
        return self.classifiers[idx].predict_proba_one(x)

    def __len__(self) -> int:
        return len(self.classifiers)

    def __getitem__(self, idx: int) -> BaseOnlineClassifier:
        return self.classifiers[idx]
