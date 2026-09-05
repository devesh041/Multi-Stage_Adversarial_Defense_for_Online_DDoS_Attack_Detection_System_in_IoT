"""
preprocessing.py — Phase 1: Online streaming preprocessing pipeline.

Implements:
  1. Incremental one-hot encoding (categorical)
  2. Online mean imputation for missing values
  3. Mini-batch K-means clustering (representative feature augmentation)
  4. Online SMOTE for minority-class balancing
  5. Online min-max normalization
  6. Information Gain based feature selection
  7. Pearson Correlation based redundant feature removal
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Optional

import numpy as np
from river import preprocessing as river_prep
from sklearn.cluster import MiniBatchKMeans

from msaad.config import (
    KMEANS_CLUSTERS, KMEANS_BATCH_SIZE,
    INFO_GAIN_THRESHOLD, PEARSON_THRESHOLD,
    SMOTE_K_NEIGHBOURS, SMOTE_RESERVOIR_SIZE,
)


# ─────────────────────────────────────────────────────────────
#  1. Online Imputer (mean imputation)
# ─────────────────────────────────────────────────────────────

class OnlineImputer:
    """Incremental mean imputation for missing (None/NaN) feature values."""

    def __init__(self):
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def learn_one(self, x: dict) -> None:
        for k, v in x.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                self._sums[k] += float(v)
                self._counts[k] += 1

    def transform_one(self, x: dict) -> dict:
        out = {}
        for k, v in x.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                if self._counts[k] > 0:
                    out[k] = self._sums[k] / self._counts[k]
                else:
                    out[k] = 0.0
            else:
                out[k] = float(v)
        return out

    def learn_transform_one(self, x: dict) -> dict:
        self.learn_one(x)
        return self.transform_one(x)


# ─────────────────────────────────────────────────────────────
#  2. Online Min-Max Scaler
# ─────────────────────────────────────────────────────────────

class OnlineMinMaxScaler:
    """Per-feature online min-max normalization to [0, 1]."""

    def __init__(self):
        self._min: dict[str, float] = {}
        self._max: dict[str, float] = {}

    def learn_one(self, x: dict) -> None:
        for k, v in x.items():
            v = float(v)
            if k not in self._min:
                self._min[k] = v
                self._max[k] = v
            else:
                if v < self._min[k]:
                    self._min[k] = v
                if v > self._max[k]:
                    self._max[k] = v

    def transform_one(self, x: dict) -> dict:
        out = {}
        for k, v in x.items():
            v = float(v)
            lo = self._min.get(k, v)
            hi = self._max.get(k, v)
            if hi == lo:
                out[k] = 0.0
            else:
                out[k] = (v - lo) / (hi - lo)
        return out

    def learn_transform_one(self, x: dict) -> dict:
        self.learn_one(x)
        return self.transform_one(x)


# ─────────────────────────────────────────────────────────────
#  3. Incremental One-Hot Encoder (for categorical features)
# ─────────────────────────────────────────────────────────────

class OnlineCategoricalEncoder:
    """
    Detects string-valued features and one-hot encodes them incrementally.
    Numeric features pass through unchanged.
    """

    def __init__(self):
        self._categories: dict[str, set] = defaultdict(set)
        self._is_categorical: dict[str, bool] = {}

    def _detect(self, k: str, v) -> bool:
        if k in self._is_categorical:
            return self._is_categorical[k]
        is_cat = isinstance(v, str)
        self._is_categorical[k] = is_cat
        return is_cat

    def learn_one(self, x: dict) -> None:
        for k, v in x.items():
            if self._detect(k, v):
                self._categories[k].add(v)

    def transform_one(self, x: dict) -> dict:
        out = {}
        for k, v in x.items():
            if self._is_categorical.get(k, False):
                for cat in self._categories[k]:
                    out[f"{k}__{cat}"] = 1.0 if v == cat else 0.0
            else:
                out[k] = v
        return out

    def learn_transform_one(self, x: dict) -> dict:
        self.learn_one(x)
        return self.transform_one(x)


# ─────────────────────────────────────────────────────────────
#  4. Mini-batch K-Means Augmentor
# ─────────────────────────────────────────────────────────────

class MiniBatchKMeansAugmentor:
    """
    Maintains a sliding buffer of recent samples.  Once the buffer has
    enough data the KMeans model is updated and cluster-distance features
    are appended to each sample.
    """

    def __init__(self, n_clusters: int = KMEANS_CLUSTERS,
                 batch_size: int = KMEANS_BATCH_SIZE):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self._buffer: list[list[float]] = []
        self._feature_names: list[str] | None = None
        self._kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=42,
            n_init="auto",
        )
        self._fitted = False

    def _to_array(self, x: dict) -> np.ndarray:
        if self._feature_names is None:
            self._feature_names = sorted(x.keys())
        return np.array([x.get(k, 0.0) for k in self._feature_names], dtype=np.float32)

    def learn_one(self, x: dict) -> None:
        arr = self._to_array(x)
        self._buffer.append(arr)
        if len(self._buffer) >= self.batch_size:
            batch = np.stack(self._buffer)
            self._kmeans.partial_fit(batch)
            self._buffer = []
            self._fitted = True

    def transform_one(self, x: dict) -> dict:
        if not self._fitted:
            return x
        arr = self._to_array(x).reshape(1, -1)
        centers = self._kmeans.cluster_centers_  # (n_clusters, n_features)
        dists = np.linalg.norm(centers - arr, axis=1)  # (n_clusters,)
        out = dict(x)
        for i, d in enumerate(dists):
            out[f"__km_dist_{i}"] = float(d)
        return out

    def learn_transform_one(self, x: dict) -> dict:
        self.learn_one(x)
        return self.transform_one(x)


# ─────────────────────────────────────────────────────────────
#  5. Online SMOTE
# ─────────────────────────────────────────────────────────────

class OnlineSMOTE:
    """
    Streaming minority-class oversampler.

    Maintains a reservoir of recent minority-class samples.  When the
    online class ratio becomes imbalanced, synthetic samples are generated
    by linear interpolation between two random reservoir members and
    yielded alongside the original sample.
    """

    def __init__(self, k: int = SMOTE_K_NEIGHBOURS,
                 reservoir_size: int = SMOTE_RESERVOIR_SIZE,
                 target_ratio: float = 1.0):
        self.k = k
        self.reservoir_size = reservoir_size
        self.target_ratio = target_ratio
        self._reservoirs: dict[int, list] = {0: [], 1: []}
        self._counts: dict[int, int] = {0: 0, 1: 0}

    def _add_to_reservoir(self, label: int, x: dict) -> None:
        r = self._reservoirs[label]
        r.append(x)
        if len(r) > self.reservoir_size:
            # Random replacement to keep reservoir fresh
            idx = random.randint(0, len(r) - 1)
            r[idx] = r[-1]
            r.pop()

    def _synthesize(self, label: int) -> dict | None:
        r = self._reservoirs[label]
        if len(r) < 2:
            return None
        a, b = random.sample(r, 2)
        lam = random.random()
        all_keys = set(a.keys()) | set(b.keys())
        return {
            k: lam * a.get(k, 0.0) + (1 - lam) * b.get(k, 0.0)
            for k in all_keys
        }

    def process(self, x: dict, y: int) -> list[tuple[dict, int]]:
        """
        Returns a list of (sample, label) pairs.  Usually just [(x, y)],
        but may include synthetic oversamples when the minority is underrepresented.
        """
        self._counts[y] += 1
        self._add_to_reservoir(y, x)

        results = [(x, y)]

        maj = max(self._counts, key=lambda k: self._counts[k])
        minority = 1 - maj
        maj_count = self._counts[maj]
        min_count = self._counts[minority]

        if maj_count == 0:
            return results

        ratio = min_count / maj_count
        if ratio < self.target_ratio:
            # How many synthetic samples to generate
            n_needed = int((maj_count * self.target_ratio - min_count))
            n_synth = min(n_needed, 3)  # cap per step to avoid explosion
            for _ in range(n_synth):
                syn = self._synthesize(minority)
                if syn is not None:
                    results.append((syn, minority))
                    self._counts[minority] += 1

        return results


# ─────────────────────────────────────────────────────────────
#  6. Online Information Gain Feature Selector
# ─────────────────────────────────────────────────────────────

class OnlineInfoGainSelector:
    """
    Tracks per-feature Information Gain online using running entropy
    estimates.  Features with IG below threshold are flagged for removal.

    Uses a simple discretisation: each numeric value is binned into one of
    B equal-width bins based on the observed [min, max] range.
    """

    def __init__(self, threshold: float = INFO_GAIN_THRESHOLD, n_bins: int = 10):
        self.threshold = threshold
        self.n_bins = n_bins
        # Class counts for H(Y)
        self._class_counts: dict[int, int] = defaultdict(int)
        self._total = 0
        # Per feature: bin × class joint counts
        self._joint: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        self._feat_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._feat_min: dict[str, float] = {}
        self._feat_max: dict[str, float] = {}
        self._dropped: set[str] = set()

    def _bin(self, k: str, v: float) -> int:
        lo = self._feat_min.get(k, v)
        hi = self._feat_max.get(k, v)
        if hi == lo:
            return 0
        b = int((v - lo) / (hi - lo) * self.n_bins)
        return min(b, self.n_bins - 1)

    def learn_one(self, x: dict, y: int) -> None:
        self._class_counts[y] += 1
        self._total += 1
        for k, v in x.items():
            v = float(v)
            if k not in self._feat_min:
                self._feat_min[k] = v
                self._feat_max[k] = v
            else:
                self._feat_min[k] = min(self._feat_min[k], v)
                self._feat_max[k] = max(self._feat_max[k], v)
            b = self._bin(k, v)
            self._joint[k][(b, y)] += 1
            self._feat_counts[k][b] += 1

    def _entropy(self, counts: dict) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        h = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def _info_gain(self, k: str) -> float:
        hy = self._entropy(self._class_counts)
        # H(Y|X) = sum_b P(X=b) * H(Y|X=b)
        hyxk = 0.0
        total_feat = sum(self._feat_counts[k].values())
        if total_feat == 0:
            return 0.0
        for b, cnt_b in self._feat_counts[k].items():
            # Class distribution given this bin
            class_given_b: dict[int, int] = defaultdict(int)
            for (bb, y), c in self._joint[k].items():
                if bb == b:
                    class_given_b[y] += c
            hyxk += (cnt_b / total_feat) * self._entropy(class_given_b)
        return hy - hyxk

    def transform_one(self, x: dict) -> dict:
        """Drop features whose IG is below threshold (if we have enough data)."""
        if self._total < 200:
            return x  # not enough data to estimate IG reliably yet
        return {k: v for k, v in x.items() if k in self._dropped or self._info_gain(k) >= self.threshold}

    def update_dropped(self) -> None:
        """Periodically recompute the set of features to drop."""
        if self._total < 200:
            return
        self._dropped = {k for k in self._feat_counts if self._info_gain(k) < self.threshold}

    def learn_transform_one(self, x: dict, y: int) -> dict:
        self.learn_one(x, y)
        if self._total % 500 == 0:
            self.update_dropped()
        return self.transform_one(x)


# ─────────────────────────────────────────────────────────────
#  7. Online Pearson Correlation Deduplicator
# ─────────────────────────────────────────────────────────────

class OnlinePearsonDeduplicator:
    """
    Removes one of each highly correlated feature pair online.
    Uses Welford's online algorithm for variance and covariance.
    """

    def __init__(self, threshold: float = PEARSON_THRESHOLD):
        self.threshold = threshold
        self._n = 0
        self._mean: dict[str, float] = defaultdict(float)
        self._M2: dict[str, float] = defaultdict(float)      # variance accumulator
        self._cov: dict[tuple, float] = defaultdict(float)   # covariance accumulator
        self._keys: list[str] = []
        self._dropped: set[str] = set()

    def learn_one(self, x: dict) -> None:
        self._n += 1
        keys = sorted(x.keys())
        if not self._keys:
            self._keys = keys
        delta: dict[str, float] = {}
        for k in keys:
            v = float(x.get(k, 0.0))
            old_mean = self._mean[k]
            self._mean[k] += (v - old_mean) / self._n
            delta[k] = v - old_mean
            delta2 = v - self._mean[k]
            self._M2[k] += delta[k] * delta2

        # Update covariance for all pairs
        for i, ki in enumerate(keys):
            for kj in keys[i + 1:]:
                vi = float(x.get(ki, 0.0))
                vj = float(x.get(kj, 0.0))
                di = vi - self._mean[ki]
                dj = vj - self._mean[kj]
                self._cov[(ki, kj)] += (di * dj - self._cov[(ki, kj)]) / self._n

    def _pearson(self, ki: str, kj: str) -> float:
        cov = self._cov.get((ki, kj), self._cov.get((kj, ki), 0.0))
        var_i = self._M2.get(ki, 0.0) / max(self._n - 1, 1)
        var_j = self._M2.get(kj, 0.0) / max(self._n - 1, 1)
        denom = math.sqrt(var_i * var_j)
        if denom == 0:
            return 0.0
        return cov / denom

    def update_dropped(self) -> None:
        if self._n < 300:
            return
        new_dropped: set[str] = set()
        keys = self._keys
        for i, ki in enumerate(keys):
            if ki in new_dropped:
                continue
            for kj in keys[i + 1:]:
                if kj in new_dropped:
                    continue
                if abs(self._pearson(ki, kj)) >= self.threshold:
                    new_dropped.add(kj)  # drop the second of the pair
        self._dropped = new_dropped

    def transform_one(self, x: dict) -> dict:
        if self._n < 300:
            return x
        return {k: v for k, v in x.items() if k not in self._dropped}

    def learn_transform_one(self, x: dict) -> dict:
        self.learn_one(x)
        if self._n % 500 == 0:
            self.update_dropped()
        return self.transform_one(x)


# ─────────────────────────────────────────────────────────────
#  Full Pipeline
# ─────────────────────────────────────────────────────────────

class OnlinePreprocessingPipeline:
    """
    Chains all preprocessing steps in order:
      encoder → imputer → scaler → kmeans_augmentor →
      info_gain_selector → pearson_deduplicator

    SMOTE is handled separately so the caller receives a list of
    (processed_x, y) pairs per input sample.
    """

    def __init__(self):
        self.encoder     = OnlineCategoricalEncoder()
        self.imputer     = OnlineImputer()
        self.scaler      = OnlineMinMaxScaler()
        self.kmeans      = MiniBatchKMeansAugmentor()
        self.smote       = OnlineSMOTE()
        self.ig_selector = OnlineInfoGainSelector()
        self.pearson     = OnlinePearsonDeduplicator()

    def learn_transform_one(
        self, x: dict, y: int
    ) -> list[tuple[dict, int]]:
        """
        Process a single raw sample.  Returns a list of (processed_x, y)
        because SMOTE may expand one input into multiple outputs.
        """
        # Step 1: encode categorical features
        x = self.encoder.learn_transform_one(x)
        # Step 2: impute missing values
        x = self.imputer.learn_transform_one(x)
        # Step 3: normalize
        x = self.scaler.learn_transform_one(x)
        # Step 4: k-means cluster distance augmentation
        x = self.kmeans.learn_transform_one(x)
        # Step 5: SMOTE balancing — may produce extra samples
        samples = self.smote.process(x, y)
        # Steps 6 & 7: feature selection per produced sample
        results = []
        for sx, sy in samples:
            sx = self.ig_selector.learn_transform_one(sx, sy)
            sx = self.pearson.learn_transform_one(sx)
            results.append((sx, sy))
        return results

    def transform_one(self, x: dict) -> dict:
        """
        Transform only (no learning) — used for adversarial samples that
        should not update pipeline statistics.
        """
        x = self.encoder.transform_one(x)
        x = self.imputer.transform_one(x)
        x = self.scaler.transform_one(x)
        x = self.kmeans.transform_one(x)
        x = self.ig_selector.transform_one(x)
        x = self.pearson.transform_one(x)
        return x

    def dict_to_array(self, x: dict, feature_names: list[str]) -> np.ndarray:
        """Convert feature dict to ordered numpy array using a canonical feature list."""
        return np.array([x.get(k, 0.0) for k in feature_names], dtype=np.float32)

    def array_to_dict(self, arr: np.ndarray, feature_names: list[str]) -> dict:
        """Convert ordered numpy array back to feature dict."""
        return {k: float(v) for k, v in zip(feature_names, arr)}
