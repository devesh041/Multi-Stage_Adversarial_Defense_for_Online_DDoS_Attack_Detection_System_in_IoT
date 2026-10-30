"""
evaluation.py — Phase 4: Evaluation loop and metrics calculation.

Three evaluation scenarios:
  1. Original   — clean data, no adversarial injection, no RADP
  2. Under Attack — adversarial injection, no RADP
  3. Recovered  — adversarial injection + RADP active

Metrics per scenario:
  - Accuracy, Precision, Recall, F1-score
  - Learning Time (ms)
  - Throughput (samples/second)
"""

from __future__ import annotations

import time
import copy
import os
import csv
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from msaad.config import (
    MAX_SAMPLES_PER_SCENARIO, WARMUP_SAMPLES, MINI_BATCH_SIZE,
    RESULTS_DIR,
)
from msaad.preprocessing import OnlinePreprocessingPipeline
from msaad.adversarial import AdversarialAttackGenerator
from msaad.radp import RADPLayer
from msaad.classifiers import Layer2Classifiers
from msaad.mabtse import MABTSERouter


# ─────────────────────────────────────────────────────────────
#  Metrics calculation
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    elapsed_sec: float,
    learn_time_ms: float,
) -> dict:
    """
    Compute classification and throughput metrics.

    Returns a dict with keys:
      accuracy, precision, recall, f1,
      learn_time_ms, throughput_sps (samples per second)
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "learn_time_ms": 0.0, "throughput_sps": 0.0,
        }

    n = len(y_true)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    tput = n / max(elapsed_sec, 1e-9)

    return {
        "accuracy":        round(acc, 6),
        "precision":       round(prec, 6),
        "recall":          round(rec, 6),
        "f1":              round(f1, 6),
        "learn_time_ms":   round(learn_time_ms, 4),
        "throughput_sps":  round(tput, 2),
    }


# ─────────────────────────────────────────────────────────────
#  MSAAD System
# ─────────────────────────────────────────────────────────────

class MSAADSystem:
    """
    Full MSAAD system encapsulating all phases.
    Supports warm-up and three evaluation scenarios.
    """

    def __init__(self, feature_names: Optional[list[str]] = None):
        self.pipeline    = OnlinePreprocessingPipeline()
        self.atk_gen     = AdversarialAttackGenerator()
        self.feature_names: Optional[list[str]] = feature_names  # set after warm-up
        self.radp:  Optional[RADPLayer]      = None
        self.layer2: Optional[Layer2Classifiers] = None
        self.router: Optional[MABTSERouter]  = None
        self._warmup_clean:  list[np.ndarray] = []
        self._warmup_adv:    list[np.ndarray] = []
        self._warmup_labels: list[int]        = []

    def _dict_to_array(self, x: dict) -> np.ndarray:
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())
        return np.array(
            [x.get(k, 0.0) for k in self.feature_names], dtype=np.float32
        )

    def _array_to_dict(self, arr: np.ndarray) -> dict:
        if self.feature_names is None:
            raise RuntimeError("feature_names not set")
        return {k: float(v) for k, v in zip(self.feature_names, arr)}

    # ── Warm-up ───────────────────────────────────────────────

    def warmup(self, stream, n_samples: int = WARMUP_SAMPLES) -> None:
        """
        Process warm-up samples:
          - Seed the preprocessing pipeline
          - Build the ART surrogate model
          - Collect clean + adversarial arrays for RADP pre-training
          - Initialize Layer 2 classifiers and LinUCB router
        """
        print(f"[WARMUP] Starting warm-up on {n_samples} samples...")
        count = 0
        for raw_x, y in stream:
            if count >= n_samples:
                break
            # Pipeline: learn + transform
            processed = self.pipeline.learn_transform_one(raw_x, y)
            for proc_x, proc_y in processed:
                arr = self._dict_to_array(proc_x)
                self.atk_gen.update(arr, proc_y)
                self._warmup_clean.append(arr)
                self._warmup_labels.append(proc_y)
                count += 1
                if count >= n_samples:
                    break

        print(f"[WARMUP] Collected {count} clean samples. Generating adversarial warmup samples...")

        # Generate adversarial samples for RADP warm-up
        if len(self._warmup_clean) > 0:
            X_clean = np.stack(self._warmup_clean)
            # Use FGSM for quick warmup adversarial generation
            batch_size = min(MINI_BATCH_SIZE, len(X_clean))
            X_adv_batches = []
            for i in range(0, len(X_clean), batch_size):
                batch = X_clean[i: i + batch_size]
                if self.atk_gen.is_ready:
                    X_adv_batches.append(
                        self.atk_gen.generate(batch, method="fgsm")
                    )
                else:
                    noise = np.random.uniform(-0.1, 0.1, batch.shape).astype(np.float32)
                    X_adv_batches.append(np.clip(batch + noise, 0.0, 1.0))
            X_adv = np.vstack(X_adv_batches) if X_adv_batches else X_clean.copy()
            self._warmup_adv = list(X_adv)

        # Set stable feature names from warmup
        if self.feature_names is None and self._warmup_clean:
            raise RuntimeError("Feature names not resolved during warmup")

        feature_dim = len(self.feature_names) if self.feature_names else 1

        # Initialize RADP
        print(f"[WARMUP] Initializing RADP (feature_dim={feature_dim})...")
        self.radp = RADPLayer(feature_dim=feature_dim)
        if len(self._warmup_clean) > 0 and len(self._warmup_adv) > 0:
            X_c = np.stack(self._warmup_clean[:len(self._warmup_adv)])
            X_a = np.stack(self._warmup_adv[:len(self._warmup_clean)])
            self.radp.warmup(X_c, X_a)

        # Initialize Layer 2 classifiers
        print("[WARMUP] Initializing Layer 2 classifiers...")
        self.layer2 = Layer2Classifiers()
        for arr, lbl in zip(self._warmup_clean, self._warmup_labels):
            x_dict = self._array_to_dict(arr)
            self.layer2.learn_one(x_dict, lbl)

        # Initialize LinUCB router
        print("[WARMUP] Initializing LinUCB router...")
        self.router = MABTSERouter(self.layer2)
        print("[WARMUP] Complete.\n")

    # ── Scenario: Original ────────────────────────────────────

    def run_original(
        self, stream, max_samples: int = MAX_SAMPLES_PER_SCENARIO
    ) -> dict:
        """
        Scenario 1: Baseline performance on clean streaming data.
        No adversarial injection. No RADP.
        """
        print(f"[SCENARIO 1/3] Original — evaluating on {max_samples} clean samples...")
        y_true, y_pred = [], []
        total_learn_ms = 0.0
        t_start = time.perf_counter()
        count = 0

        for raw_x, y in stream:
            if count >= max_samples:
                break
            processed = self.pipeline.learn_transform_one(raw_x, y)
            for proc_x, proc_y in processed:
                x_dict = proc_x
                t0 = time.perf_counter()
                pred, arm = self.router.learn_one(x_dict, proc_y)
                total_learn_ms += (time.perf_counter() - t0) * 1000
                y_true.append(proc_y)
                y_pred.append(pred)
                # Update surrogate for ART
                arr = self._dict_to_array(x_dict)
                self.atk_gen.update(arr, proc_y)
                count += 1
                if count >= max_samples:
                    break

        elapsed = time.perf_counter() - t_start
        avg_learn_ms = total_learn_ms / max(count, 1)
        return compute_metrics(y_true, y_pred, elapsed, avg_learn_ms)

    # ── Scenario: Under Attack ────────────────────────────────

    def run_under_attack(
        self,
        stream,
        attack_method: str = "fgsm",
        max_samples: int = MAX_SAMPLES_PER_SCENARIO,
    ) -> dict:
        """
        Scenario 2: Adversarial perturbations injected, no RADP defense.
        All samples are perturbed before being passed to classifiers.
        """
        print(f"[SCENARIO 2/3] Under Attack ({attack_method}) — evaluating on {max_samples} samples...")
        y_true, y_pred = [], []
        total_learn_ms = 0.0
        t_start = time.perf_counter()
        count = 0

        for raw_x, y in stream:
            if count >= max_samples:
                break
            # Preprocess (no learning update — use transform only to avoid polluting stats)
            proc_x = self.pipeline.transform_one(raw_x)
            arr = self._dict_to_array(proc_x)
            arr = arr.reshape(1, -1)

            # Inject adversarial perturbation
            arr_adv = self.atk_gen.generate(arr, method=attack_method)
            x_adv_dict = self._array_to_dict(arr_adv[0])

            t0 = time.perf_counter()
            pred, arm = self.router.learn_one(x_adv_dict, y)
            total_learn_ms += (time.perf_counter() - t0) * 1000

            y_true.append(y)
            y_pred.append(pred)
            count += 1
            if count >= max_samples:
                break

        elapsed = time.perf_counter() - t_start
        avg_learn_ms = total_learn_ms / max(count, 1)
        return compute_metrics(y_true, y_pred, elapsed, avg_learn_ms)

    # ── Scenario: Recovered ───────────────────────────────────

    def run_recovered(
        self,
        stream,
        attack_method: str = "fgsm",
        max_samples: int = MAX_SAMPLES_PER_SCENARIO,
    ) -> dict:
        """
        Scenario 3: Adversarial perturbations injected, RADP active.
        RADP prunes detected adversarial samples before classifiers see them.
        """
        print(f"[SCENARIO 3/3] Recovered ({attack_method}) — evaluating on {max_samples} samples...")
        y_true, y_pred = [], []
        total_learn_ms = 0.0
        t_start = time.perf_counter()
        count = 0
        pruned = 0

        for raw_x, y in stream:
            if count >= max_samples:
                break
            proc_x = self.pipeline.transform_one(raw_x)
            arr = self._dict_to_array(proc_x)
            arr_batch = arr.reshape(1, -1)

            # Inject adversarial perturbation
            arr_adv = self.atk_gen.generate(arr_batch, method=attack_method)
            arr_adv_flat = arr_adv[0]

            # RADP: detect and potentially prune
            is_pruned, adv_proba = self.radp.process(
                arr_adv_flat, is_adversarial_truth=1
            )

            # Also update RAD on clean sample
            self.radp.rad.learn_one(arr, 0)

            # Update GAN buffers occasionally
            if count % MINI_BATCH_SIZE == 0:
                self.radp.online_update_gan(arr_batch, arr_adv)

            t0 = time.perf_counter()
            if is_pruned:
                # Pruned: Layer 1 RADP detected adversarial attack.
                # Flagged as attack (DDoS=1) and filtered/dropped to protect Layer 2.
                pruned += 1
                pred = 1
                learn_time = (time.perf_counter() - t0) * 1000
            else:
                # Evaded Layer 1: passed through to Layer 2/3 ensemble
                x_dict = self._array_to_dict(arr_adv_flat)
                pred, arm = self.router.learn_one(x_dict, y)
                learn_time = (time.perf_counter() - t0) * 1000

            total_learn_ms += learn_time
            y_true.append(y)
            y_pred.append(pred)
            count += 1
            if count >= max_samples:
                break

        elapsed = time.perf_counter() - t_start
        avg_learn_ms = total_learn_ms / max(count, 1)
        metrics = compute_metrics(y_true, y_pred, elapsed, avg_learn_ms)
        metrics["pruned_samples"] = pruned
        metrics["prune_rate"] = round(pruned / max(count, 1), 4)
        print(f"         RADP pruned {pruned}/{count} samples ({metrics['prune_rate']*100:.1f}%)")
        return metrics


# ─────────────────────────────────────────────────────────────
#  Results output
# ─────────────────────────────────────────────────────────────

def print_results_table(results: dict[str, dict]) -> None:
    """Pretty-print a table of evaluation results."""
    header_cols = ["Scenario", "Accuracy", "Precision", "Recall", "F1",
                   "Learn(ms)", "Throughput(sps)"]
    col_widths = [22, 10, 10, 10, 10, 12, 16]

    # Header (ASCII-safe for Windows cp1252 terminal)
    header = "  ".join(f"{h:{w}}" for h, w in zip(header_cols, col_widths))
    sep = "  ".join("-" * w for w in col_widths)
    bar = "=" * len(sep)
    print("\n" + bar)
    print("  MSAAD EVALUATION RESULTS")
    print(bar)
    print(header)
    print(sep)

    for scenario, m in results.items():
        row = [
            scenario,
            f"{m.get('accuracy', 0):.4f}",
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"{m.get('f1', 0):.4f}",
            f"{m.get('learn_time_ms', 0):.3f}",
            f"{m.get('throughput_sps', 0):.1f}",
        ]
        print("  ".join(f"{v:{w}}" for v, w in zip(row, col_widths)))

    print(bar + "\n")


def save_results_csv(results: dict[str, dict], dataset_name: str) -> str:
    """Save evaluation results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, f"evaluation_results_{dataset_name}.csv")
    fieldnames = [
        "scenario", "dataset", "accuracy", "precision", "recall", "f1",
        "learn_time_ms", "throughput_sps", "pruned_samples", "prune_rate",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, m in results.items():
            row = {"scenario": scenario, "dataset": dataset_name}
            row.update(m)
            writer.writerow(row)
    print(f"[RESULTS] Saved to: {filepath}")
    return filepath
