"""
run_all.py — One-shot runner for the complete MSAAD evaluation.

Runs all datasets x all attack methods x all 3 scenarios.
Logs output to both the terminal AND a timestamped log file.

Usage:
    python run_all.py
    python run_all.py --samples 2000 --warmup 500
    python run_all.py --fast          # quick smoke test (200 samples)
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from io import StringIO

# ── Ensure project root is importable ──────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from msaad.config import MAX_SAMPLES_PER_SCENARIO, WARMUP_SAMPLES, RESULTS_DIR
from msaad.data_loader import stream_ciciot23, stream_iotid20, stream_cybersecurity
from msaad.evaluation import MSAADSystem, print_results_table, save_results_csv


# ─────────────────────────────────────────────────────────────
#  Tee: writes to both stdout and a file simultaneously
# ─────────────────────────────────────────────────────────────

class Tee:
    """Redirect stdout writes to both the real stdout and a file."""

    def __init__(self, filepath: str):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._stdout.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    # Make Tee behave like a real stream
    def fileno(self):
        return self._stdout.fileno()

    def isatty(self):
        return False


# ─────────────────────────────────────────────────────────────
#  Dataset stream factory
# ─────────────────────────────────────────────────────────────

def get_stream(dataset: str, split: str = "train"):
    if dataset == "ciciot23":
        return stream_ciciot23(split=split)
    elif dataset == "iotid20":
        return stream_iotid20()
    elif dataset == "cybersecurity":
        return stream_cybersecurity()
    raise ValueError(f"Unknown dataset: {dataset!r}")


# ─────────────────────────────────────────────────────────────
#  Core evaluation logic
# ─────────────────────────────────────────────────────────────

def run_dataset(
    dataset: str,
    samples: int,
    warmup_n: int,
    attacks: list[str],
) -> dict[str, dict]:
    """
    Run all attack methods and all 3 scenarios for a single dataset.
    Returns a dict keyed by scenario label.
    """
    print("=" * 70)
    print(f"  MSAAD - Dataset: {dataset.upper()}")
    print(f"  Samples/scenario: {samples}  |  Warm-up: {warmup_n}")
    print(f"  Attacks: {', '.join(a.upper() for a in attacks)}")
    print("=" * 70 + "\n")

    # ── Warm-up (shared across all attacks/scenarios)
    system = MSAADSystem()
    warmup_stream = get_stream(dataset, split="train")
    system.warmup(warmup_stream, n_samples=warmup_n)

    all_results: dict[str, dict] = {}

    for atk in attacks:
        atk_label = atk.upper()
        print(f"\n{'-' * 60}")
        print(f"  Attack: {atk_label}")
        print(f"{'-' * 60}")

        # Scenario 1: Original (only run once — not attack-specific)
        orig_key = f"Original ({atk_label})"
        sys1 = copy.deepcopy(system)
        stream_o = get_stream(dataset, split="test")
        all_results[orig_key] = sys1.run_original(stream_o, max_samples=samples)

        # Scenario 2: Under Attack
        atk_key = f"Under Attack ({atk_label})"
        sys2 = copy.deepcopy(system)
        stream_a = get_stream(dataset, split="test")
        all_results[atk_key] = sys2.run_under_attack(
            stream_a, attack_method=atk, max_samples=samples
        )

        # Scenario 3: Recovered (RADP active)
        rec_key = f"Recovered ({atk_label})"
        sys3 = copy.deepcopy(system)
        stream_r = get_stream(dataset, split="test")
        all_results[rec_key] = sys3.run_recovered(
            stream_r, attack_method=atk, max_samples=samples
        )

    return all_results


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MSAAD — run all datasets, all attacks, all scenarios"
    )
    p.add_argument(
        "--samples", type=int, default=MAX_SAMPLES_PER_SCENARIO,
        help=f"Samples per scenario (default: {MAX_SAMPLES_PER_SCENARIO})"
    )
    p.add_argument(
        "--warmup", type=int, default=WARMUP_SAMPLES,
        help=f"Warm-up samples (default: {WARMUP_SAMPLES})"
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Quick smoke test: 200 samples, 150 warmup, FGSM only"
    )
    p.add_argument(
        "--attacks", nargs="+",
        choices=["fgsm", "bim", "pgd", "cw"],
        default=["fgsm", "bim", "pgd", "cw"],
        help="Attack methods to run (default: all four)"
    )
    p.add_argument(
        "--datasets", nargs="+",
        choices=["ciciot23", "iotid20", "cybersecurity"],
        default=["ciciot23", "iotid20"],
        help="Datasets to evaluate (default: ciciot23 iotid20)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Fast mode overrides
    if args.fast:
        samples  = 200
        warmup_n = 150
        attacks  = ["fgsm"]
        datasets = ["ciciot23"]
        print("[INFO] Fast mode enabled: 200 samples, FGSM only, CICIoT23\n")
    else:
        samples  = args.samples
        warmup_n = args.warmup
        attacks  = args.attacks
        datasets = args.datasets

    # ── Create timestamped log file ────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(RESULTS_DIR, f"msaad_run_{timestamp}.log")

    tee = Tee(log_path)
    sys.stdout = tee   # all print() now goes to BOTH terminal and file

    print(f"\n{'#' * 70}")
    print(f"#  MSAAD FULL EVALUATION RUN")
    print(f"#  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}\n")

    global_start = time.perf_counter()
    grand_results: dict[str, dict[str, dict]] = {}

    for ds in datasets:
        t0 = time.perf_counter()
        results = run_dataset(ds, samples, warmup_n, attacks)
        elapsed = time.perf_counter() - t0
        grand_results[ds] = results

        # Print per-dataset table
        print_results_table(results)

        # Save per-dataset CSV
        csv_path = save_results_csv(results, dataset_name=ds)

        print(f"  [TIME] {ds.upper()} completed in {elapsed:.1f}s\n")

    # ── Grand summary table ────────────────────────────────────
    total_elapsed = time.perf_counter() - global_start
    print(f"\n{'#' * 70}")
    print(f"#  ALL EVALUATIONS COMPLETE")
    print(f"#  Total wall-clock time: {total_elapsed:.1f}s")
    print(f"{'#' * 70}\n")

    # Restore stdout before exiting
    sys.stdout = tee._stdout
    tee.close()

    print("\n[DONE] Evaluation completed successfully.")


if __name__ == "__main__":
    main()
