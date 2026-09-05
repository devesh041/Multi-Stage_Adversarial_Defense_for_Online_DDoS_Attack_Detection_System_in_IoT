"""
main.py — Entry point for the MSAAD framework.

Usage:
  python -m msaad.main [options]
  python main.py [options]

Options:
  --dataset   {ciciot23, iotid20, cybersecurity, all}  (default: ciciot23)
  --samples   N  samples per scenario                  (default: 10000)
  --warmup    N  warm-up samples                       (default: 500)
  --attack    {fgsm, bim, pgd, cw, all}               (default: fgsm)
  --scenario  {original, attack, recovered, all}       (default: all)
  --verbose       print progress every N samples
"""

from __future__ import annotations

import argparse
import sys
import os
import copy

# Ensure project root is on sys.path when run as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from msaad.config import MAX_SAMPLES_PER_SCENARIO, WARMUP_SAMPLES
from msaad.data_loader import (
    stream_ciciot23, stream_iotid20, stream_cybersecurity
)
from msaad.evaluation import (
    MSAADSystem, print_results_table, save_results_csv
)


# ─────────────────────────────────────────────────────────────
#  Dataset factory
# ─────────────────────────────────────────────────────────────

def get_stream(dataset: str, split: str = "train"):
    """Return a fresh generator for the given dataset."""
    if dataset == "ciciot23":
        return stream_ciciot23(split=split)
    elif dataset == "iotid20":
        return stream_iotid20()
    elif dataset == "cybersecurity":
        return stream_cybersecurity()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


# ─────────────────────────────────────────────────────────────
#  Main evaluation loop
# ─────────────────────────────────────────────────────────────

def run_evaluation(
    dataset: str,
    samples: int,
    warmup_n: int,
    attack_method: str,
    scenario: str,
) -> None:
    print("=" * 65)
    print(f"  MSAAD - Multi-Stage Adversarial Attack Defense Framework")
    print(f"  Dataset: {dataset.upper()}  |  Samples/scenario: {samples}")
    print(f"  Attack:  {attack_method.upper()}  |  Scenario:         {scenario}")
    print("=" * 65 + "\n")

    # ── Step 1: Warm-up ──────────────────────────────────────
    system = MSAADSystem()
    warmup_stream = get_stream(dataset, split="train")
    system.warmup(warmup_stream, n_samples=warmup_n)

    results: dict[str, dict] = {}
    attacks = ["fgsm", "bim", "pgd", "cw"] if attack_method == "all" else [attack_method]

    for atk in attacks:
        atk_label = atk.upper()
        print(f"\n{'-'*55}")
        print(f"  Attack Method: {atk_label}")
        print(f"{'-'*55}")

        # ── Scenario 1: Original ─────────────────────────────
        if scenario in ("original", "all"):
            sys1 = copy.deepcopy(system)
            stream_orig = get_stream(dataset, split="test")
            m = sys1.run_original(stream_orig, max_samples=samples)
            results[f"Original ({atk_label})"] = m

        # ── Scenario 2: Under Attack ─────────────────────────
        if scenario in ("attack", "all"):
            sys2 = copy.deepcopy(system)
            stream_atk = get_stream(dataset, split="test")
            m = sys2.run_under_attack(stream_atk, attack_method=atk, max_samples=samples)
            results[f"Under Attack ({atk_label})"] = m

        # ── Scenario 3: Recovered ────────────────────────────
        if scenario in ("recovered", "all"):
            sys3 = copy.deepcopy(system)
            stream_rec = get_stream(dataset, split="test")
            m = sys3.run_recovered(stream_rec, attack_method=atk, max_samples=samples)
            results[f"Recovered ({atk_label})"] = m

    # ── Output ───────────────────────────────────────────────
    print_results_table(results)
    save_results_csv(results, dataset_name=dataset)

    # MAB arm summary (from last system instance)
    try:
        print("  MAB Arm Statistics:")
        for arm_name, stats in sys3.router.arm_summary.items():
            print(f"    {arm_name:15s}  pulls={stats['pulls']:5d}  "
                  f"success_rate={stats['success_rate']:.4f}")
    except Exception:
        pass
    print()


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MSAAD: Multi-Stage Adversarial Attack Defense for IoT DDoS detection"
    )
    p.add_argument(
        "--dataset", choices=["ciciot23", "iotid20", "cybersecurity", "all"],
        default="ciciot23",
        help="Dataset to use (default: ciciot23)",
    )
    p.add_argument(
        "--samples", type=int, default=MAX_SAMPLES_PER_SCENARIO,
        help=f"Samples per evaluation scenario (default: {MAX_SAMPLES_PER_SCENARIO})",
    )
    p.add_argument(
        "--warmup", type=int, default=WARMUP_SAMPLES,
        help=f"Warm-up samples (default: {WARMUP_SAMPLES})",
    )
    p.add_argument(
        "--attack", choices=["fgsm", "bim", "pgd", "cw", "all"],
        default="fgsm",
        help="Adversarial attack method (default: fgsm)",
    )
    p.add_argument(
        "--scenario", choices=["original", "attack", "recovered", "all"],
        default="all",
        help="Which scenario(s) to evaluate (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    datasets = (
        ["ciciot23", "iotid20", "cybersecurity"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for ds in datasets:
        run_evaluation(
            dataset=ds,
            samples=args.samples,
            warmup_n=args.warmup,
            attack_method=args.attack,
            scenario=args.scenario,
        )


if __name__ == "__main__":
    main()
