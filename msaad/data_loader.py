"""
data_loader.py — Streaming CSV readers for CICIoT23 and IoTID20 datasets.

Each generator yields (features_dict, binary_label) tuples, filtering rows to
Normal (0) and DDoS (1) only, skipping all other attack categories.
"""

from __future__ import annotations

import csv
import os
from typing import Generator, Optional

from msaad.config import (
    CICIOT23_TRAIN, CICIOT23_TEST, CICIOT23_VAL,
    IOTID20_MAIN, IOTID20_CYBER,
    CICIOT23_LABEL_COL, IOTID20_LABEL_COL, CYBER_LABEL_COL,
    IOTID20_DROP_COLS,
    ciciot23_label_to_binary,
    iotid20_label_to_binary,
    cyber_label_to_binary,
)

# ─────────────────────────────────────────────────────────────
#  Generic helpers
# ─────────────────────────────────────────────────────────────

def _parse_float(val: str) -> Optional[float]:
    """Try to parse a string as float; return None on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _row_to_features(
    row: dict,
    label_col: str,
    drop_cols: set[str] | None = None,
) -> dict[str, Optional[float]]:
    """
    Convert a CSV row dict to a feature dict, excluding the label column
    and any explicitly dropped columns. Values are cast to float (None if
    unparseable — handled downstream by the imputer).
    """
    drop = drop_cols or set()
    drop = drop | {label_col}
    return {
        k: _parse_float(v)
        for k, v in row.items()
        if k not in drop
    }


# ─────────────────────────────────────────────────────────────
#  CICIoT23 streams
# ─────────────────────────────────────────────────────────────

def stream_ciciot23(
    split: str = "train",
    max_samples: int | None = None,
) -> Generator[tuple[dict, int], None, None]:
    """
    Stream (features, label) from the CICIoT23 dataset.

    Args:
        split: one of "train", "test", "validation"
        max_samples: stop after this many *accepted* rows (None = unlimited)
    """
    paths = {
        "train":      CICIOT23_TRAIN,
        "test":       CICIOT23_TEST,
        "validation": CICIOT23_VAL,
    }
    path = paths[split]
    if not os.path.exists(path):
        raise FileNotFoundError(f"CICIoT23 {split} file not found: {path}")

    count = 0
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_label = row.get(CICIOT23_LABEL_COL, "")
            binary = ciciot23_label_to_binary(raw_label)
            if binary is None:
                continue  # skip non-DDoS/Normal classes
            features = _row_to_features(row, CICIOT23_LABEL_COL)
            yield features, binary
            count += 1
            if max_samples is not None and count >= max_samples:
                return


# ─────────────────────────────────────────────────────────────
#  IoTID20 streams
# ─────────────────────────────────────────────────────────────

def stream_iotid20(
    max_samples: int | None = None,
) -> Generator[tuple[dict, int], None, None]:
    """
    Stream (features, label) from the IoT Network Intrusion Dataset.
    Metadata columns (IP addresses, timestamps, Flow_ID) are dropped.
    """
    if not os.path.exists(IOTID20_MAIN):
        raise FileNotFoundError(f"IoTID20 main file not found: {IOTID20_MAIN}")

    count = 0
    with open(IOTID20_MAIN, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_label = row.get(IOTID20_LABEL_COL, "")
            binary = iotid20_label_to_binary(raw_label)
            if binary is None:
                continue
            features = _row_to_features(row, IOTID20_LABEL_COL, IOTID20_DROP_COLS)
            yield features, binary
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def stream_cybersecurity(
    max_samples: int | None = None,
) -> Generator[tuple[dict, int], None, None]:
    """
    Stream (features, label) from the cybersecurity_attacks.csv file.
    Only numeric columns are kept; text/categorical fields are dropped.
    """
    if not os.path.exists(IOTID20_CYBER):
        raise FileNotFoundError(f"Cybersecurity file not found: {IOTID20_CYBER}")

    # Numeric columns in cybersecurity_attacks.csv
    numeric_cols = {
        "Source Port", "Destination Port", "Protocol",
        "Packet Length", "Anomaly Scores",
    }
    drop_cols = {CYBER_LABEL_COL}

    count = 0
    with open(IOTID20_CYBER, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_label = row.get(CYBER_LABEL_COL, "")
            binary = cyber_label_to_binary(raw_label)
            if binary is None:
                continue
            # Keep only numeric-ish columns to avoid polluting the feature space
            features = {
                k: _parse_float(v)
                for k, v in row.items()
                if k in numeric_cols
            }
            yield features, binary
            count += 1
            if max_samples is not None and count >= max_samples:
                return


# ─────────────────────────────────────────────────────────────
#  Unified interleaved stream
# ─────────────────────────────────────────────────────────────

def unified_stream(
    datasets: list[str] | None = None,
    max_samples: int | None = None,
) -> Generator[tuple[dict, int], None, None]:
    """
    Round-robin interleave chosen dataset streams.

    Args:
        datasets: list subset of ["ciciot23", "iotid20", "cybersecurity"]
                  Defaults to ["ciciot23", "iotid20"].
        max_samples: total samples to yield (None = unlimited)
    """
    if datasets is None:
        datasets = ["ciciot23", "iotid20"]

    generators = []
    for name in datasets:
        if name == "ciciot23":
            generators.append(stream_ciciot23(split="train", max_samples=None))
        elif name == "iotid20":
            generators.append(stream_iotid20(max_samples=None))
        elif name == "cybersecurity":
            generators.append(stream_cybersecurity(max_samples=None))

    active = list(range(len(generators)))
    count = 0
    while active:
        next_active = []
        for idx in active:
            try:
                item = next(generators[idx])
                yield item
                count += 1
                if max_samples is not None and count >= max_samples:
                    return
                next_active.append(idx)
            except StopIteration:
                pass  # this generator exhausted
        active = next_active
