"""
config.py — Central configuration for the MSAAD framework.
All paths, hyperparameters, and constants live here.
"""

import os

# ─────────────────────────────────────────────
#  Root project path
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
#  Dataset paths
# ─────────────────────────────────────────────
CICIOT23_TRAIN      = os.path.join(PROJECT_ROOT, "CICIOT23", "train",      "train.csv")
CICIOT23_TEST       = os.path.join(PROJECT_ROOT, "CICIOT23", "test",       "test.csv")
CICIOT23_VAL        = os.path.join(PROJECT_ROOT, "CICIOT23", "validation", "validation.csv")
IOTID20_MAIN        = os.path.join(PROJECT_ROOT, "IoTID20",  "IoT Network Intrusion Dataset.csv")
IOTID20_CYBER       = os.path.join(PROJECT_ROOT, "IoTID20",  "archive (3)", "cybersecurity_attacks.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Label mappings → binary {0: Normal, 1: DDoS}
# ─────────────────────────────────────────────

# CICIoT23: label column is 'label'
CICIOT23_LABEL_COL = "label"
CICIOT23_NORMAL_LABELS = {"BenignTraffic"}
# Any label starting with DDoS or DoS or Mirai maps to 1
def ciciot23_label_to_binary(label: str) -> int | None:
    label = str(label).strip()
    if label in CICIOT23_NORMAL_LABELS:
        return 0
    if label.startswith("DDoS") or label.startswith("DoS") or label.startswith("Mirai"):
        return 1
    return None  # skip: Recon, XSS, etc.

# IoTID20 main: label column is 'Label'
IOTID20_LABEL_COL = "Label"
IOTID20_NORMAL_LABELS = {"Normal"}
def iotid20_label_to_binary(label: str) -> int | None:
    label = str(label).strip()
    if label in IOTID20_NORMAL_LABELS:
        return 0
    # DDoS, DoS, Mirai variants
    if any(k in label for k in ("DDoS", "DoS", "Mirai", "Flooding", "Flood")):
        return 1
    return None  # skip Scan, MITM, etc.

# IoTID20 cybersecurity: label column is 'Attack Type'
CYBER_LABEL_COL = "Attack Type"
CYBER_NORMAL_LABELS = {"Normal"}
def cyber_label_to_binary(label: str) -> int | None:
    label = str(label).strip()
    if label in CYBER_NORMAL_LABELS:
        return 0
    if "DDoS" in label or "DoS" in label:
        return 1
    return None

# IoTID20 metadata columns to drop (non-feature)
IOTID20_DROP_COLS = {"Flow_ID", "Src_IP", "Dst_IP", "Timestamp"}

# ─────────────────────────────────────────────
#  Streaming / evaluation settings
# ─────────────────────────────────────────────
MAX_SAMPLES_PER_SCENARIO = 10_000   # samples per scenario (Original / Under Attack / Recovered)
WARMUP_SAMPLES           = 500      # warm-up phase before evaluation
MINI_BATCH_SIZE          = 32       # batch size for GAN updates and C&W attack
STREAM_CHUNK_SIZE        = 1_000    # CSV reading chunk size (rows per pandas chunk)

# ─────────────────────────────────────────────
#  Phase 1 — Preprocessing
# ─────────────────────────────────────────────
KMEANS_CLUSTERS          = 8        # mini-batch K-means clusters
KMEANS_BATCH_SIZE        = 256      # mini-batch K-means update batch
INFO_GAIN_THRESHOLD      = 0.01     # drop features below this IG
PEARSON_THRESHOLD        = 0.95     # drop one of a pair above this correlation
SMOTE_K_NEIGHBOURS       = 5        # online SMOTE interpolation neighbours
SMOTE_RESERVOIR_SIZE     = 200      # max stored minority samples

# ─────────────────────────────────────────────
#  Phase 2 — Adversarial attacks (ART)
# ─────────────────────────────────────────────
ADV_EPSILON              = 0.1
BIM_EPS_STEP             = 0.01
BIM_MAX_ITER             = 10
PGD_EPS_STEP             = 0.01
PGD_MAX_ITER             = 10
CW_CONFIDENCE            = 0.1
CW_MAX_ITER              = 20
ART_BUFFER_SIZE          = 512      # sliding window for ART surrogate model training

# ─────────────────────────────────────────────
#  Phase 3 Layer 1 — GAN / RAD
# ─────────────────────────────────────────────
GAN_NOISE_DIM            = 32
GAN_HIDDEN_DIM           = 64
GAN_LR                   = 1e-3
GAN_WARMUP_EPOCHS        = 5
RAD_THRESHOLD            = 0.5      # adversarial detection threshold

# ─────────────────────────────────────────────
#  Phase 3 Layer 2 — Base classifiers
# ─────────────────────────────────────────────
ARF_N_MODELS             = 10
SRP_N_MODELS             = 10
KNN_N_NEIGHBORS          = 5
KNN_WINDOW_SIZE          = 500

# ─────────────────────────────────────────────
#  Phase 3 Layer 3 — Thompson Sampling MAB
# ─────────────────────────────────────────────
MAB_ALPHA_INIT           = 1.0      # Beta prior alpha
MAB_BETA_INIT            = 1.0      # Beta prior beta
MAB_ARM_NAMES            = ["ARF-ADWIN", "SRP-DDM", "KNN-ADWIN", "Ensemble"]
AUWPAE_WINDOW_SIZE       = 200      # sliding window for per-classifier accuracy weighting
