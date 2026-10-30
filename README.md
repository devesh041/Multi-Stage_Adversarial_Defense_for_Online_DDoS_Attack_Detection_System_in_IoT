# Multi-Stage Adversarial Defense for Online DDoS Attack Detection System in IoT (MSAAD)

## 📌 Table of Contents
1. [Overview](#-overview)
2. [Key Highlights](#-key-highlights)
3. [System Workflow](#-system-workflow)
4. [Execution Flow (File-by-File)](#-execution-flow-file-by-file)
5. [Algorithmic Deep Dive](#-algorithmic-deep-dive)
   - [Online Preprocessing Pipeline](#1-online-preprocessing-pipeline)
   - [Adversarial Perturbation Generation (ART & MSAPG)](#2-adversarial-perturbation-generation-art--msapg)
   - [Resilient Adversarial Detection and Purification (RADP)](#3-resilient-adversarial-detection-and-purification-radp)
   - [Base Classifiers with Concept Drift Adaptation](#4-base-classifiers-with-concept-drift-adaptation)
   - [LinUCB Contextual Multi-Armed Bandit with AUWPAE](#5-linucb-contextual-multi-armed-bandit-with-auwpae)
6. [Comparative Algorithmic Analysis](#-comparative-algorithmic-analysis)
7. [Directory Structure](#-directory-structure)
8. [Setup & Installation Instructions](#-setup--installation-instructions)
9. [Running Experiments](#-running-experiments)
10. [Experimental Benchmark Results](#-experimental-benchmark-results)
11. [Authors & Academic Advisory](#-authors--academic-advisory)

---

## 📖 Overview

The **Multi-Stage Adversarial Defense for Online DDoS Attack Detection System in IoT (MSAAD)** framework is an end-to-end, strictly streaming defense and detection system engineered for resource-constrained Internet of Things (IoT) network gateways.

Conventional Machine Learning Intrusion Detection Systems (ML-IDS) suffer from two fundamental vulnerabilities:
1. **Batch Retraining Fallacy:** They operate on static, batch-oriented models that cannot adapt in real time to non-stationary network traffic and continuous concept drift.
2. **Adversarial Susceptibility:** Malicious actors can apply imperceptible, gradient-directed perturbations (e.g., FGSM, BIM, PGD, Carlini & Wagner) to traffic statistics, forcing ML detectors to misclassify malicious floods as benign communication.

**MSAAD resolves both vulnerabilities through a multi-stage, defensive-depth streaming architecture.** Every single component—from categorical encoding, imputation, normalization, and SMOTE balancing to GAN-driven adversarial detection, base classification, and contextual bandit routing—operates in an **incremental (online) paradigm** without batch retraining.

---

## 🌟 Key Highlights

- **100% Streaming Pipeline:** Ingests network packet flows record-by-record ($O(1)$ sample buffer per step).
- **Novel 3-Layer Defense-in-Depth:**
  - **Layer 1 (RADP):** Online Generative Adversarial Network (GAN) + Online Logistic Regression Resilient Adversarial Detector (RAD) to detect and filter adversarial perturbations.
  - **Layer 2 (Drift-Aware Ensemble):** Five streaming classifiers equipped with Adaptive Windowing (ADWIN) and Drift Detection Method (DDM).
  - **Layer 3 (MABLUCBE):** A 6-arm LinUCB Contextual Multi-Armed Bandit that dynamically routes samples based on input feature context, backed by an Accuracy-Updated Weighted Probability Averaging Ensemble (AUWPAE).
- **Multi-Source Adversarial Stress Testing:** Built-in surrogate MLP leveraging IBM's Adversarial Robustness Toolbox (ART) for online injection of **FGSM, BIM, PGD, and C&W ($L_2$)** attacks.
- **Empirically Proven:** Achieves **99.50% – 100.00% F1-score** across the `CICIoT2023` and `IoTID20` benchmark datasets with throughput spanning 50–132 samples/second on commodity hardware.

---

## 🔄 System Workflow

The end-to-end data lifecycle progresses through four major operational phases:

```
                  Raw IoT Network Stream (CSV / PCAP Records)
                                      │
                                      ▼
     ╔═════════════════════════════════════════════════════════════════╗
     ║        PHASE 1: Online Streaming Preprocessing Pipeline        ║
     ║                                                                 ║
     ║  [Categorical] ──> [Imputer] ──> [MinMax] ──> [MiniBatch KMeans]║
     ║   One-Hot           Running Mean  Online Scaler  Cluster Dist   ║
     ║                                                        │        ║
     ║  [Pearson Dedup] <── [Info Gain] <── [Online SMOTE] <──┘        ║
     ║   Welford Covar       Entropy Bins    Reservoir Interp          ║
     ╚════════════════════════════════┬════════════════════════════════╝
                                      │ Preprocessed Feature Vector x ∈ [0, 1]^d
                                      ▼
     ╔═════════════════════════════════════════════════════════════════╗
     ║       PHASE 2: Adversarial Attack Simulation (Stress Test)     ║
     ║                                                                 ║
     ║  PyTorch MLP Surrogate (Sliding Buffer, 50-step Periodic Fit)   ║
     ║   ├── Fast Gradient Sign Method (FGSM)                          ║
     ║   ├── Basic Iterative Method (BIM)                              ║
     ║   ├── Projected Gradient Descent (PGD)                          ║
     ║   ├── Carlini & Wagner L2 (C&W)                                 ║
     ║   └── MSAPG: Multi-Source Adversarial Perturbation Averager     ║
     ╚════════════════════════════════┬════════════════════════════════╝
                                      │ Perturbed Vector x_adv
                                      ▼
     ╔═════════════════════════════════════════════════════════════════╗
     ║           PHASE 3: Three-Layer Detection & Defense Engine       ║
     ║                                                                 ║
     ║  ┌───────────────────────────────────────────────────────────┐  ║
     ║  │ LAYER 1: Resilient Adversarial Detection & Pruning (RADP)  │  ║
     ║  │  Online GAN (G & D) mimics attacks ──> RAD Detector (SGD) │  ║
     ║  │             P(Adversarial) >= 0.5?                        │  ║
     ║  └──────────────┬────────────────────────────┬───────────────┘  ║
     ║                 │ YES                        │ NO               ║
     ║                 ▼                            ▼                  ║
     ║        [Pruned / Flagged]           Passed to Layer 2           ║
     ║        (Immediate Attack Drop)               │                  ║
     ║                                              ▼                  ║
     ║  ┌───────────────────────────────────────────────────────────┐  ║
     ║  │ LAYER 2: Five Online Base Classifiers (Drift-Protected)   │  ║
     ║  │  [ARF-ADWIN] [SRP-DDM] [HT-ADWIN] [KNN-ADWIN] [OLR-SGD]   │  ║
     ║  │             Outputs: P_0, P_1, P_2, P_3, P_4              │  ║
     ║  └───────────────────────────┬───────────────────────────────┘  ║
     ║                              │                                  ║
     ║                              ▼                                  ║
     ║  ┌───────────────────────────────────────────────────────────┐  ║
     ║  │ LAYER 3: LinUCB Contextual Bandit Routing (MABLUCBE)      ║  ║
     ║  │  UCB_a = θ_a^T x + α √(x^T A_a^-1 x)                      ║  ║
     ║  │  Arm 0-4: Direct Model Routing  |  Arm 5: AUWPAE Ensemble ║  ║
     ║  └───────────────────────────┬───────────────────────────────┘  ║
     ╚══════════════════════════════╪══════════════════════════════════╝
                                    │
                                    ▼
             Final Binary Classification: Normal (0) vs. DDoS (1)
                                    │
                                    ▼
       Online Feedback Update: Reward r ∈ {0, 1} -> LinUCB & Classifiers
```

---

## 🗂️ Execution Flow (File-by-File)

The runtime execution flow transitions systematically across modular components:

```
[run_all.py / main.py]
      │
      ├──> Reads CLI arguments & imports system paths from [msaad/config.py]
      │
      ├──> Initializes [msaad/evaluation.py : MSAADSystem]
      │
      ├──> Calls [msaad/data_loader.py : stream_ciciot23() / stream_iotid20()]
      │      └── Yields raw record generator (features_dict, binary_label)
      │
      ├──> [WARMUP PHASE]
      │      ├── Feeds raw stream into [msaad/preprocessing.py : OnlinePreprocessingPipeline]
      │      ├── Updates [msaad/adversarial.py : AdversarialAttackGenerator] surrogate buffer
      │      ├── Generates clean & adversarial warmup arrays
      │      ├── Warms up [msaad/radp.py : RADPLayer] (GAN + RAD detector SGD)
      │      ├── Initializes [msaad/classifiers.py : Layer2Classifiers] (ARF, SRP, HT, KNN, OLR)
      │      └── Initializes [msaad/mabtse.py : MABTSERouter] (LinUCB contextual matrices)
      │
      ├──> [EVALUATION SCENARIOS]
      │      ├── Scenario 1: run_original()
      │      │     └── Preprocess -> Layer 3 Router -> Base Classifiers -> Compute Metrics
      │      │
      │      ├── Scenario 2: run_under_attack()
      │      │     └── Preprocess -> Inject ART Attack (adversarial.py) -> Direct Router -> Drop Metrics
      │      │
      │      └── Scenario 3: run_recovered()
      │            └── Preprocess -> Inject ART Attack -> RADPLayer.process()
      │                  ├── If Pruned: Flagged as DDoS (1), Dropped from polluting Layer 2
      │                  └── If Passed: Routed via MABTSERouter to Layer 2 Classifiers
      │
      └──> Results Table Displayed & Saved to CSV in [results/]
```

### Detailed File Responsibilities
1. **`msaad/config.py`**: Central system repository containing dataset paths, attack hyperparameters ($\epsilon=0.1$, step sizes, iterations), GAN architectures, learning rates, and threshold boundaries.
2. **`msaad/data_loader.py`**: High-performance streaming generators (`stream_ciciot23`, `stream_iotid20`, `stream_cybersecurity`). Drops metadata to eliminate target leakage, converts multi-class labels into uniform binary ground truths (`0: Normal`, `1: DDoS`), and yields single instances.
3. **`msaad/preprocessing.py`**: Houses the unified `OnlinePreprocessingPipeline` chaining 7 incremental transformation sub-modules.
4. **`msaad/adversarial.py`**: Houses `SurrogateModel` and `AdversarialAttackGenerator`. Wraps an online PyTorch MLP in ART's `PyTorchClassifier` to generate gradient attacks in streaming memory.
5. **`msaad/radp.py`**: Houses `Generator`, `Discriminator`, `AdversarialGAN`, `ResilientAdversarialDetector`, and `RADPLayer` orchestrating Layer 1 defense and sample pruning.
6. **`msaad/classifiers.py`**: Implements standardized wrappers over five streaming classifiers (`ARFADWINClassifier`, `SRPDDMClassifier`, `HTADWINClassifier`, `KNNADWINClassifier`, `OnlineLogisticRegressionClassifier`).
7. **`msaad/mabtse.py`**: Implements `LinUCBMAB` (disjoint ridge-regression contextual bandit with Sherman-Morrison rank-1 updates) and `AUWPAE` (sliding-window accuracy probability combiner).
8. **`msaad/evaluation.py`**: Encapsulates the complete `MSAADSystem` class, metric computators (`accuracy`, `macro-precision`, `macro-recall`, `macro-f1`, `throughput`, `learning_time`), ASCII table formatter, and CSV persistence logic.
9. **`run_all.py` / `msaad/main.py`**: Driver scripts providing full CLI argument parsing, multi-dataset execution loops, console logging, and `StreamTee` dual stdout/file capture.

---

## 🔬 Algorithmic Deep Dive

### 1. Online Preprocessing Pipeline

Network traffic in IoT exhibits non-stationary behavior and heavy feature cross-correlations. MSAAD handles this without batch storage via seven cascading algorithms:

1. **Incremental One-Hot Encoder:** Dynamically tracks categorical unique feature states; passes numeric features untouched.
2. **Online Imputer:** Maintains incremental sums $\sum x_i$ and counts $N_i$ to perform mean imputation $\mu_i = \frac{\sum x_i}{N_i}$ for missing attributes in $O(1)$ time.
3. **Online Min-Max Normalizer:** Tracks running extrema $[x_{\min, i}, x_{\max, i}]$ per feature, mapping streaming inputs strictly onto $[0, 1]$:
   $$\hat{x}_i = \frac{x_i - x_{\min, i}}{x_{\max, i} - x_{\min, i} + \epsilon}$$
4. **Mini-Batch K-Means Augmentor:** Buffers batches of 256 samples, updates centroids via mini-batch Lloyd's algorithm, and appends $k=8$ Euclidean distance features to the vector:
   $$f_{\text{dist}, j} = \frac{\|x - c_j\|_2}{\max_k \|x - c_k\|_2}$$
5. **Online SMOTE (Reservoir Sampling + Synthesis):**
   Maintains a reservoir of size $R=200$ for minority-class instances. When the streaming imbalance ratio drops below $\tau=1.0$, synthetic vectors are generated on the fly via convex combination between random reservoir pairs $x_a, x_b$:
   $$x_{\text{syn}} = \lambda x_a + (1 - \lambda) x_b, \quad \lambda \sim U(0, 1)$$
6. **Online Information Gain Feature Selector:**
   Tracks dynamic empirical joint probability distributions $P(X \in B_b, Y = y)$ across $B=10$ uniform bins per feature. Periodically computes Shannon Entropy $H(Y)$ and conditional entropy $H(Y|X_i)$ every 500 samples:
   $$IG(Y, X_i) = H(Y) - \sum_{b=1}^{B} P(X_i \in B_b) H(Y | X_i \in B_b)$$
   Features failing the threshold ($IG < 0.01$) are pruned.
7. **Online Pearson Correlation Deduplicator (Vectorized Welford's Algorithm):**
   Computes exact streaming variance and covariance matrices in a single pass without storing past data. Given running mean vector $\bar{x}_n$ and comoment matrix $C_n$:
   $$\bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n}$$
   $$C_n = C_{n-1} + (x_n - \bar{x}_{n-1})(x_n - \bar{x}_n)^T$$
   $$\text{Corr}(i, j) = \frac{C_{n, ij}}{\sqrt{C_{n, ii} \cdot C_{n, jj}}}$$
   If $|\text{Corr}(i, j)| \ge 0.95$, redundant feature $j$ is pruned online.

---

### 2. Adversarial Perturbation Generation (ART & MSAPG)

To simulate adversarial attacks in streaming memory, MSAAD maintains a lightweight PyTorch Multi-Layer Perceptron (MLP: $d \to 64 \to 32 \to 2$) inside a sliding window of size $W=512$. Every 50 incoming samples, the surrogate refits via Adam optimizer for 5 epochs.

Wrapped inside ART's `PyTorchClassifier`, four gradient attack methods are evaluated:
- **Fast Gradient Sign Method (FGSM):**
  $$x_{\text{adv}} = \text{clip}_{[0, 1]}\left(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))\right)$$
  *(Hyperparameters: $\epsilon = 0.1$)*
- **Basic Iterative Method (BIM):**
  $$x_0 = x, \quad x_{t+1} = \text{clip}_{x, \epsilon}\left(x_t + \alpha \cdot \text{sign}(\nabla_{x_t} \mathcal{L}(\theta, x_t, y))\right)$$
  *(Hyperparameters: $\epsilon = 0.1, \alpha = 0.01, T = 10$ iterations)*
- **Projected Gradient Descent (PGD):**
  Same iterative formulation as BIM, initialized with uniform random perturbation $x_0 = x + \delta, \; \delta \sim U(-\epsilon, \epsilon)$.
- **Carlini & Wagner $L_2$ (C&W):**
  Solves the optimization problem:
  $$\min_\delta \|\delta\|_2^2 + c \cdot f(x + \delta), \quad \text{where } f(x') = \max(\log P(y_{\text{orig}}|x') - \max_{i \ne y} \log P(i|x'), -\kappa)$$
  *(Hyperparameters: confidence $\kappa = 0.1$, max iterations $= 20$, batch size $= 32$)*

#### Multi-Source Adversarial Perturbation Generator (MSAPG)
To train the defense GAN against diverse perturbation distributions, MSAPG computes an ensemble average perturbation:
$$x_{\text{MSAPG}} = \frac{1}{4} \left( x_{\text{FGSM}} + x_{\text{BIM}} + x_{\text{PGD}} + x_{\text{CW}} \right)$$

---

### 3. Resilient Adversarial Detection and Purification (RADP)

RADP operates as **Layer 1** of the defense architecture. It contains two interconnected sub-modules:

1. **Streaming Adversarial GAN:**
   - **Generator ($G_\phi$):** Takes Gaussian noise $z \sim \mathcal{N}(0, I_{32})$ and projects it through LeakyReLU layers to synthesize realistic adversarial feature vectors $\hat{x}_{\text{adv}} \in [0, 1]^d$ via Sigmoid activation.
   - **Discriminator ($D_\psi$):** Classifies real adversarial samples (from MSAPG) vs. generator fakes using Binary Cross-Entropy loss:
     $$\min_G \max_D \mathbb{E}_{x \sim \mathcal{D}_{\text{adv}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log (1 - D(G(z)))]$$
2. **Resilient Adversarial Detector (RAD):**
   A streaming logistic regression classifier updated via L2-regularized SGD with decaying learning rate:
   $$\hat{y}_{\text{adv}} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
   $$w \leftarrow w - \eta_t \left( (\hat{y}_{\text{adv}} - y_{\text{true}}) x + \lambda w \right), \quad \eta_t = \frac{\eta_0}{1 + 0.001 \cdot t}$$
   **Pruning Decision:**
   $$\text{Action}(x) = \begin{cases} \text{PRUNE (Flag as DDoS \& Drop from Layer 2)}, & \text{if } \hat{y}_{\text{adv}} \ge 0.5 \\ \text{PASS (Forward to Layer 2 Classifiers)}, & \text{if } \hat{y}_{\text{adv}} < 0.5 \end{cases}$$

---

### 4. Base Classifiers with Concept Drift Adaptation

Samples passing Layer 1 reach five heterogeneous streaming models in **Layer 2**:

1. **Adaptive Random Forest (ARF-ADWIN):** Ensemble of 10 Hoeffding Trees with online bagging. Employs the ADWIN algorithm to monitor per-tree error rates; replaces underperforming trees when drift occurs.
2. **Streaming Random Patches (SRP-DDM):** Ensembles 10 base estimators trained over random feature subspaces, paired with Drift Detection Method (DDM) for warnings and drift triggers.
3. **Hoeffding Tree with ADWIN (HT-ADWIN):** Incremental decision tree utilizing the Hoeffding Bound to guarantee split selections asymptotically equivalent to batch trees. Resets tree structure upon ADWIN drift alert.
4. **K-Nearest Neighbors with ADWIN (KNN-ADWIN):** Maintains a sliding buffer of recent exemplars ($k=5$). Drops stale neighborhood points upon drift detection.
5. **Online Logistic Regression (OLR):** River-based incremental SGD pipeline preconditioned with online standard scaling.

---

### 5. LinUCB Contextual Multi-Armed Bandit with AUWPAE

**Layer 3** implements the disjoint **LinUCB** algorithm to select the optimal model arm per sample context:

$$\text{Arms } \mathcal{A} = \{0: \text{ARF}, \; 1: \text{SRP}, \; 2: \text{HT}, \; 3: \text{KNN}, \; 4: \text{OLR}, \; 5: \text{AUWPAE Ensemble}\}$$

Given feature context vector $x_t \in \mathbb{R}^d$ ($L_2$-normalized, dimension $d=16$):
1. For each arm $a \in \mathcal{A}$, compute estimated ridge regression parameter $\hat{\theta}_a$ and Upper Confidence Bound:
   $$\hat{\theta}_a = A_a^{-1} b_a$$
   $$\text{UCB}_a(x_t) = \hat{\theta}_a^T x_t + \alpha \sqrt{x_t^T A_a^{-1} x_t}$$
   *(where $A_a \in \mathbb{R}^{d \times d}$ is initialized to $I_d$, $b_a \in \mathbb{R}^d$ to $0$, and exploration parameter $\alpha = 1.0$)*
2. Select arm with maximum UCB: $a_t = \arg\max_{a} \text{UCB}_a(x_t)$.
3. **Accuracy-Updated Weighted Probability Averaging Ensemble (AUWPAE):**
   When arm 5 is pulled, probabilities from all 5 base models are aggregated using sliding-window ($W=200$) accuracy weights:
   $$w_i = \frac{\sum_{k=1}^{W} \mathbb{I}(\hat{y}_{i, k} = y_k)}{W}, \quad P_{\text{ensemble}}(y=1) = \frac{\sum_{i=1}^5 w_i \cdot P_i(y=1)}{\sum_{i=1}^5 w_i}$$
4. **Sherman-Morrison Incremental Inverse Matrix Update:**
   Upon observing binary feedback reward $r_t \in \{0, 1\}$ (1 for correct classification, 0 for error), update $A_{a_t}$ and $b_{a_t}$ without matrix inversion:
   $$A_{a_t}^{-1} \leftarrow A_{a_t}^{-1} - \frac{A_{a_t}^{-1} x_t x_t^T A_{a_t}^{-1}}{1 + x_t^T A_{a_t}^{-1} x_t}, \quad b_{a_t} \leftarrow b_{a_t} + r_t x_t$$

---

## 📊 Comparative Algorithmic Analysis

| Component | Chosen Algorithm | Alternatives Evaluated | Time Complexity (Per Sample) | Space Complexity | Selection Justification |
|---|---|---|:---:|:---:|---|
| **Class Balancing** | **Online Reservoir SMOTE** | Online Under-sampling, Random Oversampling | $O(d)$ | $O(R \cdot d)$ | Interpolates realistic minority traffic in streaming memory without sample dropping. |
| **Feature Selection** | **Streaming Information Gain** | Online PCA, Variance Thresholding | $O(d \cdot B)$ | $O(d \cdot B \cdot \|C\|)$ | Captures non-linear mutual dependence with true class labels; zero dense matrix transformation overhead. |
| **Collinearity Filter** | **Vectorized Welford Correlation** | Batch Pearson, Matrix Factorization | $O(d^2)$ | $O(d^2)$ | Single-pass numerical stability; exact streaming covariance without historical caches. |
| **Adversarial Detection** | **RADP (GAN + Online RAD)** | Autoencoder Reconstruction, Defense-GAN | $O(d \cdot \text{dim}_h)$ | $O(\text{dim}_{\text{params}})$ | Evaluates sample perturbation likelihood in $< 1 \text{ ms}$; filters attacks before classifiers are exposed. |
| **Classifier Selection** | **LinUCB Contextual Bandit** | Majority Voting, Thompson Sampling, Stacking | $O(d^2)$ via Sherman-Morrison | $O(\|\mathcal{A}\| \cdot d^2)$ | Exploits instantaneous feature context $x_t$ rather than static unconditioned historical accuracy. |

---

## 📁 Directory Structure

```
Multi-Stage_Adversarial_Defense_for_Online_DDoS_Attack_Detection_System_in_IoT/
│
├── CICIOT23/                               # Canadian Institute for Cybersecurity Dataset
│   ├── train/train.csv                     # Ingestion train split (warmup)
│   ├── test/test.csv                       # Ingestion test split (unseen evaluation)
│   └── validation/validation.csv           # Validation split
│
├── IoTID20/                                # IoT Network Intrusion Dataset (IEEE DataPort)
│   ├── IoT Network Intrusion Dataset.csv   # Primary flow records
│   └── archive (3)/cybersecurity_attacks.csv
│
├── msaad/                                  # Core MSAAD Python Engine Package
│   ├── __init__.py                         # Package exposure
│   ├── config.py                           # Central configuration & hyperparameters
│   ├── data_loader.py                      # Streaming chunked CSV generators
│   ├── preprocessing.py                    # 7-stage online preprocessing pipeline
│   ├── adversarial.py                      # PyTorch MLP surrogate + ART attack suite
│   ├── radp.py                             # Layer 1: GAN, MSAPG, RAD detector, & pruner
│   ├── classifiers.py                      # Layer 2: 5 drift-aware River streaming models
│   ├── mabtse.py                           # Layer 3: LinUCB contextual bandit + AUWPAE
│   ├── evaluation.py                       # MSAADSystem coordinator & metric analytics
│   └── main.py                             # Modular CLI entrypoint
│
├── results/                                # Timestamped logs and CSV evaluation tables
│   ├── evaluation_results_ciciot23.csv
│   ├── evaluation_results_iotid20.csv
│   └── msaad_run_*.log
│
├── COMMANDS.md                             # Exhaustive 26-command operational playbook
├── requirements.txt                        # Production dependency manifests
├── run_all.py                              # Master testbed driver across all permutations
├── MSAAD_Presentation.pptx                 # 12-slide widescreen academic defense deck
└── README.md                               # System documentation (this file)
```

---

## ⚙️ Setup & Installation Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit recommended)
- Git & PowerShell / Bash
- Recommended: 8 GB+ RAM, multi-core CPU

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Multi-Stage_Adversarial_Defense_for_Online_DDoS_Attack_Detection_System_in_IoT.git
cd Multi-Stage_Adversarial_Defense_for_Online_DDoS_Attack_Detection_System_in_IoT
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Windows Encoding Setup (PowerShell only)
On Windows platforms, ensure standard UTF-8 console streaming:
```powershell
$env:PYTHONIOENCODING="utf-8"
```

### 5. Verify Installation
```bash
python -c "import river, torch, art, sklearn; print('All core libraries verified successfully!')"
```

---

## 🚀 Running Experiments

The framework provides two execution entrypoints: `run_all.py` (comprehensive multi-dataset driver) and `msaad/main.py` (granular single-scenario driver).

### 1. Fast Smoke Test (~30 Seconds)
Executes a rapid pipeline verification using 200 samples and the FGSM attack on `CICIoT2023`:
```bash
python run_all.py --fast
```

### 2. Full Multi-Dataset, Multi-Attack Benchmark
Executes all 4 attacks across all 3 scenarios on both `CICIoT2023` and `IoTID20`:
```bash
python run_all.py --samples 5000 --warmup 500
```

### 3. Attack-Specific Evaluations via `main.py`
Evaluate specific attacks and scenarios:
```bash
# Evaluate Projected Gradient Descent (PGD) across all three scenarios
python -m msaad.main --dataset ciciot23 --samples 5000 --attack pgd --scenario all

# Evaluate Carlini & Wagner (C&W L2) optimization attack
python -m msaad.main --dataset ciciot23 --samples 1000 --warmup 300 --attack cw --scenario all

# Evaluate baseline clean traffic (Original scenario only)
python -m msaad.main --dataset iotid20 --samples 5000 --scenario original
```

### CLI Options Reference (`main.py`):
| Flag | Options | Default | Description |
|---|---|---|---|
| `--dataset` | `ciciot23`, `iotid20`, `cybersecurity`, `all` | `ciciot23` | Target benchmark dataset |
| `--samples` | Integer | `10000` | Testing instances evaluated per scenario |
| `--warmup` | Integer | `500` | Unsupervised/supervised initial pipeline warmup samples |
| `--attack` | `fgsm`, `bim`, `pgd`, `cw`, `all` | `fgsm` | Adversarial generation method |
| `--scenario`| `original`, `attack`, `recovered`, `all` | `all` | Operational evaluation mode |

---

## 📈 Experimental Benchmark Results

### 1. Canadian Institute for Cybersecurity (CICIoT2023)
*Evaluated on 10,000 streaming test samples with 500-sample warmup under FGSM ($\epsilon = 0.1$):*

| Evaluation Scenario | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score | Throughput | Pruned Adversarial Samples |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scenario 1: Original (Clean)** | 99.50% | 99.51% | 99.50% | **99.50%** | **132.08 sps** | N/A (Defense Inactive) |
| **Scenario 2: Under Attack** | 100.00% | 100.00% | 100.00% | **100.00%** | **92.92 sps** | 0 (Unprotected) |
| **Scenario 3: Recovered (RADP Active)** | 100.00% | 100.00% | 100.00% | **100.00%** | **50.03 sps** | **200 / 200 (100.0% Pruned)** |

### 2. IoT Network Intrusion Dataset (IoTID20)
*Evaluated on 10,000 streaming test samples with 500-sample warmup under FGSM ($\epsilon = 0.1$):*

| Evaluation Scenario | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score | Throughput | Pruned Adversarial Samples |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scenario 1: Original (Clean)** | 100.00% | 100.00% | 100.00% | **100.00%** | **104.53 sps** | N/A (Defense Inactive) |
| **Scenario 2: Under Attack** | 100.00% | 100.00% | 100.00% | **100.00%** | **79.88 sps** | 0 (Unprotected) |
| **Scenario 3: Recovered (RADP Active)** | 100.00% | 100.00% | 100.00% | **100.00%** | **76.85 sps** | **1,444 / 2,000 (72.2% Pruned)** |

---

## 👥 Authors & Academic Advisory

**Developers & Researchers:**
- **Devesh** — Roll No: `20233115`
- **Govind Saini** — Roll No: `20233128`
- **Gaurav Mahor** — Roll No: `20233127`
- **Krishna Gond** — Roll No: `20233341`

**Academic Supervisor & Evaluator:**
- **Dr. Indu Dohare**
- Department of Computer Science & Engineering

---

## 📄 License
This research project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
