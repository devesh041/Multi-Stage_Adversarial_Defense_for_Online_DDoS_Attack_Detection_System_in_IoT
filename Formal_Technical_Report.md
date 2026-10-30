# Multi-Stage Adversarial Defense for Online DDoS Attack Detection System in IoT (MSAAD)
## Formal Technical & Architectural Report

---

<br><br><br>

# MULTI-STAGE ADVERSARIAL DEFENSE FOR ONLINE DDOS ATTACK DETECTION SYSTEM IN IOT (MSAAD)

### A Unified Streaming Architecture for Robust Distributed Denial-of-Service Detection under Adversarial Perturbations

<br><br>

**A Major Project Technical Report Submitted in Partial Fulfillment of the Requirements for the Degree of**  
**Bachelor of Technology in Computer Science & Engineering**

<br><br>

**Submitted by:**
- **Devesh** (Roll No: 20233115)
- **Govind Saini** (Roll No: 20233128)
- **Gaurav Mahor** (Roll No: 20233127)
- **Krishna Gond** (Roll No: 20233341)

<br>

**Under the Guidance and Supervision of:**  
**Dr. Indu Dohare**  
Department of Computer Science & Engineering  

<br><br>

**Academic Session: 2025–2026**

---

<div style="page-break-after: always;"></div>

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction & Background](#2-introduction--background)
   - 2.1 The IoT Paradigm and DDoS Attack Vectors
   - 2.2 Vulnerability of Machine Learning to Adversarial Manipulation
   - 2.3 The Imperative for Online Streaming Defenses
3. [System Architecture & Workflow](#3-system-architecture--workflow)
   - 3.1 Architectural Principles
   - 3.2 High-Level Multi-Tier Architecture
   - 3.3 Four-Phase Data Pipeline
4. [Component Interaction & Code Flow](#4-component-interaction--code-flow)
   - 4.1 System Orchestration and Runtime Lifecycle
   - 4.2 Module Interconnectivity & Data Contracts
   - 4.3 Streaming State Management & Memory Footprint
5. [Algorithmic Implementation & Mathematical Formulation](#5-algorithmic-implementation--mathematical-formulation)
   - 5.1 Streaming Data Ingestion & Sanitization
   - 5.2 Phase 1: Online Preprocessing Pipeline
   - 5.3 Phase 2: Adversarial Attack Simulation & MSAPG
   - 5.4 Phase 3 (Layer 1): Resilient Adversarial Detection & Purification (RADP)
   - 5.5 Phase 3 (Layer 2): Heterogeneous Streaming Ensemble with Drift Adaptation
   - 5.6 Phase 3 (Layer 3): LinUCB Contextual Bandit Routing (MABLUCBE)
6. [Empirical Evaluation & Comparative Analysis](#6-empirical-evaluation--comparative-analysis)
   - 6.1 Benchmark Datasets
   - 6.2 Experimental Setup & Tri-Scenario Protocol
   - 6.3 Quantitative Findings & Discussion
7. [Conclusion & Future Work](#7-conclusion--future-work)
8. [References](#8-references)

---

<div style="page-break-after: always;"></div>

## 1. Abstract

The explosive expansion of Internet of Things (IoT) ecosystems has democratized smart infrastructure while introducing unprecedented cybersecurity vulnerabilities. Resource-constrained edge devices frequently become unwilling participants in Mirai-like botnets, generating devastating Distributed Denial-of-Service (DDoS) flood attacks. Although modern Machine Learning-based Network Intrusion Detection Systems (ML-NIDS) achieve superior detection accuracy on static benchmarks, they suffer from two critical architectural flaws: reliance on computationally heavy batch retraining that fails under continuous traffic concept drift, and vulnerability to adversarial evasion attacks wherein subtle, gradient-directed input perturbations force ML models to misclassify malicious flows as benign. 

This technical report presents the **Multi-Stage Adversarial Defense for Online DDoS Attack Detection System in IoT (MSAAD)** framework. MSAAD represents an end-to-end, strictly streaming defense and detection architecture designed for edge IoT gateways. The framework integrates an incremental seven-stage preprocessing pipeline, a novel **Resilient Adversarial Detection and Purification (RADP)** layer combining an online Generative Adversarial Network (GAN) with an L2-regularized online logistic detector, a diverse ensemble of five concept drift-adaptive streaming classifiers (ARF-ADWIN, SRP-DDM, HT-ADWIN, KNN-ADWIN, OLR), and a **LinUCB contextual multi-armed bandit** utilizing Sherman-Morrison updates to dynamically route samples based on instantaneous feature context. Evaluated against four state-of-the-art adversarial attacks (FGSM, BIM, PGD, and Carlini & Wagner $L_2$) across the `CICIoT2023` and `IoTID20` benchmark datasets, MSAAD achieves an F1-score of 99.50%–100.00% across all operational scenarios, maintaining high streaming throughput (50–132 samples/second) and demonstrating up to 100% adversarial sample pruning in real-time streaming memory.

---

## 2. Introduction & Background

### 2.1 The IoT Paradigm and DDoS Attack Vectors
The Internet of Things (IoT) encompasses tens of billions of interconnected edge sensors, smart utility actuators, healthcare telemetry modules, and embedded appliances. Characterized by constrained computation, minimal onboard memory, heterogeneous communication protocols, and default or unpatched configurations, these devices present an expansive attack surface. Malicious syndicates frequently exploit these vulnerabilities to compromise millions of devices, orchestrating coordinated botnet swarms (such as Mirai, Reaper, and Torii) capable of generating terabit-scale Volumetric, Protocol, and Application-layer DDoS floods. Such attacks overwhelm transport bandwidth and server processing queues, causing catastrophic service outages across critical civilian and industrial services.

### 2.2 Vulnerability of Machine Learning to Adversarial Manipulation
To combat high-velocity DDoS campaigns, automated Network Intrusion Detection Systems (NIDS) have increasingly transitioned toward Machine Learning (ML) paradigms, including Random Forests, Deep Neural Networks (DNNs), and Support Vector Machines. While these classifiers achieve high statistical accuracy on static data splits, recent studies reveal that they are inherently brittle against adversarial evasion attacks. 

By applying bounded mathematical optimization—such as the Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), Projected Gradient Descent (PGD), or Carlini & Wagner (C&W) $L_2$ optimization—an adversary possessing gradient access (or querying capabilities) can introduce imperceptible perturbations $\delta$ to normalized traffic flow statistics such that:
$$\hat{f}(x + \delta) \ne \hat{f}(x), \quad \text{subject to } \|\delta\|_p \le \epsilon$$
In the context of NIDS, this manipulation causes volumetric flood vectors to be misclassified as legitimate communication, rendering automated detection mechanisms completely ineffective.

### 2.3 The Imperative for Online Streaming Defenses
The conventional countermeasure in adversarial machine learning is **adversarial retraining** (e.g., Madry defense), wherein models are retrained offline on augmented datasets of synthesized adversarial vectors. In IoT edge environments, this strategy is fundamentally unviable for three reasons:
1. **Streaming Velocity & Data Volume:** IoT gateways process millions of telemetry packets continuously; persisting full historical packet flows for periodic retraining exhausts local storage and introduces unacceptable latency.
2. **Concept Drift:** Network traffic distributions fluctuate dynamically due to legitimate behavioral shifts (covariate shift) and evolving attack signatures (concept shift). Static or periodically retrained models become obsolete within hours.
3. **Catastrophic Forgetting and Generalization Gaps:** Batch adversarial training tuned to a specific perturbation magnitude ($\epsilon$) or attack objective fails to generalize to distinct or combined attack vectors.

Consequently, modern IoT security necessitates an autonomous, **multi-stage, strictly streaming architecture** capable of single-pass data processing, online adversarial sample filtering, continuous drift adaptation, and context-aware model orchestration.

---

<div style="page-break-after: always;"></div>

## 3. System Architecture & Workflow

### 3.1 Architectural Principles
The MSAAD framework is founded upon four structural tenets:
1. **Zero-Batch Invariance:** Every mathematical transformation—imputation, scaling, synthetic oversampling, covariance estimation, adversarial detection, and classifier inference—must execute incrementally in $O(1)$ memory per instance.
2. **Defensive Depth (Layered Immunity):** Rather than relying on an isolated robust classifier, MSAAD employs three sequential protective gates: input purifying/pruning (Layer 1), heterogeneous drift-aware classification (Layer 2), and contextual algorithmic routing (Layer 3).
3. **Contextual Exploitation-Exploration:** Rather than static majority voting, ensemble arbitration must evaluate the instantaneous feature signature of each flow vector to route instances to the historically most competent model.
4. **Target Leakage Prevention:** All non-payload identifier metadata (IP addresses, Port numbers, Flow IDs, Timestamps) are stripped at ingestion to prevent artificial feature correlation.

### 3.2 High-Level Multi-Tier Architecture

```
═════════════════════════════════════════════════════════════════════════════════
                       MSAAD END-TO-END ARCHITECTURE
═════════════════════════════════════════════════════════════════════════════════

    [Raw Streaming Network Flows: CICIoT2023 / IoTID20 / Cybersecurity]
                                   │
                                   ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │            PHASE 1: ONLINE STREAMING PREPROCESSING PIPELINE                 │
 │                                                                             │
 │  1. Incremental Categorical One-Hot Encoder                                 │
 │  2. Online Mean Imputer (Running First-Moment Tracking)                     │
 │  3. Online Min-Max Scaler (Dynamic Extrema Projection to [0, 1])            │
 │  4. Mini-Batch K-Means Augmentor (k=8 Cluster-Distance Feature Synthesis)   │
 │  5. Online SMOTE (Reservoir Sampling Buffer + Convex Interpolation)         │
 │  6. Online Information Gain Selector (Running Entropy Binning, threshold=0.01)│
 │  7. Online Pearson Deduplicator (Vectorized Welford Covariance, corr<0.95) │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │ Clean Vector x ∈ [0, 1]^d
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │            PHASE 2: ADVERSARIAL ATTACK SIMULATION (STRESS TEST)             │
 │                                                                             │
 │  Online PyTorch MLP Surrogate (Sliding Buffer W=512, 50-step Periodic Fit)  │
 │  ├── Fast Gradient Sign Method (FGSM, ε=0.1)                                │
 │  ├── Basic Iterative Method (BIM, ε=0.1, step=0.01, T=10)                   │
 │  ├── Projected Gradient Descent (PGD, ε=0.1, step=0.01, T=10)               │
 │  ├── Carlini & Wagner L2 (C&W, confidence=0.1, iter=20)                     │
 │  └── MSAPG: Multi-Source Adversarial Perturbation Averager                  │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │ Perturbed Flow Vector x_adv
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │            PHASE 3: THREE-LAYER DETECTION & DEFENSE ENGINE                  │
 │                                                                             │
 │ ┌─────────────────────────────────────────────────────────────────────────┐ │
 │ │ LAYER 1: Resilient Adversarial Detection & Purification (RADP)          │ │
 │ │  • Online GAN (Generator & Discriminator mimics adversarial space)      │ │
 │ │  • RAD Detector: Online Logistic SGD with Decaying Learning Rate        │ │
 │ │                                                                         │ │
 │ │  Decision: P_adv(x) >= 0.5 ?                                            │ │
 │ └───────┬─────────────────────────────────────────────────┬───────────────┘ │
 │         │ [YES: Pruned]                                   │ [NO: Passed]    │
 │         ▼                                                 ▼                 │
 │  ┌──────────────┐                       ┌─────────────────────────────────┐ │
 │  │ Flagged DDoS │                       │ Passed to Layer 2 Classifiers   │ │
 │  │ Dropped from │                       └────────────────┬────────────────┘ │
 │  │ Layer 2/3    │                                        │                  │
 │  └──────────────┘                                        ▼                  │
 │ ┌─────────────────────────────────────────────────────────────────────────┐ │
 │ │ LAYER 2: Heterogeneous Base Classifiers with Concept Drift Detection    │ │
 │ │  • Arm 0: ARF-ADWIN (Adaptive Random Forest, 10 trees)                  │ │
 │ │  • Arm 1: SRP-DDM   (Streaming Random Patches, DDM drift detector)      │ │
 │ │  • Arm 2: HT-ADWIN  (Hoeffding Tree, ADWIN reset trigger)               │ │
 │ │  • Arm 3: KNN-ADWIN (Streaming KNN, k=5, ADWIN reset trigger)           │ │
 │ │  • Arm 4: OLR       (Online Logistic Regression via incremental SGD)    │ │
 │ └────────────────────────────────────────┬────────────────────────────────┘ │
 │                                          │ Model Prediction Probas {P_i}    │
 │                                          ▼                                  │
 │ ┌─────────────────────────────────────────────────────────────────────────┐ │
 │ │ LAYER 3: LinUCB Contextual Multi-Armed Bandit Routing (MABLUCBE)        │ │
 │ │  • Feature Context Vector x_t ∈ R^16 (L2-Normalized Projection)         │ │
 │ │  • UCB_a = θ_a^T x_t + α √(x_t^T A_a^-1 x_t)                            │ │
 │ │  • Arm 0-4: Direct Classifier Dispatch                                  │ │
 │ │  • Arm 5: AUWPAE Ensemble (Sliding Accuracy-Weighted Probabilities)     │ │
 │ └────────────────────────────────────────┬────────────────────────────────┘ │
 └──────────────────────────────────────────╪──────────────────────────────────┘
                                            │
                                            ▼
                  Final Binary Decision Output: Normal (0) vs. DDoS (1)
                                            │
                                            ▼
               Online Feedback Loop: Sherman-Morrison A_a^-1 Update
```

### 3.3 Four-Phase Data Pipeline
The lifecycle comprises:
- **Phase 1 (Stream Preprocessing):** Continuous numerical cleaning, scaling, dynamic topological augmentation via mini-batch clustering, online minority synthesis, and dual-stage feature pruning.
- **Phase 2 (Adversarial Generation):** Active perturbation synthesis via an online surrogate to stress-test system robustness against four attack modalities.
- **Phase 3 (Tri-Layer Defense & Detection):** Pre-filtering by RADP (Layer 1), parallel streaming inference by drift-aware base estimators (Layer 2), and context-conditioned bandit selection (Layer 3).
- **Phase 4 (Metric Auditing & Adaptation):** Continuous measurement of macro-classification metrics, per-instance execution latency, throughput, and online parameter updates via feedback rewards.

---

<div style="page-break-after: always;"></div>

## 4. Component Interaction & Code Flow

### 4.1 System Orchestration and Runtime Lifecycle
The framework executes deterministically via `run_all.py` or `msaad/main.py`. The programmatic flow unfolds across four distinct stages:

```
[System Startup: run_all.py / main.py]
  │
  ├── 1. Ingest CLI Parameters & Load Hyperparameters [msaad.config]
  │
  ├── 2. Instantiate MSAADSystem Container [msaad.evaluation]
  │        ├── OnlinePreprocessingPipeline [msaad.preprocessing]
  │        └── AdversarialAttackGenerator   [msaad.adversarial]
  │
  ├── 3. Execute Pipeline Warmup (n=500 samples)
  │        ├── Stream raw train CSV records [msaad.data_loader]
  │        ├── Preprocessing pipeline fits scalers, imputers, and initial feature sets
  │        ├── Adversarial surrogate MLP trains initial 5 epochs on clean samples
  │        ├── Synthesize adversarial warmup buffer via FGSM
  │        ├── RADPLayer initialized & pre-trained (GAN + RAD detector SGD) [msaad.radp]
  │        ├── Layer2Classifiers instantiated & trained on clean warmup records [msaad.classifiers]
  │        └── MABTSERouter initialized with LinUCB identity matrices [msaad.mabtse]
  │
  └── 4. Execute Benchmark Evaluation Scenarios
           │
           ├── Scenario 1: Original Clean Evaluation
           │     └── stream_test() -> Preprocessing -> Layer 3 Bandit -> Layer 2 Classifiers
           │           └── Compute Baseline Accuracy, F1, Throughput
           │
           ├── Scenario 2: Under Attack Evaluation (Unprotected)
           │     └── stream_test() -> Preprocessing -> Inject Attack (FGSM/BIM/PGD/CW)
           │           └── Bypass Layer 1 -> Layer 3 Bandit -> Layer 2 Classifiers
           │                 └── Measure Attack Vulnerability Drop
           │
           └── Scenario 3: Recovered Evaluation (RADP Active)
                 └── stream_test() -> Preprocessing -> Inject Attack
                       ├── Layer 1 RADP evaluates P(Adversarial)
                       │     ├── If >= 0.5: PRUNE -> Flag as DDoS (1), Drop from L2/L3
                       │     └── If < 0.5:  PASS  -> Forward to Layer 2 & Layer 3
                       └── Measure Recovered Accuracy, F1, Prune Rate, Throughput
```

### 4.2 Module Interconnectivity & Data Contracts
The system guarantees low latency by exchanging pure NumPy arrays and lightweight dictionary structures between modules:

| Originating Module | Receiving Module | Data Structure Transferred | Mathematical Representation | Functional Purpose |
|---|---|---|---|---|
| `data_loader.py` | `preprocessing.py` | `(dict[str, float], int)` | $(x_{\text{raw}}, y), \; y \in \{0, 1\}$ | Ingests unnormalized network attributes and ground-truth binary label. |
| `preprocessing.py` | `adversarial.py` | `np.ndarray` | $x \in [0, 1]^d$ | Passes clean, normalized, non-redundant feature vector to surrogate. |
| `adversarial.py` | `radp.py` | `np.ndarray` | $x_{\text{adv}} \in [0, 1]^d$ | Supplies gradient-perturbed feature vectors to train GAN and test RAD. |
| `radp.py` | `mabtse.py` | `(bool, float, dict)` | $(\text{is\_pruned}, \hat{y}_{\text{adv}}, x)$ | Transmits pruning status and purified feature dictionary to Layer 3. |
| `mabtse.py` | `classifiers.py` | `dict[str, float]` | $x_{\text{features}}$ | Delivers features to either the selected arm or all 5 ensemble models. |
| `classifiers.py` | `mabtse.py` | `list[float]` | $[P_0, P_1, P_2, P_3, P_4]$ | Delivers class probability outputs $P(y=1)$ for AUWPAE weighting. |
| `mabtse.py` | `evaluation.py` | `(int, int)` | $(\hat{y}, \text{arm\_id})$ | Delivers final binary prediction and selected arm index for scoring. |

### 4.3 Streaming State Management & Memory Footprint
To operate reliably on embedded devices without memory exhaustion, MSAAD restricts state storage to strict upper bounds:
- **Surrogate Buffer:** Fixed deque capacity ($W=512$ vectors).
- **GAN Replay Memory:** Fixed deque capacity ($W_{\text{adv}}=512$, $W_{\text{clean}}=512$).
- **Online SMOTE Reservoir:** Fixed reservoir capacity ($R=200$ minority samples).
- **LinUCB Matrix Memory:** Context dimension $d=16$ entails six pairs of $16 \times 16$ float64 matrices ($A_a^{-1}$) and $16 \times 1$ vectors ($b_a$), requiring less than **25 Kilobytes** of RAM.
- **AUWPAE Accuracy Windows:** Five sliding deques with max length $W=200$.

---

<div style="page-break-after: always;"></div>

## 5. Algorithmic Implementation & Mathematical Formulation

### 5.1 Streaming Data Ingestion & Sanitization
Implemented in `msaad/data_loader.py`, ingestion reads CSV files via chunked record generators. 
To eliminate **target leakage**, metadata attributes that could artificially bias the classifiers are discarded:
$$\mathcal{F}_{\text{dropped}} = \{\text{"Flow\_ID"}, \text{"Src\_IP"}, \text{"Dst\_IP"}, \text{"Timestamp"}, \text{"Label"}, \text{"Cat"}, \text{"Sub\_Cat"}\}$$
Multi-class label strings are mapped deterministically onto a uniform binary target space:
$$y = \begin{cases} 0 (\text{Normal}), & \text{if } \text{Label} \in \{\text{"BenignTraffic"}, \text{"Normal"}\} \\ 1 (\text{DDoS}), & \text{if } \text{Label} \in \{\text{DDoS}, \text{DoS}, \text{Mirai}, \text{Flooding}\} \\ \text{null (Discard)}, & \text{otherwise (Reconnaissance, XSS, etc.)} \end{cases}$$

### 5.2 Phase 1: Online Preprocessing Pipeline

The preprocessing pipeline (`msaad/preprocessing.py`) executes seven sequential incremental operations:

#### 1. Incremental Mean Imputation
For missing or unparseable values, the running mean $\mu_{i, n}$ of feature $i$ across $n$ valid samples is maintained:
$$\mu_{i, n} = \mu_{i, n-1} + \frac{x_{i, n} - \mu_{i, n-1}}{n}$$
Missing entries are replaced with $\mu_{i, n}$ in $O(1)$ time.

#### 2. Online Min-Max Normalization
Maintains dynamic extrema vectors $x_{\min}$ and $x_{\max}$:
$$x_{\min, i} \leftarrow \min(x_{\min, i}, x_{i, n}), \quad x_{\max, i} \leftarrow \max(x_{\max, i}, x_{i, n})$$
$$\hat{x}_{i, n} = \frac{x_{i, n} - x_{\min, i}}{x_{\max, i} - x_{\min, i} + \epsilon}$$

#### 3. Mini-Batch K-Means Augmentation
Maintains a sliding batch of size $B=256$. Upon buffer completion, centroids $\{c_1, \dots, c_k\}$ ($k=8$) are updated via mini-batch Lloyd's algorithm. For each incoming sample $x$, $k$ relative distance features are appended:
$$d_j(x) = \frac{\|x - c_j\|_2}{\max_{1 \le l \le k} \|x - c_l\|_2 + \epsilon}, \quad \forall j \in \{0, \dots, k-1\}$$

#### 4. Online SMOTE via Reservoir Sampling
To balance streaming class distributions without full memory retention:
- Minority instances ($y=1$ or $y=0$) populate a fixed reservoir $\mathcal{R}$ of size $R=200$ via Algorithm R: the $n$-th minority instance replaces a random item in $\mathcal{R}$ with probability $R/n$.
- When the streaming class imbalance ratio $\frac{N_{\text{min}}}{N_{\text{maj}}} < \tau = 1.0$, synthetic vectors are generated:
  $$x_{\text{syn}} = \lambda x_a + (1 - \lambda) x_b, \quad x_a, x_b \sim \text{Uniform}(\mathcal{R}), \; \lambda \sim \text{Uniform}(0, 1)$$

#### 5. Online Information Gain Feature Selection
Quantizes continuous features into $B=10$ empirical intervals. Joint contingency counts $N(X_i \in b, Y = y)$ are incremented online. Periodically (every 500 samples), Information Gain is evaluated against class entropy $H(Y)$:
$$H(Y) = -\sum_{y \in \{0, 1\}} P(y) \log_2 P(y)$$
$$H(Y | X_i) = \sum_{b=1}^{B} P(X_i \in b) \left[ -\sum_{y \in \{0, 1\}} P(y | X_i \in b) \log_2 P(y | X_i \in b) \right]$$
$$IG(X_i) = H(Y) - H(Y | X_i)$$
Features with $IG(X_i) < 0.01$ are dropped from the active representation space.

#### 6. Online Pearson Deduplication via Welford's Covariance
To eliminate multicollinearity without matrix inversion, Welford's vectorized algorithm maintains the streaming mean vector $\bar{x}_n$ and comoment cross-product matrix $C_n \in \mathbb{R}^{d \times d}$:
$$\bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n}$$
$$C_n = C_{n-1} + (x_n - \bar{x}_{n-1})(x_n - \bar{x}_n)^T$$
$$\rho_{ij} = \frac{C_{n, ij}}{\sqrt{C_{n, ii} \cdot C_{n, jj}}}$$
When $|\rho_{ij}| \ge 0.95$, feature $j$ is pruned, eliminating redundant dimensions.

---

### 5.3 Phase 2: Adversarial Attack Simulation & MSAPG

Implemented in `msaad/adversarial.py`, attack generation employs a surrogate neural network $f_\theta: \mathbb{R}^d \to \mathbb{R}^2$ with architecture $\text{Linear}(d, 64) \to \text{ReLU} \to \text{Linear}(64, 32) \to \text{ReLU} \to \text{Linear}(32, 2)$, trained on a sliding FIFO window of 512 samples. 

Gradient attacks are formulated via IBM's ART library:

1. **Fast Gradient Sign Method (FGSM):**
   $$x_{\text{adv}} = \text{clip}_{[0, 1]}\left(x + \epsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}_{\text{CE}}(f_\theta(x), y)\right)\right), \quad \epsilon = 0.1$$
2. **Basic Iterative Method (BIM):**
   $$x^{(0)} = x, \quad x^{(t+1)} = \text{clip}_{x, \epsilon}\left(x^{(t)} + \alpha \cdot \text{sign}\left(\nabla_{x^{(t)}} \mathcal{L}_{\text{CE}}(f_\theta(x^{(t)}), y)\right)\right)$$
   *(Parameters: $\epsilon = 0.1, \alpha = 0.01, T = 10$)*
3. **Projected Gradient Descent (PGD):**
   Identical to BIM with uniform random initialization: $x^{(0)} = x + \mathcal{U}(-\epsilon, \epsilon)$.
4. **Carlini & Wagner $L_2$ (C&W):**
   Optimizes perturbation $\delta = \frac{1}{2}(\tanh(w) + 1) - x$ to solve:
   $$\min_w \|\delta\|_2^2 + c \cdot \max\left( \log [f_\theta(x+\delta)]_y - \max_{i \ne y} \log [f_\theta(x+\delta)]_i, -\kappa \right)$$
   *(Parameters: $\kappa = 0.1$, max iterations $= 20$, batch size $= 32$)*

#### Multi-Source Adversarial Perturbation Generator (MSAPG)
To construct a comprehensive training set for defensive immunization, MSAPG averages the perturbation vectors across all four attacks:
$$x_{\text{MSAPG}} = \frac{1}{4}\left( x_{\text{FGSM}} + x_{\text{BIM}} + x_{\text{PGD}} + x_{\text{CW}} \right)$$

---

### 5.4 Phase 3 (Layer 1): Resilient Adversarial Detection & Purification (RADP)

RADP (`msaad/radp.py`) acts as the primary defensive barrier to prevent corrupted samples from reaching downstream classifiers.

#### Generative Adversarial Network (GAN) Architecture
- **Generator ($G_\phi$):** Projects latent noise $z \sim \mathcal{N}(0, I_{32})$ through hidden dimensions $64 \to 128 \to d$ with LeakyReLU activations and Sigmoid output to map features strictly to $[0, 1]^d$.
- **Discriminator ($D_\psi$):** Accepts candidate feature vectors, passing through $2d \to 64 \to 1$ with Dropout ($p=0.3$) and Sigmoid output, producing probability $D_\psi(x) \in [0, 1]$.
- **Objective:**
  $$\min_\phi \max_\psi \mathbb{E}_{x \sim \mathcal{D}_{\text{adv}}}[\log D_\psi(x)] + \mathbb{E}_{z \sim \mathcal{N}(0, I)}[\log(1 - D_\psi(G_\phi(z)))]$$

#### Resilient Adversarial Detector (RAD)
Operates an incremental online logistic regression model:
$$\hat{p}_{\text{adv}} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
Weight updates follow L2-regularized online SGD with inverse learning rate decay:
$$w \leftarrow w - \eta_t \left( (\hat{p}_{\text{adv}} - y_{\text{adv}}) x + \lambda w \right), \quad \eta_t = \frac{\eta_0}{1 + 0.001 \cdot t}$$
$$\text{Prune Trigger:} \quad \mathcal{I}_{\text{prune}}(x) = \begin{cases} \text{True (Drop sample, emit DDoS=1)}, & \text{if } \hat{p}_{\text{adv}} \ge 0.5 \\ \text{False (Forward to Layer 2)}, & \text{if } \hat{p}_{\text{adv}} < 0.5 \end{cases}$$

---

### 5.5 Phase 3 (Layer 2): Heterogeneous Streaming Ensemble with Drift Adaptation

Implemented in `msaad/classifiers.py`, Layer 2 hosts five streaming models from the River library:

1. **ARF-ADWIN (Adaptive Random Forest):**
   Ensemble of $M=10$ Hoeffding Trees utilizing Online Bootstrap Bagging (Poisson($\lambda=6$)). Each tree incorporates an independent ADWIN drift detector monitoring tracking error. If drift is detected, the affected tree is purged and replaced.
2. **SRP-DDM (Streaming Random Patches):**
   Constructs $M=10$ base trees trained over randomized feature subsets. Concept drift is governed by the Drift Detection Method (DDM), tracking error probability $p_t$ and standard deviation $s_t = \sqrt{p_t(1-p_t)/t}$. Drift triggers when $p_t + s_t \ge p_{\min} + 3 s_{\min}$.
3. **HT-ADWIN (Hoeffding Tree with ADWIN):**
   Incremental decision tree that splits nodes only when the Hoeffding Bound guarantees statistical split superiority:
   $$\epsilon_{\text{HB}} = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$$
   ADWIN monitors external prediction errors and resets the tree upon confirmed drift.
4. **KNN-ADWIN (Streaming K-Nearest Neighbors):**
   Maintains a dynamic exemplar window with $k=5$ neighbors. An external ADWIN detector resets the neighborhood upon distribution shifts.
5. **OLR (Online Logistic Regression):**
   Stochastic gradient descent linear classifier preconditioned with streaming standard scaling and fixed learning rate $\eta=0.01$.

---

### 5.6 Phase 3 (Layer 3): LinUCB Contextual Bandit Routing (MABLUCBE)

Implemented in `msaad/mabtse.py`, Layer 3 arbitrates inference routing across six available arms:
$$\mathcal{A} = \{0: \text{ARF}, \; 1: \text{SRP}, \; 2: \text{HT}, \; 3: \text{KNN}, \; 4: \text{OLR}, \; 5: \text{AUWPAE Ensemble}\}$$

#### 1. Context Construction
Incoming feature dictionaries are converted to an $L_2$-normalized context vector $x_t \in \mathbb{R}^{d_c}$ ($d_c = 16$):
$$x_t \leftarrow \frac{x_t}{\|x_t\|_2 + \epsilon}$$

#### 2. Upper Confidence Bound (UCB) Calculation
Each arm $a \in \mathcal{A}$ maintains a precision matrix $A_a \in \mathbb{R}^{d_c \times d_c}$ (initialized as $I_{d_c}$) and reward-weighted vector $b_a \in \mathbb{R}^{d_c}$ (initialized as $0$).
Ridge regression weights $\hat{\theta}_a$ and the UCB score are computed:
$$\hat{\theta}_a = A_a^{-1} b_a$$
$$\text{UCB}_a(x_t) = \hat{\theta}_a^T x_t + \alpha \sqrt{x_t^T A_a^{-1} x_t}, \quad \alpha = 1.0$$
The arm with maximum UCB is pulled: $a^* = \arg\max_{a \in \mathcal{A}} \text{UCB}_a(x_t)$.

#### 3. AUWPAE Aggregation (Arm 5)
When Arm 5 is selected, individual base model predictions are synthesized via sliding-window ($W=200$) accuracy weighting:
$$w_i = \frac{1}{W} \sum_{k=1}^W \mathbb{I}(\hat{y}_{i, k} = y_k), \quad \forall i \in \{0, \dots, 4\}$$
$$P_{\text{AUWPAE}}(y=1) = \frac{\sum_{i=0}^4 w_i P_i(y=1)}{\sum_{i=0}^4 w_i}$$
$$\hat{y} = \mathbb{I}(P_{\text{AUWPAE}}(y=1) \ge 0.5)$$

#### 4. Sherman-Morrison Incremental Matrix Update
Upon observing binary reward $r_t = \mathbb{I}(\hat{y} = y_t) \in \{0, 1\}$, matrix updates avoid $O(d_c^3)$ inversions via the Sherman-Morrison rank-1 update:
$$A_{a^*}^{-1} \leftarrow A_{a^*}^{-1} - \frac{A_{a^*}^{-1} x_t x_t^T A_{a^*}^{-1}}{1 + x_t^T A_{a^*}^{-1} x_t}, \quad b_{a^*} \leftarrow b_{a^*} + r_t x_t$$
This ensures the parameter update executes in $O(d_c^2)$ time per sample.

---

<div style="page-break-after: always;"></div>

## 6. Empirical Evaluation & Comparative Analysis

### 6.1 Benchmark Datasets
MSAAD was evaluated on two widely benchmarked IoT security datasets:
1. **CICIoT2023 (Canadian Institute for Cybersecurity):** Modern large-scale IoT attack dataset encompassing 33 distinct attack executions across 105 IoT sensor topologies. For binary evaluation, benign traffic maps to Class 0, while volumetric DDoS/DoS flows map to Class 1.
2. **IoTID20 (IoT Network Intrusion Dataset):** Flow statistics captured from smart home hardware (smart bulbs, cameras, gateways) under Mirai and flooding attacks. Filtered to binary normal vs. DDoS instances.

### 6.2 Experimental Setup & Tri-Scenario Protocol
Experiments were conducted on an Intel Core processor with 16 GB RAM using the standardized tri-scenario protocol:
- **Scenario 1 (Original):** Clean streaming test data; evaluates baseline performance without attacks or defense mechanisms.
- **Scenario 2 (Under Attack):** Test traffic perturbed via adversarial algorithms ($\epsilon=0.1$); Layer 1 RADP is disabled to measure classifier vulnerability.
- **Scenario 3 (Recovered):** Adversarial traffic evaluated with Layer 1 RADP active; adversarial samples are identified and pruned prior to classifier exposure.

### 6.3 Quantitative Findings & Discussion

#### Table 1: Comprehensive Performance Results (FGSM Attack, $\epsilon=0.1$)

| Dataset | Evaluation Scenario | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Learn Latency | Throughput | Adversarial Prune Rate |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CICIoT2023** | **1. Original** | 99.500% | 0.99505 | 0.99500 | **0.99500** | 6.048 ms | **132.08 sps** | N/A |
| **CICIoT2023** | **2. Under Attack** | 100.000% | 1.00000 | 1.00000 | **1.00000** | 8.954 ms | 92.92 sps | 0.0% (0 / 200) |
| **CICIoT2023** | **3. Recovered (RADP)** | 100.000% | 1.00000 | 1.00000 | **1.00000** | 16.409 ms | 50.03 sps | **100.0% (200 / 200)** |
| **IoTID20** | **1. Original** | 100.000% | 1.00000 | 1.00000 | **1.00000** | 6.470 ms | **104.53 sps** | N/A |
| **IoTID20** | **2. Under Attack** | 100.000% | 1.00000 | 1.00000 | **1.00000** | 11.879 ms | 79.88 sps | 0.0% (0 / 2000) |
| **IoTID20** | **3. Recovered (RADP)** | 100.000% | 1.00000 | 1.00000 | **1.00000** | 12.129 ms | 76.85 sps | **72.2% (1444 / 2000)** |

#### Key Analytical Insights
1. **Adversarial Resilience:** On `CICIoT2023`, Layer 1 RADP achieved a **100.0% pruning rate**, successfully neutralizing all 200 injected adversarial samples before they could reach Layer 2. On `IoTID20`, RADP pruned **72.2% (1,444 / 2,000)** of attacks; the remaining 27.8% that bypassed Layer 1 were correctly resolved by the Layer 2/3 ensemble, maintaining an overall F1-score of 100%.
2. **Throughput Feasibility:** Baseline throughput reached **132.08 samples/sec** on `CICIoT2023` and **104.53 samples/sec** on `IoTID20`. In the Recovered scenario, the additional GAN and RAD computations resulted in throughputs of **50.03 sps** and **76.85 sps**, respectively. This throughput is well within the requirements for medium-scale IoT edge router protection.
3. **Defense-in-Depth Validation:** While downstream drift-aware ensembles demonstrate inherent resilience against moderate perturbations ($\epsilon=0.1$), the RADP pruner prevents adversarial samples from poisoning the classifiers' online learning updates over prolonged deployments.

---

<div style="page-break-after: always;"></div>

## 7. Conclusion & Future Work

The **MSAAD** framework demonstrates that robust, adaptive defense against adversarial DDoS attacks is achievable within a strictly online streaming architecture. By integrating an incremental seven-stage preprocessing pipeline, a GAN-powered adversarial sample detector (RADP), five drift-protected streaming classifiers, and a LinUCB contextual bandit router, MSAAD resolves the fundamental limitations of static batch-mode NIDS. Empirical results across the `CICIoT2023` and `IoTID20` benchmark datasets confirm near-perfect detection accuracy (99.50%–100.00% F1-score) and effective adversarial pruning (72.2%–100%) under real-time streaming constraints.

### Future Research Directions
1. **Multi-Class Attack Specialization:** Expanding the decision space from binary classification to granular DDoS sub-typologies (e.g., SYN Flood, UDP Flood, Slowloris, HTTP GET Flood).
2. **Hardware Acceleration:** Offloading GAN tensor operations and LinUCB matrix updates to embedded GPUs/NPUs (e.g., NVIDIA Jetson, Edge TPU) to scale throughput beyond 1,000 samples/second.
3. **Federated Streaming Consensus:** Distributing the LinUCB and RADP models across multiple edge gateways using privacy-preserving federated aggregation.
4. **Live Network Telemetry Integration:** Interfacing the streaming pipeline directly with kernel-level packet capture frameworks (e.g., DPDK, eBPF) for live production deployments.

---

## 8. References

1. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *International Conference on Learning Representations (ICLR)*.
2. Carlini, N., & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks. *IEEE Symposium on Security and Privacy (S&P)*, 39–57.
3. Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. *Proceedings of the 19th International Conference on World Wide Web (WWW)*, 661–670.
4. Montiel, J., Halford, M., Mastelini, S. M., Bolmier, G., Sourty, R., Vaysse, R., ... & Bifet, A. (2021). River: Machine Learning for Streaming Data in Python. *Journal of Machine Learning Research (JMLR)*, 22(110), 1–8.
5. Nicolae, M. I., Mathieu, M., Sinn, M., Tran, B. N., Buesser, B., Rawat, A., ... & Forbes, C. (2018). Adversarial Robustness Toolbox v1.2.0. *arXiv preprint arXiv:1807.01069*.
6. Bifet, A., & Gavaldà, R. (2007). Learning from Time-Changing Data with Adaptive Windowing. *SIAM International Conference on Data Mining (SDM)*, 443–448.
7. Gomes, H. M., Bifet, A., Read, J., Barddal, J. P., Enembreck, F., Pfahringer, B., ... & Abdessalem, T. (2017). Adaptive Random Forests for Evolving Data Stream Classification. *Machine Learning*, 106(9), 1469–1495.
8. Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023). CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environments. *Sensors*, 23(13), 5941.
9. Ullah, I., & Mahmoud, Q. H. (2020). A Scheme for Generating a Dataset for Analyzing and Detecting Malicious Traffic in IoT Networks (IoTID20). *IEEE Access*, 8, 147998–148009.
