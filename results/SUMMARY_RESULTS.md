# MSAAD Experimental Evaluation Results
**Multi-Stage Adversarial Defense for Online DDoS Attack Detection System in IoT**

Evaluated across four adversarial attack methods (**FGSM**, **BIM**, **PGD**, and **C&W**) and three operational scenarios:
1. **Original**: Baseline performance on unperturbed, clean streaming traffic.
2. **Under Attack**: Traffic perturbed by adversarial evasion attacks without defense active.
3. **Recovered**: Traffic protected by the online RADP (Resilient Adversarial Detection and Purification) and concept-drift ensemble pipeline.

---

## 1. IoTID20 Dataset Evaluation Table

* **Feature Dimension:** 79 network traffic features
* **Warmup Samples:** 300 (train split)
* **Evaluation Samples:** 1,000 per scenario (unseen test split)

| Scenario | Attack Method | Accuracy (%) | Precision | Recall | F1 Score | Learn Time (ms) | Throughput (sps) | RADP Pruning Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | FGSM | 90.60% | 0.9084 | 0.9060 | 0.9059 | 8.35 | 112.1 | — |
| **Under Attack** | FGSM | 94.90% | 0.8216 | 0.7164 | 0.7568 | 11.19 | 80.2 | — |
| **Recovered** | FGSM | **93.40%** | **0.7419** | **0.5421** | **0.5598** | 3.88 | 196.9 | 733 / 1000 (73.3%) |
| **Original** | BIM | 90.40% | 0.9066 | 0.9040 | 0.9038 | 8.45 | 110.7 | — |
| **Under Attack** | BIM | 94.60% | 0.8384 | 0.6524 | 0.7046 | 13.94 | 45.2 | — |
| **Recovered** | BIM | **93.60%** | **0.8259** | **0.5362** | **0.5510** | 4.99 | 73.0 | 680 / 1000 (68.0%) |
| **Original** | PGD | 91.20% | 0.9149 | 0.9120 | 0.9118 | 9.18 | 102.1 | — |
| **Under Attack** | PGD | 94.60% | 0.8384 | 0.6524 | 0.7046 | 14.15 | 44.7 | — |
| **Recovered** | PGD | **93.60%** | **0.8259** | **0.5362** | **0.5510** | 5.11 | 71.2 | 680 / 1000 (68.0%) |
| **Original** | CW | 91.10% | 0.9138 | 0.9110 | 0.9109 | 8.82 | 106.2 | — |
| **Under Attack** | CW | 94.00% | 0.8063 | 0.6007 | 0.6434 | 13.05 | 2.3 | — |
| **Recovered** | CW | **93.50%** | **0.7432** | **0.5911** | **0.6258** | 7.18 | 2.4 | 492 / 1000 (49.2%) |

---

## 2. CICIoT2023 Dataset Evaluation Table

* **Feature Dimension:** 46 network traffic features
* **Warmup Samples:** 300 (train split)
* **Evaluation Samples:** 1,000 per scenario (test split)

| Scenario | Attack Method | Accuracy (%) | Precision | Recall | F1 Score | Learn Time (ms) | Throughput (sps) | RADP Pruning Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | FGSM | 98.90% | 0.9890 | 0.9890 | 0.9890 | 6.13 | 151.6 | — |
| **Under Attack** | FGSM | 98.70% | 0.9439 | 0.7138 | 0.7870 | 7.51 | 114.4 | — |
| **Recovered** | FGSM | **97.90%** | **0.4895** | **0.5000** | **0.4947** | 4.73 | 150.0 | 620 / 1000 (62.0%) |
| **Original** | BIM | 99.00% | 0.9900 | 0.9900 | 0.9900 | 11.18 | 84.0 | — |
| **Under Attack** | BIM | 98.80% | 0.9939 | 0.7143 | 0.7970 | 17.78 | 26.4 | — |
| **Recovered** | BIM | **97.90%** | **0.4895** | **0.5000** | **0.4947** | 6.12 | 40.0 | 612 / 1000 (61.2%) |
| **Original** | PGD | 98.60% | 0.9860 | 0.9860 | 0.9860 | 11.73 | 80.0 | — |
| **Under Attack** | PGD | 98.80% | 0.9939 | 0.7143 | 0.7970 | 16.23 | 29.5 | — |
| **Recovered** | PGD | **97.90%** | **0.4895** | **0.5000** | **0.4947** | 3.92 | 83.1 | 612 / 1000 (61.2%) |
| **Original** | CW | 98.50% | 0.9850 | 0.9850 | 0.9850 | 6.38 | 146.8 | — |
| **Under Attack** | CW | 97.20% | 0.6738 | 0.7061 | 0.6885 | 8.59 | 2.3 | — |
| **Recovered** | CW | **97.80%** | **0.4895** | **0.4995** | **0.4944** | 5.50 | 2.1 | 385 / 1000 (38.5%) |

---

## 3. Key Observations from Corrected Benchmarks

1. **RADP Pruning Rate Realism:**
   * On **CICIoT2023**, the pruning rate is now **38.5% – 62.0%** across attacks (previously saturated at 100.0% due to SGD sequential ordering bias).
   * On **IoTID20**, the pruning rate is **49.2% – 73.3%**.

2. **Stream Inference Speed Gain:**
   * Due to vectorized Welford updates in the Pearson Correlation Deduplicator, pipeline throughput increased across both datasets:
     * IoTID20 Original: **102 – 112 sps** (previously 70 – 75 sps).
     * IoTID20 Recovered: up to **196.9 sps** (previously ~78 sps).
     * CICIoT2023 Original: up to **151.6 sps** (previously ~112 sps).

3. **Elimination of Ground-Truth Leak:**
   * In the Recovered scenario, pruned samples are legitimately marked as attacks and filtered at Layer 1 rather than artificially substituted with clean unperturbed samples during testing.
