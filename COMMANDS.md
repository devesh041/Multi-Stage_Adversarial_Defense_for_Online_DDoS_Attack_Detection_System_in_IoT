# MSAAD — Command Reference Guide
# ============================================================
# Every command you can run for the MSAAD framework,
# with a description of what it does.
# ============================================================
#
# BEFORE RUNNING ANYTHING:
#   1. Open PowerShell
#   2. Navigate to the project folder
#   3. Set UTF-8 encoding (required on Windows, do this once per session):
#
#      $env:PYTHONIOENCODING="utf-8"
#
# ============================================================


# ─────────────────────────────────────────────────────────────
# SECTION 1: SETUP COMMANDS
# ─────────────────────────────────────────────────────────────

# [CMD-01] Install all required Python packages
# What it does: Downloads and installs river, torch, scikit-learn,
#               adversarial-robustness-toolbox, pandas, numpy, scipy,
#               and packaging from PyPI.
pip install pandas river torch scikit-learn adversarial-robustness-toolbox numpy scipy packaging

# [CMD-02] Verify all packages import correctly
# What it does: Runs a quick import test on all 8 MSAAD modules.
#               Prints "ALL IMPORTS SUCCESSFUL" if everything is OK.
python -c "import sys; sys.path.insert(0, '.'); from msaad.config import *; from msaad.data_loader import stream_ciciot23; from msaad.preprocessing import OnlinePreprocessingPipeline; from msaad.adversarial import AdversarialAttackGenerator; from msaad.radp import RADPLayer; from msaad.classifiers import Layer2Classifiers; from msaad.mabtse import MABTSERouter; from msaad.evaluation import MSAADSystem; print('ALL IMPORTS SUCCESSFUL')"


# ─────────────────────────────────────────────────────────────
# SECTION 2: QUICK / SMOKE TEST COMMANDS
# ─────────────────────────────────────────────────────────────

# [CMD-03] Fast smoke test — run_all.py (recommended first run)
# What it does: Runs the full pipeline end-to-end on CICIoT23 with
#               200 samples and FGSM attack only. Takes ~30 seconds.
#               Prints results to terminal AND saves log to results/ folder.
python run_all.py --fast

# [CMD-04] Fast smoke test — via main.py (alternative)
# What it does: Same as above but calls the lower-level main.py directly.
#               Evaluates all 3 scenarios (Original, Under Attack, Recovered)
#               with 200 samples and FGSM attack on CICIoT23.
python -m msaad.main --dataset ciciot23 --samples 200 --warmup 150 --attack fgsm --scenario all


# ─────────────────────────────────────────────────────────────
# SECTION 3: FULL EVALUATION — run_all.py
# ─────────────────────────────────────────────────────────────

# [CMD-05] Run EVERYTHING at once (all datasets, all attacks, all scenarios)
# What it does: Evaluates CICIoT23 and IoTID20 datasets.
#               Runs all 4 attack methods: FGSM, BIM, PGD, C&W.
#               Runs all 3 scenarios per attack: Original, Under Attack, Recovered.
#               Logs all output to both terminal and a timestamped .log file.
#               Saves one CSV per dataset to the results/ folder.
#               ESTIMATED TIME: 60-120 minutes (C&W is slowest).
python run_all.py

# [CMD-06] Run both datasets with only FGSM attack
# What it does: Same as CMD-05 but skips BIM, PGD, and C&W.
#               Faster — good for comparing datasets without the slow attacks.
#               ESTIMATED TIME: 10-20 minutes.
python run_all.py --attacks fgsm

# [CMD-07] Run both datasets with FGSM and PGD (skip BIM and C&W)
# What it does: Evaluates the two most common and contrasting attacks:
#               FGSM (single-step, fast) and PGD (multi-step, stronger).
#               ESTIMATED TIME: 20-30 minutes.
python run_all.py --attacks fgsm pgd

# [CMD-08] Run both datasets with all attacks on fewer samples (faster)
# What it does: Uses 2000 samples per scenario instead of the default 10000.
#               Gives all 4 attack results much faster for quick comparisons.
#               ESTIMATED TIME: 15-25 minutes.
python run_all.py --samples 2000 --warmup 300

# [CMD-09] Run only CICIoT23 dataset with all 4 attacks
# What it does: Skips IoTID20. Evaluates only the CICIoT23 train/test splits.
#               ESTIMATED TIME: 30-60 minutes.
python run_all.py --datasets ciciot23

# [CMD-10] Run only IoTID20 dataset with all 4 attacks
# What it does: Skips CICIoT23. Evaluates the IoT Network Intrusion dataset.
#               ESTIMATED TIME: 30-60 minutes.
python run_all.py --datasets iotid20

# [CMD-11] Run cybersecurity_attacks.csv dataset
# What it does: Evaluates the third dataset (cybersecurity_attacks.csv)
#               which has fewer numeric features (5 columns).
python run_all.py --datasets cybersecurity --attacks fgsm --samples 2000

# [CMD-12] Run all three datasets together
# What it does: Evaluates CICIoT23, IoTID20, and cybersecurity_attacks.csv.
#               ESTIMATED TIME: 90-180 minutes (very long run).
python run_all.py --datasets ciciot23 iotid20 cybersecurity

# [CMD-13] Custom full run — specify everything yourself
# What it does: Full control over dataset, attacks, samples, and warmup count.
#               Example: 5000 samples, 500 warmup, FGSM+BIM only, CICIoT23.
python run_all.py --datasets ciciot23 --attacks fgsm bim --samples 5000 --warmup 500


# ─────────────────────────────────────────────────────────────
# SECTION 4: INDIVIDUAL SCENARIO COMMANDS — msaad.main
# ─────────────────────────────────────────────────────────────

# [CMD-14] Run ONLY the "Original" scenario (baseline, no attack)
# What it does: Evaluates clean data performance — no adversarial injection,
#               no RADP defense. This is the baseline accuracy.
python -m msaad.main --dataset ciciot23 --samples 5000 --warmup 500 --attack fgsm --scenario original

# [CMD-15] Run ONLY the "Under Attack" scenario
# What it does: Injects FGSM adversarial perturbations into every sample
#               before the classifier sees it. No RADP defense active.
#               Shows how much accuracy drops under attack.
python -m msaad.main --dataset ciciot23 --samples 5000 --warmup 500 --attack fgsm --scenario attack

# [CMD-16] Run ONLY the "Recovered" scenario (RADP defense active)
# What it does: Injects adversarial perturbations BUT the RADP layer
#               detects and prunes them before the classifiers see them.
#               Shows how much accuracy is recovered by the defense.
python -m msaad.main --dataset ciciot23 --samples 5000 --warmup 500 --attack fgsm --scenario recovered

# [CMD-17] Run all 3 scenarios with BIM attack on CICIoT23
# What it does: BIM (Basic Iterative Method) — a stronger, iterative version
#               of FGSM. Runs all 3 scenarios to compare attack vs recovery.
python -m msaad.main --dataset ciciot23 --samples 5000 --warmup 500 --attack bim --scenario all

# [CMD-18] Run all 3 scenarios with PGD attack on CICIoT23
# What it does: PGD (Projected Gradient Descent) — the strongest gradient
#               attack. Same as BIM but with random starting perturbation.
python -m msaad.main --dataset ciciot23 --samples 5000 --warmup 500 --attack pgd --scenario all

# [CMD-19] Run all 3 scenarios with C&W attack on CICIoT23
# What it does: Carlini & Wagner L2 — optimization-based attack that finds
#               minimum-distortion adversarial examples. Slowest but most
#               powerful. Reduce samples if too slow.
python -m msaad.main --dataset ciciot23 --samples 1000 --warmup 300 --attack cw --scenario all

# [CMD-20] Run all 4 attacks one-by-one on IoTID20
# What it does: Evaluates the IoTID20 dataset under each of the 4 attacks.
python -m msaad.main --dataset iotid20 --samples 5000 --warmup 500 --attack fgsm --scenario all
python -m msaad.main --dataset iotid20 --samples 5000 --warmup 500 --attack bim  --scenario all
python -m msaad.main --dataset iotid20 --samples 5000 --warmup 500 --attack pgd  --scenario all
python -m msaad.main --dataset iotid20 --samples 1000 --warmup 300 --attack cw   --scenario all


# ─────────────────────────────────────────────────────────────
# SECTION 5: SAVING OUTPUT TO A LOG FILE MANUALLY
# ─────────────────────────────────────────────────────────────
# NOTE: run_all.py saves logs automatically.
#       For msaad.main, use PowerShell's Tee-Object to log manually.

# [CMD-21] Run main.py and save terminal output to a log file (PowerShell)
# What it does: Runs the evaluation and saves everything printed to both
#               the terminal AND a file called output.log in results/.
python -m msaad.main --dataset ciciot23 --samples 5000 --attack fgsm --scenario all | Tee-Object -FilePath "results\output.log"

# [CMD-22] Run main.py and append output to an existing log file
# What it does: Like CMD-21 but appends to the file instead of overwriting.
python -m msaad.main --dataset ciciot23 --samples 5000 --attack bim --scenario all | Tee-Object -FilePath "results\output.log" -Append

# [CMD-23] Redirect all output (stdout + stderr) to a file
# What it does: Saves ALL output including warnings and errors to a log file.
#               Nothing is printed to the terminal (runs silently).
python -m msaad.main --dataset ciciot23 --samples 5000 --attack fgsm --scenario all > results\full_output.log 2>&1


# ─────────────────────────────────────────────────────────────
# SECTION 6: VIEWING RESULTS
# ─────────────────────────────────────────────────────────────

# [CMD-24] List all result files generated
# What it does: Shows all CSV and log files in the results/ folder.
dir results\

# [CMD-25] View a CSV result file in the terminal
# What it does: Prints the evaluation CSV file directly in the terminal.
#               Each row = one evaluation scenario with all 6 metrics.
type results\evaluation_results_ciciot23.csv

# [CMD-26] View the most recent log file
# What it does: Opens the latest .log file in Notepad for easy reading.
#               Replace the filename with your actual timestamped log file.
notepad results\msaad_run_20260905_170000.log


# ─────────────────────────────────────────────────────────────
# SECTION 7: RECOMMENDED RUNS (in order of time)
# ─────────────────────────────────────────────────────────────
#
#  TIME      COMMAND
#  --------  -------------------------------------------------------
#  ~30 sec   python run_all.py --fast
#  ~5 min    python run_all.py --datasets ciciot23 --attacks fgsm --samples 2000
#  ~15 min   python run_all.py --attacks fgsm bim --samples 2000
#  ~30 min   python run_all.py --attacks fgsm bim pgd --samples 5000
#  ~60 min   python run_all.py --samples 5000
#  ~2+ hrs   python run_all.py   (full: 10000 samples, all attacks, all datasets)
#
