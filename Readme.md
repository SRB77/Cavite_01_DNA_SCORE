# Developer DNA Matrix (DDM)
> **An AI-driven behavioral intelligence framework that evaluates software developers through deep analysis of their GitHub proof-of-work artifacts — not resumes.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.0-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)
[![Course](https://img.shields.io/badge/Course-Advanced%20ML%20%26%20DL-purple?style=flat)]()
[![Phase](https://img.shields.io/badge/Phase-3%20Complete-brightgreen?style=flat)]()
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)]()

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Problem Statement](#2-problem-statement)
3. [Phase Roadmap](#3-phase-roadmap)
4. [The Developer DNA Score (DDS)](#4-the-developer-dna-score-dds)
5. [System Architecture](#5-system-architecture)
6. [Dataset](#6-dataset)
7. [Repository Structure](#7-repository-structure)
8. [Notebook Guide](#8-notebook-guide)
9. [Feature Engineering](#9-feature-engineering)
10. [Model Results — All Phases](#10-model-results--all-phases)
11. [Continuous DNA Score Output (Phase 3)](#11-continuous-dna-score-output-phase-3)
12. [Key Findings](#12-key-findings)
13. [How to Run](#13-how-to-run)
14. [Requirements](#14-requirements)
15. [References](#15-references)

---

## 1. Quick Start

### Option A — One-command setup *(recommended)*

```bash
git clone https://github.com/yourusername/developer-dna-matrix.git
cd developer-dna-matrix
bash setup.sh
```

Checks Python version → creates venv → installs dependencies → runs the full Phase 3 pipeline → saves all outputs to `outputs/`.

### Option B — Docker *(zero environment assumptions)*

```bash
docker build -t ddm-phase3 .
docker run --rm -v $(pwd)/outputs:/app/outputs ddm-phase3
```

### Option C — Jupyter notebooks *(step-by-step)*

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order: `01_eda` → `02_FeatureEng` → `03_modelapplication` → `04_DeepLearning` → `05_phase3_DDM`

### Option D — Google Colab

1. Clone this repository and open any notebook in Google Colab
2. Upload the required input CSV when prompted (see [Notebook Guide](#8-notebook-guide))
3. Run all cells in order
4. Download outputs at the end of each notebook

---

## 2. Problem Statement

Traditional hiring evaluates developers through resumes and interviews — both of which measure **communication ability**, not **execution capability**. A developer can write an impressive resume without having built anything meaningful. Conversely, the most capable engineers often have sparse resumes but rich GitHub histories.

**The gap:** There is no standardised, objective, behavior-centric system that quantifies developer competence from real-world proof-of-work artifacts.

**DDM solves this** by treating a developer's GitHub history as a behavioral fingerprint. Every commit, pull request, language choice, test file, CI configuration, and collaboration pattern reveals something quantifiable about who that developer is and how they work.

### What makes DDM different from GitHub statistics dashboards

| GitHub Statistics | Developer DNA Matrix |
|---|---|
| Shows raw numbers (commits, stars) | Derives behavioral rates (velocity, decay) |
| No tier classification | 3-tier classification with ensemble F1 > 0.81 |
| No quality signals | Tests, CI/CD, commit quality scored |
| Static snapshot | Temporal growth trajectory modeled |
| No comparative ranking | Continuous 0–100 score + confidence output |

---

## 3. Phase Roadmap

| Phase | Model(s) | Best F1 | Key Contribution | Status |
|---|---|---|---|---|
| **1** | LR · SVM · RF | **0.8116** (SVM) | ML baseline, 28 features, failure analysis | ✅ Complete |
| **2** | MLP (128→64, BatchNorm) | **0.8164** | Deep learning, +0.005 over Phase 1 | ✅ Complete |
| **3** | GBM + Ensemble | **see outputs/** | Better data · 33 features · continuous score | ✅ Complete |

### Phase 3 Core Insight

> Phase 2 showed only +0.005 improvement over Phase 1 despite a more powerful model.  
> **Root cause: data quality, not model capacity, was the bottleneck.**  
> Phase 3 fixes this with a richer dataset, 5 new features, and a soft-vote ensemble.

---

## 4. The Developer DNA Score (DDS)

The DDS is computed as a weighted sum of 5 behavioral dimensions:

```
DDS = 0.30 × Proof-of-Work
    + 0.25 × Skill Genome
    + 0.20 × Execution Pattern
    + 0.15 × Thinking Blueprint
    + 0.10 × Growth Signature
```

### The 5 DNA Dimensions

#### Proof-of-Work Core (30%)
*What has this developer actually built and shipped?*

| Signal | Description |
|---|---|
| `total_commits` | Raw execution volume across all repositories |
| `commits_last_90d` | Recent activity — still actively coding? |
| `pull_requests_merged` | Code quality gate — PRs reviewed and accepted |
| `issues_closed` | Problem-solving evidence — task completion ability |
| `stars_received` | Social proof — real-world value creation |

#### Skill Genome (25%)
*How technically broad and deep is this developer?*

| Signal | Description |
|---|---|
| `language_entropy` | Shannon entropy H = −Σ pᵢ log₂(pᵢ) of language distribution |
| `language_count` | Number of distinct programming languages used |
| `framework_count` | Framework breadth — React, PyTorch, Spring etc. |
| `primary_language` | Dominant technical specialisation |

#### Execution Pattern (20%)
*Does this developer write code professionally?*

| Signal | Description |
|---|---|
| `has_readme_pct` | % of repos with README — documentation discipline |
| `has_tests_pct` | % of repos with test files — engineering maturity |
| `has_ci_pct` | % of repos with CI/CD config — DevOps awareness |
| `commit_message_avg_len` | Commit message quality — communication habit |

#### Thinking Blueprint (15%)
*Does this developer think architecturally?*

| Signal | Description |
|---|---|
| `fork_to_original_ratio` | Creator vs. consumer ratio |
| `languages_per_repo_avg` | Full-stack / systems thinking signal |
| `avg_repo_description_len` | Architectural communication ability |

#### Growth Signature (10%)
*Is this developer growing or stagnating?*

| Signal | Description |
|---|---|
| `commit_trend_slope` | Linear regression slope on monthly commit history |
| `activity_decay_lambda` | Exponential decay λ = −ln(A_recent/A_old)/Δt |
| `years_active` | Career length normaliser |

---

## 5. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RAW BEHAVIORAL DATA                           │
│   Phase 1/2: 6,000 developer profiles (standard distribution)        │
│   Phase 3:   7,200+ profiles — standard + hybrid + boundary + noise  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│              NOTEBOOK 1 — EDA                                        │
│  • Statistical characterisation (skewness, power-law, entropy)       │
│  • Outlier detection (IQR 3× method)                                 │
│  • Correlation analysis (Pearson heatmap)                            │
│  • Class balance check (Gini impurity)                               │
│  • t-SNE + PCA dimensionality reduction                              │
│  • Activity decay curve fitting                                      │
│  OUTPUT: github_dna_processed.csv                                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│              NOTEBOOK 2 — FEATURE ENGINEERING                        │
│  Phase 1/2:  28 features across 5 DNA dimensions                     │
│  Phase 3:   +5 new features → 33 total                               │
│                                                                      │
│  Group 1: Log transforms    → log1p(commits, stars, PRs, issues)     │
│  Group 2: Rate features     → velocity, recency, repos_per_year      │
│  Group 3: Quality composites→ execution_quality, impact_weight       │
│  Group 4: Growth features   → growth_momentum, consistency_index     │
│  Group 5: Phase 3 new       → consistency_score, tech_diversity,     │
│                               quality_volume_ratio, collab_index,    │
│                               recent_activity_ratio                  │
│  StandardScaler: fit on train only — no data leakage                 │
│  OUTPUT: X_features.csv, y_labels.csv, selected_features.json       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│              NOTEBOOK 3 — PHASE 1: ML BASELINE                       │
│                                                                      │
│  Model 1: Logistic Regression (linear baseline)                      │
│           F1 = 0.8066 — establishes linear ceiling                   │
│                    │                                                  │
│                    ▼                                                  │
│  Model 2: SVM with RBF Kernel (tuned via GridSearchCV)               │
│           F1 = 0.8116 ★ PHASE 1 WINNER                               │
│           K(x,z) = exp(−γ||x−z||²)                                  │
│                    │                                                  │
│                    ▼                                                  │
│  Model 3: Random Forest (300 trees, GridSearchCV tuned)              │
│           F1 = 0.8007                                                │
│                    │                                                  │
│                    ▼                                                  │
│  Failure Analysis → Intermediate class misclassification diagnosed   │
│  OUTPUT: model_comparison_results.csv, figures                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│              NOTEBOOK 4 — PHASE 2: DEEP LEARNING                     │
│                                                                      │
│  MLP Architecture:                                                   │
│    Input(33) → Dense(128) → BatchNorm → Dropout(0.3)                │
│             → Dense(64)  → BatchNorm → Dropout(0.2)                 │
│             → Softmax(3)                                             │
│  Optimizer: Adam · Early stopping · Class-weighted loss              │
│  F1 = 0.8164 ★ PHASE 2 WINNER (+0.005 over Phase 1)                 │
│                                                                      │
│  Key finding: minimal gain → problem is data, not model capacity     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│              NOTEBOOK 5 — PHASE 3: ENSEMBLE + CONTINUOUS SCORE       │
│                                                                      │
│  GBM (400 trees, depth=5, subsample=0.85)                           │
│       +                                                              │
│  MLP (128→64, early stopping)                                        │
│       +                                                              │
│  SVM RBF (CalibratedClassifierCV)                                    │
│       │                                                              │
│       └─→ Soft Vote (weights 3 · 2 · 2) ★ PHASE 3 WINNER           │
│                                                                      │
│  OUTPUT: dna_score (0–100) · tier · confidence · per-class probs    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│              DEVELOPER DNA SCORE + TIER + CONFIDENCE                 │
│                                                                      │
│   Beginner (0)       Intermediate (1)       Expert (2)              │
│   Score: 0–33        Score: 33–60           Score: 60–100           │
│                                                                      │
│   + confidence (0–1) + prob_beginner + prob_intermediate + prob_expert│
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. Dataset

### Phase 1 / 2 Dataset

| Property | Value |
|---|---|
| Total rows | 6,000 developer profiles |
| Total columns | 29 (raw) → 28 features after engineering |
| Null values | 0 |
| Duplicate developer IDs | 0 |
| Class balance | 2,000 per tier (balanced) |
| Source | Realistic synthetic — trimodal latent skill distribution |

### Phase 3 Improved Dataset

| Profile Type | Count | Purpose |
|---|---|---|
| Standard (per tier × 3) | 6,000 | Core distribution |
| Hybrid: high volume, low quality | 800 | Fix Phase 1 failure mode |
| Hybrid: low volume, high quality | 800 | Fix Phase 1 failure mode |
| Boundary: Beginner ↔ Intermediate | 400 | Ambiguous zone coverage |
| Boundary: Intermediate ↔ Expert | 400 | Ambiguous zone coverage |
| Inconsistent (noise) | 200 | Real-world messiness |
| **Total** | **~7,200** | Balanced across tiers |

### Why This Dataset Design

A purely scraped GitHub dataset was evaluated but rejected — it lacked commit-level signals which are the core DNA signals. The synthetic dataset was constructed with:

- **Trimodal latent skill distribution** — a hidden skill score (0–1) drives all features, creating natural class overlap
- **8% label noise** — deliberately mislabels borderline cases to simulate real-world ambiguity
- **Log-normal feature distributions** — commits and stars follow Pareto distributions, not uniform
- **Phase 3 hybrid profiles** — directly targets the failure modes identified in Phase 1/2

### Key Statistical Findings from EDA

| Feature | Skewness | Finding | Action |
|---|---|---|---|
| `total_commits` | 3.2 | Power-law (Pareto) — top 20% make 80% of commits | log1p() transform |
| `stars_received` | 4.1 | Extreme right tail — viral projects distort raw signal | log1p() transform |
| `language_entropy` | 0.3 | Near-normal — Shannon entropy is naturally bounded | No transform needed |
| `has_tests_pct` | 0.8 | Bimodal — devs either test everything or nothing | Kept raw |
| `activity_decay_lambda` | 0.2 | Near-normal — lambda centered at 0 | Kept raw |

---

## 7. Repository Structure

```
developer-dna-matrix/
│
├── README.md                            ← You are here
├── requirements.txt                     ← All dependencies, pinned versions
├── setup.sh                             ← One-command reproducibility script
├── Dockerfile                           ← Containerized environment
├── .gitignore
├── LICENSE
│
├── data/
│   ├── github_dna_features.csv          ← Phase 1/2 dataset (6,000 rows × 29 cols)
│   ├── github_dna_phase3_raw.csv        ← Phase 3 improved dataset (7,200+ rows)
│   ├── github_dna_processed.csv         ← After EDA cleaning (Notebook 1 output)
│   ├── X_features.csv                   ← Feature matrix (model input)
│   ├── y_labels.csv                     ← Target labels (developer_tier)
│   ├── selected_features.json           ← Final feature names list
│   ├── model_comparison_results.csv     ← Phase 1/2 results table
│   ├── phase3_results.csv               ← Phase 3 all-model results
│   ├── phase3_continuous_output.csv     ← Score + Tier + Confidence predictions
│   └── new_users_predictions.csv        ← Output from run_new_dataset.py
│
├── notebooks/
│   ├── 01_eda.ipynb                     ← Exploratory Data Analysis
│   ├── 02_FeatureEng.ipynb              ← Feature Engineering (28 → 33 features)
│   ├── 03_modelapplication.ipynb        ← Phase 1: LR + SVM + RF
│   ├── 04_DeepLearning.ipynb            ← Phase 2: MLP
│   └── 05_phase3_DDM.ipynb              ← Phase 3: GBM + Ensemble + Continuous Score
│
├── src/
│   ├── features.py                      ← Feature engineering functions
│   ├── model.py                         ← Model training & evaluation
│   ├── dl_model.py                      ← MLP architecture (Phase 2)
│   └── utils.py                         ← Shared utilities
│
├── phase3_pipeline.py                   ← Full Phase 3 standalone pipeline
├── run_new_dataset.py                   ← Score new developers from CSV
│
├── outputs/
│   ├── phase3_confusion_matrix.png      ← GBM vs MLP vs Ensemble
│   ├── cross_phase_comparison.png       ← F1 across all 3 phases
│   ├── continuous_score_distribution.png
│   ├── feature_importance_phase3.png    ← Top-20 GBM features
│   └── DDM_Phase3_Architecture.pdf      ← Downloadable architecture diagram
│
├── figures/
│   ├── eda_01_class_distribution.png
│   ├── eda_02_distributions.png
│   ├── eda_03_correlation.png
│   ├── eda_04_outliers.png
│   ├── eda_05_language_entropy.png
│   ├── eda_06_activity_decay.png
│   ├── eda_07_execution_pattern.png
│   ├── eda_08_tsne_pca.png
│   ├── fe_01_log_transforms.png
│   ├── fe_02_rate_features.png
│   ├── fe_03_quality_features.png
│   ├── fe_04_growth_features.png
│   ├── fe_05_feature_correlation.png
│   ├── fe_06_importance_preview.png
│   ├── model_01_comparison.png
│   ├── model_02_confusion_matrices.png
│   └── model_03_feature_importance.png
│
└── reports/
    ├── DDM_Phase1_Report.pdf
    └── DDM_Phase3_Report.pdf
```

---

## 8. Notebook Guide

### Notebook 1 — EDA (`01_eda.ipynb`)

**Input:** `data/github_dna_features.csv`  
**Output:** `data/github_dna_processed.csv`

| Cell | What it does | Rubric criterion |
|---|---|---|
| 1 | Imports and global plot style | — |
| 2 | Load dataset and sanity check | Dataset Quality |
| 3 | Statistical summary with interpretation | Dataset Quality |
| 4 | Target label distribution + Gini impurity | Dataset Quality |
| 5 | Power-law detection — skewness + kurtosis for 6 features | Dataset Quality (Score 8) |
| 6 | Pearson correlation heatmap + multicollinearity detection | Feature Engineering |
| 7 | Outlier detection via IQR 3× method | Dataset Quality |
| 8 | Language entropy histogram + Pearson r with DDS | Feature Engineering |
| 9 | Activity decay λ distribution + simulated curves | Theoretical Rigor |
| 10 | Execution pattern by tier (readme, tests, CI) | Dataset Quality |
| 11 | t-SNE + PCA scree plot | Feature Engineering (Score 8) |
| 12 | EDA summary dashboard | — |
| 13 | Preprocessing + save `github_dna_processed.csv` | Dataset Quality |

**Key EDA findings that drove modeling decisions:**
- `total_commits` skewness = 3.2 → log1p() applied in Notebook 2
- t-SNE showed **non-linear** tier separation → justified SVM RBF kernel over LR
- Gini impurity = 0.667 → no class imbalance handling needed
- Expert λ < 0 (growing) in 68% of cases → decay lambda is strong tier predictor

---

### Notebook 2 — Feature Engineering (`02_FeatureEng.ipynb`)

**Input:** `data/github_dna_processed.csv`  
**Output:** `data/X_features.csv`, `data/y_labels.csv`, `data/selected_features.json`

| Cell | What it does | Features created |
|---|---|---|
| 1 | Imports | — |
| 2 | Load processed dataset | — |
| 3 | Map existing columns to DNA dimensions | — |
| 4 | Group 1: Log transforms | `log_total_commits`, `log_stars_received`, `log_pull_requests_merged`, `log_issues_closed`, `log_total_repos` |
| 5 | Group 2: Rate features | `commit_velocity`, `recency_score`, `repos_per_year`, `pr_merge_rate` |
| 6 | Group 3: Quality composites | `execution_quality`, `language_count`, `impact_weight`, `profile_completeness` |
| 7 | Group 4: Growth & consistency | `growth_momentum`, `consistency_index`, `primary_language_encoded` |
| 8 | Complete feature inventory | Summary of all 28 features |
| 9 | Correlation check on new features | Validates no new multicollinearity |
| 10 | Final feature selection (X and y) | Drop leakage, text, target columns |
| 11 | Feature importance preview | Validates engineered features add signal |
| 12 | Save all outputs | — |

**Features dropped and why:**

| Column dropped | Reason |
|---|---|
| `developer_dna_score` | **DATA LEAKAGE** — this is the ground truth answer |
| `developer_tier` | **TARGET VARIABLE** — goes into y, not X |
| `developer_id` | Text identifier — not numeric |
| `languages_used` | Raw text string — models cannot parse |
| `primary_language` | Replaced by `primary_language_encoded` |
| `total_commits` | Replaced by `log_total_commits` (de-skewed) |
| `commits_last_90d` | Replaced by `recency_score` (relative metric) |
| `stars_received` | Replaced by `log_stars_received` |
| `pull_requests_merged` | Replaced by `log_pull_requests_merged` |
| `issues_closed` | Replaced by `log_issues_closed` |
| `account_age_days` | Replaced by `years_active` (cleaner unit) |

---

### Notebook 3 — Phase 1: Model Application (`03_modelapplication.ipynb`)

**Input:** `data/X_features.csv`, `data/y_labels.csv`, `data/selected_features.json`  
**Output:** `data/model_comparison_results.csv`, 3 figures

| Cell | What it does | Rubric criterion |
|---|---|---|
| 1 | Imports | — |
| 2 | Load data from Notebook 2 | — |
| 3 | Train/test split (80/20 stratified) + StandardScaler | Theoretical Rigor |
| 4 | Logistic Regression (linear baseline) | Model Application Score 7 |
| 5 | SVM RBF — base + GridSearchCV tuning | Model Application Score 8 |
| 6 | Random Forest — weak baseline (n=30, depth=4) | Model Application Score 7 |
| 7 | Random Forest — tuned GridSearchCV (36 combos × 5 folds) | Model Application Score 8 |
| 8 | 3-model comparison table + bar chart | Model Application Score 9 |
| 9 | Confusion matrices — all 3 models | Model Application Score 9 |
| 10 | Feature importance (permutation for SVM + RF reference) | Model Application Score 9 |
| 11 | Failure analysis — 5 cases explained mathematically | Model Application Score 10 |
| 12 | Final summary + download all outputs | — |

---

### Notebook 4 — Phase 2: Deep Learning (`04_DeepLearning.ipynb`)

**Input:** `data/X_features.csv`, `data/y_labels.csv`  
**Output:** trained MLP, Phase 2 confusion matrix

| Cell | What it does |
|---|---|
| 1–2 | Imports, load data |
| 3 | MLP architecture: Input(28) → Dense(128) → BatchNorm → Dropout(0.3) → Dense(64) → BatchNorm → Dropout(0.2) → Softmax(3) |
| 4 | Training: Adam · early stopping · class-weighted loss · 80/10/10 split |
| 5 | Evaluation: F1 = 0.8164 — per-class breakdown |
| 6 | Comparison vs Phase 1 SVM |
| 7 | Failure analysis: Intermediate still the weakest class |
| 8 | Conclusion: minimal gain → data is the bottleneck, not model capacity |

---

### Notebook 5 — Phase 3: Ensemble + Continuous Score (`05_phase3_DDM.ipynb`)

**Input:** `data/github_dna_phase3_raw.csv` (generated by `phase3_pipeline.py`)  
**Output:** `data/phase3_continuous_output.csv`, 4 figures

| Cell | What it does |
|---|---|
| 1 | Imports |
| 2 | Load Phase 3 dataset — profile type breakdown |
| 3 | Feature engineering (33 features) — Phase 1 + 5 new |
| 4 | Train/Val/Test split 70/15/15 · StandardScaler fit on train only |
| 5 | GBM: 400 trees · depth 5 · subsample 0.85 |
| 6 | Ensemble: soft vote GBM(w=3) + MLP(w=2) + SVM(w=2) |
| 7 | Continuous DNA score output (0–100 + tier + confidence) |
| 8 | Cross-phase confusion matrix comparison (Phase 1 vs 2 vs 3) |
| 9 | GBM feature importance — new Phase 3 features highlighted |
| 10 | Intermediate class deep-dive — recall improvement |
| 11 | Final model comparison table (all phases) |
| 12 | Summary and Phase 4 direction |

---

## 9. Feature Engineering

### Why feature engineering was necessary

Raw GitHub signals have three problems that prevent direct modeling:

1. **Skewed distributions** — `total_commits` has skewness = 3.2. `log1p()` compresses the tail.
2. **Age bias in counts** — 500 commits in 1 year ≠ 500 commits in 10 years. Rate features remove this.
3. **Fragmented quality signals** — `execution_quality = 0.40×tests + 0.30×readme + 0.30×CI` combines three habits into one signal.

### All 33 features (Phase 3)

```
PROOF-OF-WORK (30%)
  log_total_commits · log_stars_received · log_pull_requests_merged
  log_issues_closed · log_total_repos · commit_velocity · recency_score
  repos_per_year · pr_merge_rate

SKILL GENOME (25%)
  language_entropy · language_count · framework_count
  primary_language_encoded · tech_diversity [Phase 3 NEW]

EXECUTION PATTERN (20%)
  has_readme_pct · has_tests_pct · has_ci_pct · execution_quality
  commit_message_avg_len · consistency_score [Phase 3 NEW]
  quality_volume_ratio [Phase 3 NEW]

THINKING BLUEPRINT (15%)
  fork_to_original_ratio · languages_per_repo_avg · impact_weight
  profile_completeness · avg_repo_description_len
  collab_index [Phase 3 NEW]

GROWTH SIGNATURE (10%)
  commit_trend_slope · activity_decay_lambda · years_active
  growth_momentum · consistency_index
  recent_activity_ratio [Phase 3 NEW]
```

### Phase 3 New Features — Why Each Was Added

| Feature | Formula | Problem it solves |
|---|---|---|
| `consistency_score` | Beta(1+4s, 1+2(1−s)) | Captures regularity — not just volume |
| `tech_diversity` | lang_entropy × fw_count / (fw_count+3) | Breadth × depth interaction |
| `quality_volume_ratio` | qual_signal / log1p(commits) | Fixes "high commits + low quality" failure |
| `collab_index` | log(PRs+issues) / log(repos) | Collaboration intensity signal |
| `recent_activity_ratio` | commits_90d / expected_per_quarter | Growth trajectory vs career baseline |

---

## 10. Model Results — All Phases

### Train / Test Split

```
Phase 1/2 : 80% train / 20% test — stratified
Phase 3   : 70% train / 15% val / 15% test — stratified
Scaling   : StandardScaler fit on training data only — no leakage
Metric    : F1 Macro (penalises any ignored class equally)
```

### Complete Model Comparison Table

| Phase | Model | F1 Macro | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| 1 | Logistic Regression | 0.8066 | 0.8047 | 0.8095 | 0.8033 |
| 1 | **SVM RBF ★** | **0.8116** | **0.8095** | **0.8147** | **0.8083** |
| 1 | Random Forest | 0.8007 | 0.7997 | 0.8019 | 0.7967 |
| 2 | **MLP ★** | **0.8164** | — | — | — |
| 3 | GBM | see outputs/ | — | — | — |
| 3 | **Ensemble ★** | **see outputs/** | — | — | — |

See `outputs/cross_phase_comparison.png` for the full visual breakdown and `outputs/phase3_confusion_matrix.png` for side-by-side confusion matrices.

### Why SVM Won Phase 1

1. **Feature space suitability** — after engineering, all 28 features are continuous and scaled. `K(x,z) = exp(−γ||x−z||²)` maps this space to infinite dimensions where class boundaries become linearly separable.
2. **Global margin maximisation** — unaffected by feature correlation, unlike RF's random feature sampling.
3. **Non-linear boundaries confirmed by t-SNE** (Notebook 1, Cell 11) — justified RBF kernel over Logistic Regression.
4. **GridSearchCV** — 16 combinations × 5 folds = 80 fits. Best: C=10, gamma=scale.

### Why RF Underperformed SVM

RF (F1=0.8007) scored 0.0109 below SVM despite 36 combos × 5 folds = 180 fits. The reason is structural: RF samples √28 ≈ 5 features per split. Several engineered features are correlated (`log_total_commits` ↔ `commit_velocity`, `execution_quality` ↔ `has_tests_pct`) — this splits importance signal across splits. SVM uses all 28 features simultaneously via the kernel; correlation does not hurt it.

### Why Phase 3 Uses an Ensemble

Each base model fails on *different* cases:
- **GBM** excels at tabular feature interactions but can overfit on boundary cases
- **MLP** captures non-linear relationships GBM misses
- **SVM** provides strong margin-based separation, proven in Phase 1

Soft voting averages class probabilities before argmax — especially valuable in the ambiguous Intermediate zone where confidence is genuinely low.

### Intermediate Class Analysis

The Intermediate class has the lowest per-class F1 across all phases. This is not a modeling failure — it reflects genuine behavioral overlap. Phase 3 addresses this directly with 400 boundary samples and the `quality_volume_ratio` feature, which separates the two most common failure modes:

- **Under-classification** (Expert → Intermediate): strong quality signals, low volume
- **Over-classification** (Beginner → Intermediate): high volume signals, weak quality

---

## 11. Continuous DNA Score Output (Phase 3)

Phase 3 moves beyond a tier label. Every prediction includes:

| Field | Type | Description |
|---|---|---|
| `dna_score` | float [0, 100] | Continuous skill score |
| `tier` | int {0, 1, 2} | Classification |
| `tier_name` | str | Beginner / Intermediate / Expert |
| `confidence` | float [0, 1] | Max class probability |
| `prob_beginner` | float | Per-class probability |
| `prob_intermediate` | float | Per-class probability |
| `prob_expert` | float | Per-class probability |

**Score formula:**

```
raw_score = 0.30 × PoW_percentile
          + 0.25 × SkillGenome_percentile
          + 0.20 × ExecutionPattern_percentile
          + 0.15 × ThinkingBlueprint_percentile
          + 0.10 × GrowthSignature_percentile

dna_score = 0.65 × (raw_score × 100) + 0.35 × tier_midpoint
```

Tier midpoints: Beginner = 16.5, Intermediate = 46.5, Expert = 83.0.  
The 35% blend anchors the score within the predicted tier — preventing contradictions such as score = 65 with tier = Beginner.

**Score a new developer:**
```bash
python run_new_dataset.py --input data/new_users_dataset.csv --output data/my_predictions.csv
```

---

## 12. Key Findings

**Finding 1 — Power-law distributions in commit data**  
GitHub commit data follows a Pareto distribution (skewness > 3). The top 20% of developers make ~80% of all commits. Log1p transformation is mandatory before any distance-based modeling.

**Finding 2 — Language entropy is the strongest skill signal**  
Shannon entropy of language distribution has the highest correlation with developer tier (Pearson r = 0.71, p < 0.001). Entropy quantifies linguistic diversification mathematically — a developer writing equal amounts of Python, Go, and Rust is measurably more skilled than one writing only Python.

**Finding 3 — Test coverage is the sharpest quality divider**  
Expert developers have `has_tests_pct` = 0.63 on average vs. Beginners at 0.06 — a 10× gap. No other single feature shows this magnitude of separation.

**Finding 4 — Activity decay lambda predicts growth trajectory**  
68% of Expert developers have negative λ (still actively growing), compared to 14% of Beginners. The exponential decay model `A(t) = A₀ × e^(−λt)` is a genuine behavioral insight not available in raw commit counts.

**Finding 5 — SVM outperforms RF on engineered tabular data**  
Counter to popular assumption, SVM RBF (F1=0.8116) beat a well-tuned Random Forest (F1=0.8007). The engineered feature space is more suited to kernel-based global margin maximisation than to RF's random local splitting.

**Finding 6 — Intermediate tier is irreducibly ambiguous**  
The Intermediate class has the lowest per-class F1 across all three phases. This is not a modeling failure — it reflects genuine behavioral overlap at both boundaries.

**Finding 7 — Data quality, not model capacity, was the bottleneck**  
Phase 2 MLP gained only +0.005 F1 over Phase 1 SVM despite being a far more powerful model. Phase 3 addresses this directly through dataset redesign rather than further model complexity.

**Finding 8 — Phase 3 new features add measurable signal**  
`quality_volume_ratio` and `consistency_score` rank in the top-10 GBM feature importances, directly addressing the "high commits + low quality" failure mode identified in Phase 1 failure analysis.

---

## 13. How to Run

### Reproduce the full pipeline (all phases)

```bash
git clone https://github.com/yourusername/developer-dna-matrix.git
cd developer-dna-matrix
bash setup.sh
```

### Run Phase 3 only

```bash
source venv/bin/activate
python phase3_pipeline.py
```

### Run via Docker

```bash
docker build -t ddm-phase3 .
docker run --rm -v $(pwd)/outputs:/app/outputs ddm-phase3
```

### Score new developers

```bash
python run_new_dataset.py --input data/new_users_dataset.csv --output data/predictions.csv
```

### Expected outputs after full run

```
outputs/
├── github_dna_phase3_raw.csv            ← 7,200+ row improved dataset
├── phase3_continuous_output.csv         ← Score + Tier + Confidence
├── phase3_results.csv                   ← All-model comparison table
├── phase3_confusion_matrix.png          ← GBM vs MLP vs Ensemble
├── cross_phase_comparison.png           ← F1 across all 3 phases
├── continuous_score_distribution.png    ← Score distribution by tier
├── feature_importance_phase3.png        ← Top-20 GBM features
└── DDM_Phase3_Architecture.pdf          ← Architecture diagram

figures/
├── eda_01_*.png through eda_08_*.png    ← 8 EDA figures
├── fe_01_*.png through fe_06_*.png      ← 6 Feature Engineering figures
└── model_01_*.png through model_03_*.png← 3 Phase 1 figures
```

All random seeds fixed at `np.random.seed(42)`. The pipeline is fully deterministic.

---

## 14. Requirements

```
# Core ML
scikit-learn==1.3.0
pandas==2.1.0
numpy==1.24.3
scipy==1.11.2

# Deep learning (Phase 2)
tensorflow==2.13.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Notebook environment
jupyter==1.0.0
ipykernel==6.25.1
nbformat==5.9.2

# Data generation
faker==19.3.1

# Reporting
reportlab==4.0.4
```

```bash
pip install -r requirements.txt
```

---

## 15. References

1. Cortes, C. & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273–297.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
3. Friedman, J.H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics, 29(5), 1189–1232.
4. Cover, T. & Thomas, J. (2006). *Elements of Information Theory*. Wiley.
5. Gousios, G. (2013). *The GHTorrent dataset and tool suite*. MSR '13.
6. Kalliamvakou, E. et al. (2014). *The promises and perils of mining GitHub*. MSR '14.
7. Allamanis, M. et al. (2018). *A Survey of Machine Learning for Big Code and Naturalness*. ACM CSUR.
8. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
9. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.

---

<div align="center">

**Developer DNA Matrix**  
Advanced Machine Learning & Deep Learning — Phase 3  
Behavioral Intelligence for Developer Evaluation

</div>
