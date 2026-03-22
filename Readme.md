# Developer DNA Matrix (DDM)
> **An AI-driven behavioral intelligence framework that evaluates software developers through deep analysis of their GitHub proof-of-work artifacts — not resumes.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.0-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)
[![Course](https://img.shields.io/badge/Course-Advanced%20ML%20%26%20DL-purple?style=flat)]()
[![Phase](https://img.shields.io/badge/Phase-1%20Complete-brightgreen?style=flat)]()

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Overview](#2-project-overview)
3. [The Developer DNA Score (DDS)](#3-the-developer-dna-score-dds)
4. [System Architecture](#4-system-architecture)
5. [Dataset](#5-dataset)
6. [Repository Structure](#6-repository-structure)
7. [Notebook Guide](#7-notebook-guide)
8. [Feature Engineering](#8-feature-engineering)
9. [Model Application & Results](#9-model-application--results)
10. [Key Findings](#10-key-findings)
11. [How to Run](#11-how-to-run)
12. [Requirements](#12-requirements)
13. [Phase Roadmap](#13-phase-roadmap)
14. [References](#14-references)

---

## 1. Problem Statement

Traditional hiring evaluates developers through resumes and interviews — both of which measure **communication ability**, not **execution capability**. A developer can write an impressive resume without having built anything meaningful. Conversely, the most capable engineers often have sparse resumes but rich GitHub histories.

**The gap:** There is no standardised, objective, behavior-centric system that quantifies developer competence from real-world proof-of-work artifacts.

**DDM solves this** by treating a developer's GitHub history as a behavioral fingerprint. Every commit, pull request, language choice, test file, CI configuration, and collaboration pattern reveals something quantifiable about who that developer is and how they work.

---

## 2. Project Overview

The Developer DNA Matrix (DDM) is a multi-phase ML and Deep Learning project that:

- Extracts **23 behavioral signals** from GitHub developer profiles
- Engineers **28 feature dimensions** across 5 DNA dimensions
- Trains **3 classification models** (Logistic Regression, SVM, Random Forest)
- Produces a **Developer DNA Score (DDS)** on a 0–1 scale
- Classifies developers into **3 tiers**: Beginner (0), Intermediate (1), Expert (2)

### What makes DDM different from GitHub statistics dashboards

| GitHub Statistics | Developer DNA Matrix |
|---|---|
| Shows raw numbers (commits, stars) | Derives behavioral rates (velocity, decay) |
| No tier classification | 3-tier classification with F1=0.8116 |
| No quality signals | Tests, CI/CD, commit quality scored |
| Static snapshot | Temporal growth trajectory modeled |
| No comparative ranking | Percentile-based tier assignment |

---

## 3. The Developer DNA Score (DDS)

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
| `pull_requests_merged` | Code quality gate — PRs that were reviewed and accepted |
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

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW BEHAVIORAL DATA                          │
│         GitHub Profiles — 6,000 Developer Behavioral           │
│         Signals collected via REST API / generated              │
│         with realistic distributions and noise                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                NOTEBOOK 1 — EDA                                 │
│  • Statistical characterisation (skewness, power-law, entropy)  │
│  • Outlier detection (IQR 3× method)                            │
│  • Correlation analysis (Pearson heatmap)                       │
│  • Class balance check (Gini impurity)                          │
│  • t-SNE + PCA dimensionality reduction                         │
│  • Activity decay curve fitting                                 │
│  OUTPUT: github_dna_processed.csv                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               NOTEBOOK 2 — FEATURE ENGINEERING                  │
│  Group 1: Log transforms   → log1p(commits, stars, PRs, issues) │
│  Group 2: Rate features    → velocity, recency, repos_per_year  │
│  Group 3: Quality composites → execution_quality, impact_weight │
│  Group 4: Growth features  → growth_momentum, consistency_index │
│  Encoding: primary_language → LabelEncoder integer              │
│  OUTPUT: X_features.csv, y_labels.csv, selected_features.json  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               NOTEBOOK 3 — MODEL APPLICATION                    │
│                                                                 │
│  Model 1: Logistic Regression (linear baseline)                 │
│           F1 = 0.8066 — establishes linear ceiling              │
│                    │                                            │
│                    ▼                                            │
│  Model 2: SVM with RBF Kernel (tuned via GridSearchCV)          │
│           F1 = 0.8116 ★ WINNER                                  │
│           K(x,z) = exp(−γ||x−z||²)                             │
│                    │                                            │
│                    ▼                                            │
│  Model 3: Random Forest (300 trees, GridSearchCV tuned)         │
│           F1 = 0.8007                                           │
│                    │                                            │
│                    ▼                                            │
│  Failure Analysis → 5 misclassified cases explained             │
│  OUTPUT: model_comparison_results.csv, figures                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              DEVELOPER DNA SCORE + TIER LABEL                   │
│   Beginner (0)  │  Intermediate (1)  │  Expert (2)             │
│   DDS: 0.0–0.3  │    DDS: 0.3–0.5   │   DDS: 0.5–1.0          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Dataset

### Overview

| Property | Value |
|---|---|
| Total rows | 6,000 developer profiles |
| Total columns | 29 (raw) → 28 features after engineering |
| Null values | 0 |
| Duplicate developer IDs | 0 |
| Class balance | 2,000 per tier (balanced) |
| Source | GitHub behavioral signals (realistic synthetic with overlap) |

### Why This Dataset

A purely scraped GitHub dataset was evaluated but rejected — it lacked commit-level signals (total_commits, language entropy, has_tests_pct) which are the core DNA signals. The final dataset was constructed with:

- **Trimodal latent skill distribution** — a hidden skill score (0–1) drives all features, creating natural class overlap
- **8% label noise** — deliberately mislabels borderline cases to simulate real-world ambiguity
- **Log-normal feature distributions** — commits and stars follow Pareto distributions, not uniform
- **25% boundary cases** — developers whose features blend toward adjacent tiers
- **Luck factor on stars** — stars are partly random (a script can go viral), modeled explicitly

### Class Distribution

```
Beginner     (Tier 0) : 2,000 developers  (33.3%)
Intermediate (Tier 1) : 2,000 developers  (33.3%)
Expert       (Tier 2) : 2,000 developers  (33.3%)

Gini Impurity: 0.667 — perfectly balanced (no SMOTE needed)
```

### Key Statistical Findings from EDA

| Feature | Skewness | Finding | Action |
|---|---|---|---|
| `total_commits` | 3.2 | Power-law (Pareto) — top 20% make 80% of commits | log1p() transform |
| `stars_received` | 4.1 | Extreme right tail — viral projects distort raw signal | log1p() transform |
| `language_entropy` | 0.3 | Near-normal — Shannon entropy is naturally bounded | No transform needed |
| `has_tests_pct` | 0.8 | Bimodal — devs either test everything or nothing | Kept raw |
| `activity_decay_lambda` | 0.2 | Near-normal — lambda centered at 0 | Kept raw |

---

## 6. Repository Structure

```
developer-dna-matrix/
│
├── README.md                          ← You are here
├── requirements.txt                   ← All dependencies pinned
├── .gitignore
├── LICENSE
│
├── data/
│   ├── github_dna_6k_realistic.csv    ← Raw dataset (6000 rows × 29 cols)
│   ├── github_dna_processed.csv       ← After EDA cleaning (Notebook 1 output)
│   ├── github_dna_features.csv        ← After feature engineering (Notebook 2 output)
│   ├── X_features.csv                 ← Feature matrix (model input)
│   ├── y_labels.csv                   ← Target labels (developer_tier)
│   ├── selected_features.json         ← Final 28 feature names list
│   └── model_comparison_results.csv   ← Final model results table
│
├── notebooks/
│   ├── 01_eda.ipynb                   ← Exploratory Data Analysis (12 cells)
│   ├── 02_feature_engineering.ipynb   ← Feature Engineering (12 cells)
│   └── 03_model_application.ipynb     ← Model Training & Evaluation (12 cells)
│
├── src/
│   ├── features.py                    ← All feature engineering functions
│   ├── model.py                       ← Model training & evaluation functions
│   └── utils.py                       ← Utility functions (entropy, plots)
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
    ├── DDM_Phase1_Report.pdf          ← Final LaTeX report (compiled PDF)
    └── DDM_Phase1_Report.tex          ← LaTeX source
```

---

## 7. Notebook Guide

### Notebook 1 — EDA (`01_eda.ipynb`)

**Input:** `github_dna_6k_realistic.csv`
**Output:** `github_dna_processed.csv`

| Cell | What it does | Rubric criterion addressed |
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

### Notebook 2 — Feature Engineering (`02_feature_engineering.ipynb`)

**Input:** `github_dna_processed.csv`
**Output:** `X_features.csv`, `y_labels.csv`, `selected_features.json`

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
| 11 | Quick feature importance preview | Validates engineered features add signal |
| 12 | Save all outputs | — |

**Features dropped and why:**

| Column dropped | Reason |
|---|---|
| `developer_dna_score` | **DATA LEAKAGE** — this is the answer |
| `developer_tier` | **TARGET VARIABLE** — goes into y not X |
| `developer_id` | Text identifier — not numeric |
| `languages_used` | Raw text string — RF cannot parse |
| `primary_language` | Replaced by `primary_language_encoded` |
| `total_commits` | Replaced by `log_total_commits` (de-skewed) |
| `commits_last_90d` | Replaced by `recency_score` (relative metric) |
| `stars_received` | Replaced by `log_stars_received` |
| `pull_requests_merged` | Replaced by `log_pull_requests_merged` |
| `issues_closed` | Replaced by `log_issues_closed` |
| `account_age_days` | Replaced by `years_active` (cleaner unit) |

---

### Notebook 3 — Model Application (`03_model_application.ipynb`)

**Input:** `X_features.csv`, `y_labels.csv`, `selected_features.json`
**Output:** `model_comparison_results.csv`, 3 figures

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

## 8. Feature Engineering

### Why feature engineering was necessary

Raw GitHub signals have three problems that prevent direct modeling:

1. **Skewed distributions** — `total_commits` has skewness = 3.2. One developer with 25,000 commits would dominate gradient calculations. `log1p()` compresses the tail.

2. **Age bias in counts** — 500 commits in 1 year is very different from 500 commits in 10 years. Rate features (`commit_velocity = commits / active_weeks`) remove this bias.

3. **Fragmented quality signals** — `has_readme_pct`, `has_tests_pct`, `has_ci_pct` each measure one habit. `execution_quality = 0.40×tests + 0.30×readme + 0.30×CI` captures the overall professional maturity pattern.

### All 28 final features

```
PROOF-OF-WORK (30% DDS weight)
  log_total_commits        — de-skewed commit volume
  log_stars_received       — de-skewed impact signal
  log_pull_requests_merged — de-skewed PR signal
  log_issues_closed        — de-skewed issue signal
  log_total_repos          — de-skewed repo count
  commit_velocity          — commits per active week
  recency_score            — recent vs historical activity ratio
  repos_per_year           — delivery cadence
  pr_merge_rate            — code acceptance rate

SKILL GENOME (25% DDS weight)
  language_entropy         — Shannon H = −Σpᵢlog₂(pᵢ)
  language_count           — number of distinct languages
  framework_count          — framework breadth
  primary_language_encoded — dominant specialisation (label encoded)

EXECUTION PATTERN (20% DDS weight)
  has_readme_pct           — documentation rate (raw)
  has_tests_pct            — test coverage rate (raw)
  has_ci_pct               — CI/CD adoption rate (raw)
  execution_quality        — 0.40×tests + 0.30×readme + 0.30×CI
  commit_message_avg_len   — communication quality signal

THINKING BLUEPRINT (15% DDS weight)
  fork_to_original_ratio   — creator vs consumer ratio
  languages_per_repo_avg   — systems thinking breadth
  impact_weight            — log(stars+1) × log(repos+1)
  profile_completeness     — bio + company + location + blog
  avg_repo_description_len — architectural communication

GROWTH SIGNATURE (10% DDS weight)
  commit_trend_slope       — linear regression slope on monthly commits
  activity_decay_lambda    — λ = −ln(A_recent/A_old)/6
  years_active             — career length normaliser
  growth_momentum          — trend_slope / (years_active + 1)
  consistency_index        — 1 / (1 + |lambda|)
```

---

## 9. Model Application & Results

### Train / Test Split

```
Total samples : 6,000
Train set     : 4,800 (80%) — stratified
Test set      : 1,200 (20%) — stratified
Scaling       : StandardScaler (fit on train only — no leakage)
CV Strategy   : StratifiedKFold, 5 folds
Metric        : F1 Macro (correct for multi-class overlapping tiers)
```

### Why F1 Macro and not Accuracy

Accuracy treats all errors equally regardless of class. With overlapping tiers, a naive model that always predicts "Intermediate" gets ~33% accuracy for free. F1 Macro computes the harmonic mean of Precision and Recall **per class** then averages — it penalises models that ignore any one class.

### Final Model Comparison

| Rank | Model | F1 Macro | Precision | Recall | Accuracy | CV F1 (5-fold) |
|---|---|---|---|---|---|---|
| **1 ★** | **SVM RBF (tuned)** | **0.8116** | **0.8095** | **0.8147** | **0.8083** | **0.8121** |
| 2 | Logistic Regression | 0.8066 | 0.8047 | 0.8095 | 0.8033 | 0.8118 |
| 3 | Random Forest (tuned) | 0.8007 | 0.7997 | 0.8019 | 0.7967 | 0.8078 |
| 4 | RF Baseline (n=30, d=4) | 0.7851 | — | — | 0.7808 | — |

### Why SVM Won

SVM with RBF kernel outperformed all models because:

1. **Feature space suitability** — after engineering, all 28 features are continuous, scaled, and uniformly distributed. This is precisely the space where `K(x,z) = exp(−γ||x−z||²)` excels — it maps the feature space to infinite dimensions where class boundaries become linearly separable.

2. **Global margin maximisation** — SVM finds the decision boundary that maximises the margin to the nearest support vectors. This global optimisation is not affected by feature correlation the way RF's random feature sampling is.

3. **Non-linear boundaries confirmed by EDA** — t-SNE (Notebook 1, Cell 11) showed non-linear tier separation. The RBF kernel captures this directly. Logistic Regression assumes linear boundaries — its strong F1 of 0.8066 confirms significant linear structure, while SVM's higher score shows the residual non-linear gain.

4. **Optimal hyperparameters found by GridSearchCV** — 16 combinations tested (`C` ∈ {0.1, 1, 10, 100}, `gamma` ∈ {scale, auto, 0.01, 0.001}) × 5 folds = 80 fits.

### Why Random Forest Underperformed SVM

Despite extensive GridSearchCV tuning (36 combos × 5 folds = 180 fits), RF scored 0.0109 below SVM. The reason is structural:

- RF randomly samples feature subsets at each split (√28 ≈ 5 features per split)
- Several engineered features are correlated — e.g. `log_total_commits` ↔ `commit_velocity`, `execution_quality` ↔ `has_tests_pct`
- Correlated features split importance signal across splits, reducing each individual feature's discriminative power
- SVM uses ALL 28 features simultaneously via the kernel — correlation does not hurt it

### Failure Analysis Summary

Error rate on test set: **19%** (SVM best model)

**Most common error:** Intermediate tier misclassified (into Beginner or Expert)
- This is expected — Intermediate is genuinely bounded by both adjacent tiers in behavioral space

**Root cause of failures:**
The DDS formula's 30% Proof-of-Work weight creates systematic errors whenever a developer's volume signals and quality signals disagree:

- **Under-classification** (Expert predicted as Intermediate): Strong quality signals (high `language_entropy`, `execution_quality`, `has_tests_pct`) but low volume signals (low `log_total_commits`, `commit_velocity`). Kernel distance pulled toward lower tier.

- **Over-classification** (Beginner predicted as Intermediate): High volume signals (high `log_total_commits`, `impact_weight`) but weak quality signals. Quantity-without-quality is a blind spot in the current DDS labeling formula.

This failure analysis directly motivates Phase 2: a DL component can learn a flexible non-linear mapping between volume and quality that the fixed DDS formula cannot express.

---

## 10. Key Findings

### Finding 1 — Power-law distributions in commit data
GitHub commit data follows Pareto distribution (skewness > 3). The top 20% of developers make approximately 80% of all commits. Log1p transformation is mandatory before any distance-based modeling.

### Finding 2 — Language entropy is the strongest skill signal
Shannon entropy of language distribution has the highest correlation with developer tier (Pearson r = 0.71, p < 0.001). A developer who writes equal amounts of Python, Go, and Rust is provably more skilled than one who writes only Python — entropy quantifies this diversification mathematically.

### Finding 3 — Test coverage is the sharpest quality divider
Expert developers have `has_tests_pct` = 0.63 on average vs. Beginners at 0.06 — a 10× gap. No other single feature shows this magnitude of separation, making test coverage the single strongest Execution Pattern signal.

### Finding 4 — Activity decay lambda predicts growth trajectory
68% of Expert developers have negative λ (still actively growing), compared to 14% of Beginners. The exponential decay model `A(t) = A₀ × e^(−λt)` fitted to commit history is a genuine behavioral insight not available in raw commit counts.

### Finding 5 — SVM outperforms RF on engineered tabular data
Counter to popular assumption, SVM with RBF kernel (F1=0.8116) beat a well-tuned Random Forest (F1=0.8007) on this dataset. The engineered feature space — continuous, scaled, moderately correlated — is more suited to kernel-based global margin maximisation than to RF's random local splitting.

### Finding 6 — Intermediate tier is irreducibly ambiguous
The Intermediate class has the lowest per-class F1 across all models. This is not a modeling failure — it reflects genuine behavioral overlap. Some Beginner-level developers have one high-quality project (high execution_quality) that overlaps with Intermediate, while some Expert developers have low recent activity (high decay_lambda) that overlaps with Intermediate.

---

## 11. How to Run

### Option A — Google Colab (Recommended)

1. Clone this repository
2. Open `notebooks/01_eda.ipynb` in Google Colab
3. Upload `data/github_dna_6k_realistic.csv` when prompted
4. Run all cells in order
5. Download `github_dna_processed.csv` at the end
6. Open `notebooks/02_feature_engineering.ipynb`
7. Upload `github_dna_processed.csv` when prompted
8. Run all cells in order
9. Download `X_features.csv`, `y_labels.csv`, `selected_features.json`
10. Open `notebooks/03_model_application.ipynb`
11. Upload the three files from step 9
12. Run all cells in order

### Option B — Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/developer-dna-matrix.git
cd developer-dna-matrix

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/01_eda.ipynb
```

### Expected Outputs

After running all three notebooks you will have:

```
data/github_dna_processed.csv        ← 5,980 rows (after outlier removal)
data/github_dna_features.csv         ← 5,980 rows × 41 columns
data/X_features.csv                  ← 5,980 rows × 28 features
data/y_labels.csv                    ← 5,980 rows
data/model_comparison_results.csv    ← 4-row results table

figures/eda_01_*.png through eda_08_*.png     ← 8 EDA figures
figures/fe_01_*.png through fe_06_*.png       ← 6 Feature Engineering figures
figures/model_01_*.png through model_03_*.png ← 3 Model figures
```

---

## 12. Requirements

```
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.2
faker==19.3.1
jupyter==1.0.0
ipykernel==6.25.1
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 13. Phase Roadmap

### Phase 1 — Core ML Project (Current)

| Deliverable | Status |
|---|---|
| Dataset (6,000 developers) | ✅ Complete |
| EDA Notebook (12 cells, 8 figures) | ✅ Complete |
| Feature Engineering Notebook (12 cells, 6 figures) | ✅ Complete |
| Model Application — LR + SVM + RF | ✅ Complete |
| Model Comparison Table | ✅ Complete |
| Failure Analysis | ✅ Complete |
| LaTeX Report | 🔄 In Progress |
| Presentation Video | 🔄 In Progress |


---

## 14. References

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
3. Cortes, C. & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273–297.
4. Gousios, G. (2013). *The GHTorrent dataset and tool suite*. MSR '13.
5. Kalliamvakou, E. et al. (2014). *The promises and perils of mining GitHub*. MSR '14.
6. Allamanis, M. et al. (2018). *A Survey of Machine Learning for Big Code and Naturalness*. ACM CSUR.
7. Cover, T. & Thomas, J. (2006). *Elements of Information Theory*. Wiley.
8. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825–2830, 2011.

---

<div align="center">

**Developer DNA Matrix**
Advanced Machine Learning & Deep Learning — Phase 1

</div>
