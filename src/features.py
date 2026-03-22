"""
features.py — Developer DNA Matrix
====================================
All feature engineering functions.
One function per feature group, plus a master pipeline function.

Each function is documented with:
  - WHY this feature was created (domain justification)
  - FORMULA used
  - DNA dimension it belongs to
  - Expected value range
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ── DNA dimension weights (from DDS formula) ──────────────────────────────
DDS_WEIGHTS = {
    'proof_of_work'     : 0.30,
    'skill_genome'      : 0.25,
    'execution_pattern' : 0.20,
    'thinking_blueprint': 0.15,
    'growth_signature'  : 0.10,
}

# ── Columns to drop before modeling ───────────────────────────────────────
COLUMNS_TO_DROP = [
    'developer_id',           # text identifier — not numeric
    'languages_used',         # raw text string — cannot be used by RF/SVM
    'primary_language',       # text — replaced by primary_language_encoded
    'developer_dna_score',    # DATA LEAKAGE — this IS the answer
    'developer_tier',         # TARGET VARIABLE — goes into y not X
    'account_age_days',       # replaced by years_active (better unit)
    'total_commits',          # replaced by log_total_commits
    'commits_last_90d',       # replaced by recency_score
    'stars_received',         # replaced by log_stars_received
    'pull_requests_merged',   # replaced by log_pull_requests_merged
    'issues_closed',          # replaced by log_issues_closed
]

# ── Original (non-engineered) columns kept for reference ──────────────────
ORIGINAL_KEPT_COLS = [
    'total_repos', 'language_entropy', 'framework_count',
    'has_readme_pct', 'has_tests_pct', 'has_ci_pct',
    'commit_message_avg_len', 'fork_to_original_ratio',
    'languages_per_repo_avg', 'commit_trend_slope',
    'activity_decay_lambda', 'years_active',
    'avg_repo_description_len', 'primary_language_encoded',
]


# ══════════════════════════════════════════════════════════════════════════
# GROUP 1 — LOG TRANSFORMS (Fix Power-Law Skew)
# ══════════════════════════════════════════════════════════════════════════

def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log1p() transformation to power-law skewed columns.

    WHY:
        EDA Cell 5 showed total_commits has skewness = 3.2 (power-law).
        A developer with 25,000 commits would dominate distance calculations
        in SVM and gradient updates in any model.
        log1p(x) = log(x + 1) handles zeros safely.
        After transform, skewness drops below 1.0 for all columns.

    DNA Dimension: Proof-of-Work (30% DDS weight)

    Columns created:
        log_total_commits
        log_stars_received
        log_pull_requests_merged
        log_issues_closed
        log_total_repos

    Formula: log_x = log(x + 1)
    Range:   [0, ~10] after transform
    """
    skewed_cols = [
        'total_commits',
        'stars_received',
        'pull_requests_merged',
        'issues_closed',
        'total_repos',
    ]
    for col in skewed_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    return df


# ══════════════════════════════════════════════════════════════════════════
# GROUP 2 — RATE FEATURES (Remove Age Bias)
# ══════════════════════════════════════════════════════════════════════════

def add_commit_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Commit Velocity = total_commits / (years_active × 52 weeks)

    WHY:
        Raw commit count is biased by account age.
        A developer with 500 commits in 1 year is demonstrably more
        active than one with 500 commits in 10 years.
        Velocity removes the age bias and measures execution RATE.

    DNA Dimension: Proof-of-Work (30%)
    Formula: commit_velocity = total_commits / (years_active * 52 + 1)
    Range:   [0, ~200] commits per week
    """
    df['commit_velocity'] = (
        df['total_commits'] /
        (df['years_active'] * 52 + 1)
    ).round(4)
    return df


def add_recency_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recency Score = commits_last_90d / total_commits

    WHY:
        Measures whether a developer is still actively coding.
        High score (near 1.0): most commits are recent → actively growing.
        Low score (near 0.0): mostly old commits → may have dropped off.
        This is distinct from activity_decay_lambda which fits an
        exponential model — recency_score is a simpler ratio version.

    DNA Dimension: Proof-of-Work (30%)
    Formula: recency_score = commits_last_90d / (total_commits + 1)
    Range:   [0, 1]
    """
    df['recency_score'] = (
        df['commits_last_90d'] /
        (df['total_commits'] + 1)
    ).round(4)
    return df


def add_repos_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repos Per Year = total_repos / years_active

    WHY:
        Measures delivery cadence — how many projects does this
        developer ship per year? Normalises repo count by career length.

    DNA Dimension: Proof-of-Work (30%)
    Formula: repos_per_year = total_repos / (years_active + 1)
    Range:   [0, ~100]
    """
    df['repos_per_year'] = (
        df['total_repos'] /
        (df['years_active'] + 1)
    ).round(4)
    return df


def add_pr_merge_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    PR Merge Rate = pull_requests_merged / (merged + issues_closed + 1)

    WHY:
        Merged PRs indicate code that was reviewed and accepted by others.
        Raw PR count inflates with quantity; merge RATE captures quality.
        A developer whose PRs consistently get merged is demonstrably
        writing reviewable, acceptable code.

    DNA Dimension: Proof-of-Work (30%)
    Formula: pr_merge_rate = merged / (merged + issues + 1)
    Range:   [0, 1]
    """
    df['pr_merge_rate'] = (
        df['pull_requests_merged'] /
        (df['pull_requests_merged'] + df['issues_closed'] + 1)
    ).round(4)
    return df


# ══════════════════════════════════════════════════════════════════════════
# GROUP 3 — QUALITY COMPOSITE FEATURES
# ══════════════════════════════════════════════════════════════════════════

def add_execution_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execution Quality = 0.40×tests + 0.30×readme + 0.30×CI

    WHY:
        has_readme_pct, has_tests_pct, has_ci_pct each measure one
        professional habit in isolation. The composite score measures
        the OVERALL professional maturity pattern.
        Weights reflect importance to code maturity:
          - Tests (0.40): Engineering discipline — hardest to fake
          - README (0.30): Documentation discipline
          - CI/CD (0.30): DevOps maturity

    DNA Dimension: Execution Pattern (20%)
    Formula: execution_quality = 0.40*tests + 0.30*readme + 0.30*ci
    Range:   [0, 1]
    """
    df['execution_quality'] = (
        0.40 * df['has_tests_pct'] +
        0.30 * df['has_readme_pct'] +
        0.30 * df['has_ci_pct']
    ).round(4)
    return df


def add_language_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Language Count = number of distinct languages in languages_used

    WHY:
        language_entropy measures the DISTRIBUTION of language usage.
        language_count measures the RAW BREADTH.
        Together they describe both how many languages and
        how evenly distributed usage is across them.

    DNA Dimension: Skill Genome (25%)
    Formula: language_count = len(set(languages_used.split(',')))
    Range:   [1, 20]
    """
    if 'language_count' not in df.columns:
        df['language_count'] = df['languages_used'].apply(
            lambda x: len(set(str(x).split(',')))
        )
    return df


def add_impact_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impact Weight = log(stars + 1) × log(total_repos + 1)

    WHY:
        This is an INTERACTION TERM — it captures the combination of
        reach (stars) and output volume (repos).
        A developer with many stars on many repos has broader real-world
        impact than one with stars on a single viral project.
        The log-log product compresses both extremes.

    DNA Dimension: Thinking Blueprint (15%)
    Formula: impact_weight = log1p(stars) × log1p(total_repos)
    Range:   [0, ~60]
    """
    df['impact_weight'] = (
        np.log1p(df['stars_received']) *
        np.log1p(df['total_repos'])
    ).round(4)
    return df


def add_profile_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Profile Completeness = sum of binary profile fields

    WHY:
        Professional self-presentation is a proxy for architectural
        communication ability. A developer who fills in bio, company,
        location, and blog is more likely to document their code and
        think about audience.
        Range: 0 (empty profile) to 4 (fully complete).

    DNA Dimension: Thinking Blueprint (15%)
    Formula: profile_completeness = has_bio + has_company + has_location + has_blog
    Range:   [0, 4]
    """
    bio_cols   = ['has_bio', 'has_company', 'has_location', 'has_blog']
    available  = [c for c in bio_cols if c in df.columns]
    if available and 'profile_completeness' not in df.columns:
        df['profile_completeness'] = df[available].sum(axis=1)
    return df


# ══════════════════════════════════════════════════════════════════════════
# GROUP 4 — GROWTH & CONSISTENCY FEATURES
# ══════════════════════════════════════════════════════════════════════════

def add_growth_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Growth Momentum = commit_trend_slope / (years_active + 1)

    WHY:
        Raw commit_trend_slope is the slope of a linear regression on
        monthly commit counts over 12 months.
        A slope of +3.0 means very different things for a 1-year developer
        (strong growth) vs a 10-year developer (modest uptick).
        Normalising by career length makes the growth signal comparable
        across developers of different seniority.

    DNA Dimension: Growth Signature (10%)
    Formula: growth_momentum = commit_trend_slope / (years_active + 1)
    Range:   [-5, +10] approximately
    """
    df['growth_momentum'] = (
        df['commit_trend_slope'] /
        (df['years_active'] + 1)
    ).round(4)
    return df


def add_consistency_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consistency Index = 1 / (1 + |activity_decay_lambda|)

    WHY:
        activity_decay_lambda is the decay rate from the exponential model
        A(t) = A₀ × e^(−λt) fitted to commit history.
        λ = 0  → stable activity → consistency_index = 1.0
        λ > 0  → declining → consistency_index < 1.0
        λ < 0  → growing  → consistency_index between 0.5 and 1.0

        This maps the unbounded lambda to a bounded [0, 1] scale
        where 1.0 = maximally consistent and 0.0 = maximally erratic.

    DNA Dimension: Growth Signature (10%)
    Formula: consistency_index = 1 / (1 + |lambda|)
    Range:   (0, 1]
    """
    df['consistency_index'] = (
        1 / (1 + np.abs(df['activity_decay_lambda']))
    ).round(4)
    return df


# ══════════════════════════════════════════════════════════════════════════
# GROUP 5 — ENCODING
# ══════════════════════════════════════════════════════════════════════════

def encode_primary_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes primary_language (string) as integer using LabelEncoder.

    WHY:
        Random Forest and SVM cannot accept string inputs directly.
        LabelEncoder maps each unique language to a stable integer.
        Example: 'Python' → 15, 'Go' → 7, 'Rust' → 18

    DNA Dimension: Skill Genome (25%)
    Returns: df with 'primary_language_encoded' column added
    """
    if 'primary_language_encoded' not in df.columns:
        le = LabelEncoder()
        df['primary_language_encoded'] = le.fit_transform(
            df['primary_language'].astype(str)
        )
        n_langs = df['primary_language'].nunique()
        print(f"  primary_language_encoded: {n_langs} unique languages → integers")
    return df


# ══════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def engineer_all_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Master pipeline — applies all feature engineering steps in order.

    Call this once on the processed dataset to get the fully engineered
    dataframe ready for feature selection and model training.

    Steps:
        1. Log transforms      (5 new columns)
        2. Rate features       (4 new columns)
        3. Quality composites  (4 new columns)
        4. Growth features     (2 new columns)
        5. Encoding            (1 new column)

    Parameters
    ----------
    df      : pd.DataFrame — output of EDA preprocessing notebook
    verbose : bool         — print progress

    Returns
    -------
    pd.DataFrame with all engineered features added
    """
    original_cols = len(df.columns)

    # Group 1
    df = add_log_transforms(df)
    if verbose: print("  ✓ Group 1: Log transforms applied")

    # Group 2
    df = add_commit_velocity(df)
    df = add_recency_score(df)
    df = add_repos_per_year(df)
    df = add_pr_merge_rate(df)
    if verbose: print("  ✓ Group 2: Rate features created")

    # Group 3
    df = add_execution_quality(df)
    df = add_language_count(df)
    df = add_impact_weight(df)
    df = add_profile_completeness(df)
    if verbose: print("  ✓ Group 3: Quality composite features created")

    # Group 4
    df = add_growth_momentum(df)
    df = add_consistency_index(df)
    if verbose: print("  ✓ Group 4: Growth & consistency features created")

    # Group 5
    df = encode_primary_language(df)
    if verbose: print("  ✓ Group 5: Categorical encoding done")

    new_cols = len(df.columns) - original_cols
    if verbose:
        print(f"\n  Original columns  : {original_cols}")
        print(f"  New features added: {new_cols}")
        print(f"  Total columns now : {len(df.columns)}")

    return df


def get_final_feature_list() -> list:
    """
    Returns the ordered list of 28 features used as model input.

    These are the columns that go into X for model training.
    Excludes: developer_id, text columns, target, leakage columns.

    Returns
    -------
    list of str
    """
    return [
        # Proof-of-Work (30% DDS weight)
        'log_total_commits',
        'log_stars_received',
        'log_pull_requests_merged',
        'log_issues_closed',
        'log_total_repos',
        'commit_velocity',
        'recency_score',
        'repos_per_year',
        'pr_merge_rate',

        # Skill Genome (25% DDS weight)
        'language_entropy',
        'language_count',
        'framework_count',
        'primary_language_encoded',

        # Execution Pattern (20% DDS weight)
        'has_readme_pct',
        'has_tests_pct',
        'has_ci_pct',
        'execution_quality',
        'commit_message_avg_len',

        # Thinking Blueprint (15% DDS weight)
        'fork_to_original_ratio',
        'languages_per_repo_avg',
        'impact_weight',
        'profile_completeness',
        'avg_repo_description_len',

        # Growth Signature (10% DDS weight)
        'commit_trend_slope',
        'activity_decay_lambda',
        'years_active',
        'growth_momentum',
        'consistency_index',
    ]


def build_X_y(
    df: pd.DataFrame,
    feature_list: list = None
):
    """
    Builds the final feature matrix X and target vector y.

    Parameters
    ----------
    df           : pd.DataFrame — fully engineered dataframe
    feature_list : list or None — if None uses get_final_feature_list()

    Returns
    -------
    X : pd.DataFrame — feature matrix
    y : pd.Series    — target labels (developer_tier)
    available : list — features actually present in df
    """
    if feature_list is None:
        feature_list = get_final_feature_list()

    available = [f for f in feature_list if f in df.columns]
    missing   = [f for f in feature_list if f not in df.columns]

    if missing:
        print(f"  ⚠ Missing features (not in df): {missing}")
        print(f"    Run engineer_all_features() first")

    X = df[available].fillna(0)
    y = df['developer_tier']

    print(f"✓ X shape : {X.shape}")
    print(f"✓ y shape : {y.shape}")
    print(f"✓ {len(available)} / {len(feature_list)} features available")

    return X, y, available
