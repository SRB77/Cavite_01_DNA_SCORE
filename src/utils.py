"""
utils.py — Developer DNA Matrix
================================
Utility functions used across all three notebooks.
Covers: Shannon entropy, data loading, validation,
        plotting helpers, DDS score computation.
"""

import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ── Color palette (consistent across all notebooks) ───────────────────────
TIER_COLORS = {0: '#2E5FA3', 1: '#E8A020', 2: '#1A7A6B'}
TIER_NAMES  = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
TIER_LIST   = ['Beginner', 'Intermediate', 'Expert']


# ══════════════════════════════════════════════════════════════════════════
# 1. INFORMATION THEORY
# ══════════════════════════════════════════════════════════════════════════

def shannon_entropy(language_list: list) -> float:
    """
    Computes Shannon entropy H = -sum(p_i * log2(p_i))
    for a list of programming languages.

    Mathematical foundation:
        H is maximised when all languages are used equally
        (maximum diversity = maximum entropy).
        H = 0 when only one language is used (zero diversity).

    Parameters
    ----------
    language_list : list of str
        e.g. ['Python', 'Go', 'Python', 'Rust']

    Returns
    -------
    float
        Shannon entropy rounded to 4 decimal places.

    Example
    -------
    >>> shannon_entropy(['Python', 'Python', 'Python'])
    0.0
    >>> shannon_entropy(['Python', 'Go', 'Rust'])
    1.585
    """
    if not language_list:
        return 0.0
    counts = {lang: language_list.count(lang) for lang in set(language_list)}
    total  = sum(counts.values()) + 1e-9
    return round(
        -sum((c / total) * math.log2(c / total + 1e-9)
             for c in counts.values()),
        4
    )


def compute_dds_score(
    proof_of_work: float,
    skill_genome: float,
    execution_pattern: float,
    thinking_blueprint: float,
    growth_signature: float
) -> float:
    """
    Computes the Developer DNA Score (DDS) using the weighted formula.

    Formula:
        DDS = 0.30 * Proof-of-Work
            + 0.25 * Skill Genome
            + 0.20 * Execution Pattern
            + 0.15 * Thinking Blueprint
            + 0.10 * Growth Signature

    All inputs must be normalised to [0, 1].

    Parameters
    ----------
    proof_of_work       : float  — commits, stars, PRs, issues
    skill_genome        : float  — language entropy, frameworks
    execution_pattern   : float  — readme, tests, CI/CD
    thinking_blueprint  : float  — fork ratio, impact weight
    growth_signature    : float  — trend slope, decay lambda

    Returns
    -------
    float : DDS score in [0, 1]
    """
    dds = (
        0.30 * proof_of_work +
        0.25 * skill_genome +
        0.20 * execution_pattern +
        0.15 * thinking_blueprint +
        0.10 * growth_signature
    )
    return round(float(np.clip(dds, 0.0, 1.0)), 4)


# ══════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def load_data(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Loads the DDM dataset and validates required columns exist.

    Parameters
    ----------
    path    : str   — path to CSV file
    verbose : bool  — print shape and null summary

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)

    required_cols = [
        'developer_id', 'total_commits', 'language_entropy',
        'has_readme_pct', 'developer_tier'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if verbose:
        print(f"✓ Loaded : {path}")
        print(f"  Shape  : {df.shape}")
        print(f"  Nulls  : {df.isnull().sum().sum()}")
        print(f"  Dupes  : {df.duplicated(subset='developer_id').sum()}")
        print(f"\n  Tier distribution:")
        for t, name in TIER_NAMES.items():
            n = (df['developer_tier'] == t).sum()
            print(f"    {name:<15}: {n:>5}  ({n/len(df)*100:.1f}%)")

    return df


def validate_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Validates that all expected features are present in X.
    Fills any missing columns with 0 and warns.

    Parameters
    ----------
    X            : pd.DataFrame — feature matrix
    feature_names: list of str  — expected column names

    Returns
    -------
    pd.DataFrame with exactly feature_names columns in order
    """
    missing = [f for f in feature_names if f not in X.columns]
    if missing:
        print(f"  ⚠ Missing features (filled with 0): {missing}")
        for col in missing:
            X[col] = 0.0

    extra = [c for c in X.columns if c not in feature_names]
    if extra:
        print(f"  ℹ Extra columns dropped: {extra}")

    return X[feature_names].fillna(0)


def load_features_and_labels(
    x_path: str,
    y_path: str,
    features_json_path: str
):
    """
    Loads X, y, and feature names from Notebook 2 outputs.

    Parameters
    ----------
    x_path           : path to X_features.csv
    y_path           : path to y_labels.csv
    features_json_path: path to selected_features.json

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    feature_names : list of str
    """
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()

    with open(features_json_path) as f:
        feature_names = json.load(f)

    print(f"✓ X shape         : {X.shape}")
    print(f"✓ y shape         : {y.shape}")
    print(f"✓ Features        : {len(feature_names)}")
    return X, y, feature_names


def save_results(results_dict: dict, path: str) -> None:
    """
    Saves model evaluation results dictionary to CSV.

    Parameters
    ----------
    results_dict : dict — keys are column names
    path         : str  — output path
    """
    pd.DataFrame(results_dict).to_csv(path, index=False)
    print(f"✓ Results saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# 3. PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str,
    cmap: str = 'Blues',
    ax=None,
    save_path: str = None
) -> None:
    """
    Plots a styled confusion matrix heatmap.

    Parameters
    ----------
    y_true     : array-like — true labels
    y_pred     : array-like — predicted labels
    title      : str        — plot title
    cmap       : str        — matplotlib colormap
    ax         : matplotlib Axes or None
    save_path  : str or None — path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap, ax=ax,
        xticklabels=TIER_LIST, yticklabels=TIER_LIST,
        linewidths=0.5, cbar=False
    )
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel('Actual Tier')
    ax.set_xlabel('Predicted Tier')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved → {save_path}")


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    original_cols: list,
    title: str,
    top_n: int = 15,
    save_path: str = None,
    ax=None
) -> None:
    """
    Plots horizontal bar chart of feature importances.
    Colours engineered features differently from original features.

    Parameters
    ----------
    feature_names : list       — all feature names
    importances   : np.ndarray — importance values (same length)
    original_cols : list       — list of original (non-engineered) columns
    title         : str        — plot title
    top_n         : int        — how many top features to show
    save_path     : str or None
    ax            : matplotlib Axes or None
    """
    imp_df = pd.DataFrame({
        'feature'   : feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)

    imp_df['engineered'] = ~imp_df['feature'].isin(original_cols)
    colors = ['#1B3A6B' if eng else '#85B7EB'
              for eng in imp_df['engineered']]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(imp_df['feature'], imp_df['importance'],
            color=colors, edgecolor='white')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel('Importance Score')

    legend_elems = [
        mpatches.Patch(color='#1B3A6B', label='Engineered feature'),
        mpatches.Patch(color='#85B7EB', label='Original feature'),
    ]
    ax.legend(handles=legend_elems, loc='lower right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved → {save_path}")


def plot_tier_comparison(
    df: pd.DataFrame,
    col: str,
    title: str,
    xlabel: str,
    save_path: str = None
) -> None:
    """
    Plots overlapping histograms of a feature split by developer tier.

    Parameters
    ----------
    df        : pd.DataFrame — must contain 'developer_tier' column
    col       : str          — column to plot
    title     : str
    xlabel    : str
    save_path : str or None
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for tier in [0, 1, 2]:
        subset = df[df['developer_tier'] == tier][col]
        ax.hist(subset, bins=40, alpha=0.6,
                color=TIER_COLORS[tier], edgecolor='white',
                label=f"{TIER_NAMES[tier]} (μ={subset.mean():.3f})")

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved → {save_path}")
    plt.show()


def plot_model_comparison_bar(
    model_names: list,
    f1_scores: list,
    title: str = 'F1 Macro Score — All Models Compared',
    save_path: str = None
) -> None:
    """
    Plots horizontal bar chart comparing F1 scores across models.

    Parameters
    ----------
    model_names : list of str   — model names
    f1_scores   : list of float — F1 scores
    title       : str
    save_path   : str or None
    """
    colors = ['#B5D4F4', '#D3D1C7', '#9FE1CB', '#5B2D8E']
    colors = colors[:len(model_names)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, f1_scores, color=colors,
                  edgecolor='white', linewidth=0.8, width=0.55)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Macro Score')
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', fontweight='bold', fontsize=11
        )

    best = max(f1_scores)
    ax.axhline(best, color='#1B3A6B', linewidth=1,
               linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 4. STATISTICAL HELPERS
# ══════════════════════════════════════════════════════════════════════════

def skewness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarising skewness for all numeric columns.
    Flags columns needing log1p transformation (|skew| > 2).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with columns: feature, skewness, kurtosis, action
    """
    numeric = df.select_dtypes(include=np.number).columns
    rows = []
    for col in numeric:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        action = "→ apply log1p()" if abs(skew) > 2 else "✓ no transform"
        rows.append({
            'feature'  : col,
            'skewness' : round(skew, 3),
            'kurtosis' : round(kurt, 3),
            'action'   : action
        })
    return pd.DataFrame(rows).sort_values('skewness', ascending=False)


def gini_impurity(y) -> float:
    """
    Computes Gini impurity for a class label array.

    Formula: G = 1 - sum(p_i^2)
    G = 0      → perfectly imbalanced (one class only)
    G = 0.667  → perfectly balanced (3 equal classes)

    Parameters
    ----------
    y : array-like of class labels

    Returns
    -------
    float
    """
    y      = pd.Series(y)
    counts = y.value_counts()
    probs  = counts / counts.sum()
    return round(float(1 - (probs ** 2).sum()), 4)


def compute_zscore(series: pd.Series, value: float) -> float:
    """
    Computes the z-score of a single value relative to a Series.

    z = (x - mean) / std

    Parameters
    ----------
    series : pd.Series
    value  : float

    Returns
    -------
    float : z-score (positive = above mean, negative = below mean)
    """
    mean = series.mean()
    std  = series.std() + 1e-9
    return round((value - mean) / std, 3)
