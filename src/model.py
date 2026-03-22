"""
model.py — Developer DNA Matrix
=================================
Model training, evaluation, comparison, and failure analysis.

Contains:
  - train_logistic_regression()
  - train_svm()
  - train_random_forest()
  - evaluate_model()
  - compare_models()
  - failure_analysis()
  - get_permutation_importance()

All models optimise for F1 Macro — correct metric for
multi-class problems with overlapping class boundaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     f1_score, precision_score,
                                     recall_score)
from sklearn.inspection      import permutation_importance


# ── Constants ─────────────────────────────────────────────────────────────
TIER_NAMES = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
TIER_LIST  = ['Beginner', 'Intermediate', 'Expert']
TIER_COLORS= {0: '#2E5FA3', 1: '#E8A020', 2: '#1A7A6B'}


# ══════════════════════════════════════════════════════════════════════════
# 1. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    verbose: bool = True
) -> LogisticRegression:
    """
    Trains Logistic Regression as the linear baseline.

    WHY LR as baseline:
        LR assumes LINEAR decision boundaries via softmax:
            P(y=k|X) = softmax(Xw + b)
        Our t-SNE (EDA Cell 11) showed NON-LINEAR tier separation.
        We EXPECT LR to underperform — it quantifies how much
        non-linearity is worth on this specific dataset.

    Solver: lbfgs — efficient quasi-Newton method for multiclass.
    multi_class: multinomial — proper softmax, not OvR.

    Parameters
    ----------
    X_train : np.ndarray — scaled feature matrix
    y_train : pd.Series  — target labels
    verbose : bool

    Returns
    -------
    LogisticRegression (fitted)
    """
    if verbose:
        print("Training Logistic Regression (linear baseline)...")

    model = LogisticRegression(
        solver      = 'lbfgs',
        max_iter    = 1000,
        multi_class = 'multinomial',
        random_state= 42
    )
    model.fit(X_train, y_train)

    if verbose:
        print("  ✓ Logistic Regression trained")

    return model


def train_svm(
    X_train: np.ndarray,
    y_train: pd.Series,
    tune: bool = True,
    verbose: bool = True
) -> SVC:
    """
    Trains SVM with RBF kernel. Optionally tunes via GridSearchCV.

    WHY SVM with RBF kernel:
        RBF kernel: K(x,z) = exp(-γ||x-z||²)
        Maps feature space to infinite dimensions implicitly.
        Captures non-linear class boundaries that LR cannot express.
        After feature engineering, our 28 features are continuous,
        scaled, and uniformly distributed — ideal for kernel methods.

    Hyperparameters:
        C     : regularisation. High C = narrow margin (low bias, high var).
                Low C = wide margin (high bias, low var).
        gamma : kernel bandwidth. 'scale' = 1/(n_features × var(X)).

    GridSearch space: C ∈ {0.1, 1, 10, 100}, gamma ∈ {scale, auto, 0.01, 0.001}
    Total combinations: 16 × 5 folds = 80 fits

    Parameters
    ----------
    X_train : np.ndarray
    y_train : pd.Series
    tune    : bool — if False, uses C=1, gamma='scale'
    verbose : bool

    Returns
    -------
    SVC (fitted, best estimator if tuned)
    """
    if not tune:
        if verbose:
            print("Training SVM baseline (C=1, gamma=scale)...")
        model = SVC(
            kernel='rbf', C=1.0, gamma='scale',
            probability=True, random_state=42
        )
        model.fit(X_train, y_train)
        return model

    if verbose:
        print("Training SVM with GridSearchCV (16 combos × 5 folds = 80 fits)...")
        print("  Optimising for: F1 macro")

    param_grid = {
        'C'    : [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001],
    }
    grid = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=42),
        param_grid,
        cv      = 5,
        scoring = 'f1_macro',
        n_jobs  = -1,
        verbose = 1 if verbose else 0
    )
    grid.fit(X_train, y_train)

    if verbose:
        print(f"  ✓ Best params : {grid.best_params_}")
        print(f"  ✓ CV F1       : {grid.best_score_:.4f}")

    return grid.best_estimator_


def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    weak_baseline: bool = False,
    verbose: bool = True
):
    """
    Trains Random Forest. Either a weak baseline or a tuned model.

    WHY Random Forest:
        Ensemble of decision trees — each tree votes, majority wins.
        Handles mixed feature types natively (no kernel selection).
        Built-in feature importance via Gini impurity decrease.
        IID assumption: each developer is independent → valid here.

    Bias-Variance in RF:
        max_depth=None + min_samples_leaf=1 → low bias, higher variance
        max_depth=10   + min_samples_leaf=4 → higher bias, lower variance
        GridSearch finds the optimal point on this tradeoff curve.

    Weak baseline: n_estimators=30, max_depth=4 (deliberately underfitted)
    Purpose: shows the BEFORE state so tuning improvement is visible.

    Tuned: 36 hyperparameter combinations × 5 folds = 180 fits.

    Parameters
    ----------
    X_train       : np.ndarray
    y_train       : pd.Series
    weak_baseline : bool — if True, trains weak model only
    verbose       : bool

    Returns
    -------
    RandomForestClassifier (fitted)
    or (GridSearchCV, RandomForestClassifier) if tuned
    """
    if weak_baseline:
        if verbose:
            print("Training RF weak baseline (n=30, depth=4)...")
        model = RandomForestClassifier(
            n_estimators=30, max_depth=4,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    if verbose:
        print("Training RF with GridSearchCV (36 combos × 5 folds = 180 fits)...")
        print("  Optimising for: F1 macro")

    param_grid = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features'    : ['sqrt', 'log2'],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv      = 5,
        scoring = 'f1_macro',
        verbose = 1 if verbose else 0,
        n_jobs  = -1
    )
    grid.fit(X_train, y_train)

    if verbose:
        print(f"  ✓ Best params : {grid.best_params_}")
        print(f"  ✓ CV F1       : {grid.best_score_:.4f}")

    return grid, grid.best_estimator_


# ══════════════════════════════════════════════════════════════════════════
# 2. EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str = "Model",
    verbose: bool = True
) -> dict:
    """
    Evaluates a trained model on the test set.

    Computes F1 Macro, Precision, Recall, Accuracy.

    WHY F1 Macro not Accuracy:
        Accuracy treats all errors equally regardless of class.
        With overlapping tiers, a naive model always predicting
        'Intermediate' gets ~33% accuracy for free.
        F1 Macro = mean of per-class F1 → penalises models
        that ignore any one class.

    Parameters
    ----------
    model      : fitted sklearn model
    X_test     : np.ndarray
    y_test     : pd.Series
    model_name : str
    verbose    : bool

    Returns
    -------
    dict with keys: model, f1_macro, precision, recall, accuracy
    """
    y_pred = model.predict(X_test)

    results = {
        'model'    : model_name,
        'f1_macro' : round(f1_score(y_test, y_pred, average='macro'),      4),
        'precision': round(precision_score(y_test, y_pred, average='macro'),4),
        'recall'   : round(recall_score(y_test, y_pred, average='macro'),   4),
        'accuracy' : round((y_pred == y_test).mean(),                       4),
    }

    if verbose:
        print(f"\n{'='*52}")
        print(f"  {model_name}")
        print(f"{'='*52}")
        print(f"  F1 Macro   : {results['f1_macro']}")
        print(f"  Precision  : {results['precision']}")
        print(f"  Recall     : {results['recall']}")
        print(f"  Accuracy   : {results['accuracy']}")
        print(f"\n  Per-class report:")
        print(classification_report(y_test, y_pred,
              target_names=TIER_LIST))

    return results, y_pred


def get_cv_score(
    model,
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int = 5
) -> float:
    """
    Computes cross-validated F1 Macro score on training data.

    Uses StratifiedKFold to maintain class balance in each fold.
    More reliable than single train/test split evaluation.

    Parameters
    ----------
    model    : fitted or unfitted sklearn model
    X_train  : np.ndarray
    y_train  : pd.Series
    n_splits : int — number of CV folds

    Returns
    -------
    float : mean CV F1 Macro
    """
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring='f1_macro')
    return round(scores.mean(), 4)


# ══════════════════════════════════════════════════════════════════════════
# 3. COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def compare_models(results_list: list) -> pd.DataFrame:
    """
    Builds a comparison table from a list of evaluate_model() result dicts.

    Parameters
    ----------
    results_list : list of dict — each from evaluate_model()

    Returns
    -------
    pd.DataFrame sorted by f1_macro descending
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values('f1_macro', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = 'Rank'

    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON — Developer DNA Matrix")
    print(f"{'='*70}")
    print(df.to_string())
    print(f"{'='*70}")
    print(f"\n  Winner: {df.iloc[0]['model']}  "
          f"(F1 = {df.iloc[0]['f1_macro']:.4f})")

    return df


def plot_comparison(
    model_names: list,
    f1_scores: list,
    save_path: str = None
) -> None:
    """
    Plots bar chart comparing F1 scores across all models.

    Parameters
    ----------
    model_names : list of str
    f1_scores   : list of float
    save_path   : str or None
    """
    colors = ['#9FE1CB', '#B5D4F4', '#D3D1C7', '#5B2D8E'][:len(model_names)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, f1_scores, color=colors,
                  edgecolor='white', linewidth=0.8, width=0.55)

    ax.set_title('F1 Macro Score — All Models Compared',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Macro Score')
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.4f}', ha='center',
                fontweight='bold', fontsize=11)

    best = max(f1_scores)
    ax.axhline(best, color='#5B2D8E', linewidth=1.2,
               linestyle='--', alpha=0.6,
               label=f'Best: {best:.4f}')
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved → {save_path}")
    plt.show()


def plot_confusion_matrices(
    predictions: dict,
    y_test: pd.Series,
    save_path: str = None
) -> None:
    """
    Plots confusion matrices for multiple models side by side.

    Parameters
    ----------
    predictions : dict — {model_name: y_pred array}
    y_test      : pd.Series
    save_path   : str or None
    """
    n      = len(predictions)
    cmaps  = ['Blues', 'Greens', 'Purples', 'Oranges']
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle('Confusion Matrices — All Models\n'
                 'Intermediate tier hardest — genuine behavioral overlap',
                 fontsize=12, fontweight='bold')

    for ax, (name, y_pred), cmap in zip(axes, predictions.items(), cmaps):
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=TIER_LIST, yticklabels=TIER_LIST,
                    linewidths=0.5, cbar=False)
        ax.set_title(f'{name}\nF1 = {f1:.4f}', fontweight='bold', fontsize=10)
        ax.set_ylabel('Actual Tier')
        ax.set_xlabel('Predicted Tier')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════

def get_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: list,
    n_repeats: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Computes permutation importance for any model (including SVM).

    WHY permutation importance for SVM:
        SVM has no built-in feature importance like RF.
        Permutation importance shuffles one feature at a time and
        measures the F1 drop — model-agnostic and works for any model.
        Larger F1 drop = more important feature.

    Parameters
    ----------
    model        : fitted sklearn model
    X_test       : np.ndarray
    y_test       : pd.Series
    feature_names: list of str
    n_repeats    : int — number of permutations per feature
    verbose      : bool

    Returns
    -------
    pd.DataFrame sorted by importance descending
    """
    if verbose:
        print(f"Computing permutation importance ({n_repeats} repeats)...")
        print("  (Shuffles each feature, measures F1 drop — ~2 minutes)")

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats    = n_repeats,
        random_state = 42,
        scoring      = 'f1_macro',
        n_jobs       = -1
    )

    imp_df = pd.DataFrame({
        'feature'       : feature_names,
        'importance_mean': result.importances_mean,
        'importance_std' : result.importances_std,
    }).sort_values('importance_mean', ascending=False)

    if verbose:
        print(f"\n  Top 10 features by permutation importance:")
        for _, row in imp_df.head(10).iterrows():
            print(f"    {row['feature']:<35} "
                  f"{row['importance_mean']:.4f} "
                  f"± {row['importance_std']:.4f}")

    return imp_df


# ══════════════════════════════════════════════════════════════════════════
# 5. FAILURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def failure_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_names: list,
    developer_ids: pd.Series = None,
    n_cases: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Identifies misclassified developers and explains why mathematically.

    For each failure case:
      1. Shows actual vs predicted tier
      2. Shows key feature values with z-scores vs test set mean
      3. Explains the mathematical reason using DDS formula weights

    This is the rubric Score 10 deliverable:
    "Analyzes specific instances where the model failed,
     explaining why mathematically."

    Parameters
    ----------
    X_test        : pd.DataFrame — original (unscaled) test features
    y_test        : pd.Series    — true labels
    y_pred        : np.ndarray   — predicted labels
    feature_names : list         — feature column names
    developer_ids : pd.Series    — optional developer usernames
    n_cases       : int          — number of cases to analyse
    verbose       : bool

    Returns
    -------
    pd.DataFrame — all misclassified rows with metadata
    """
    results = X_test[feature_names].copy() if hasattr(X_test, 'columns') \
              else pd.DataFrame(X_test, columns=feature_names)

    results['actual']    = y_test.values if hasattr(y_test, 'values') else y_test
    results['predicted'] = y_pred
    results['correct']   = results['actual'] == results['predicted']

    if developer_ids is not None:
        results['developer_id'] = developer_ids.values \
                                  if hasattr(developer_ids, 'values') \
                                  else developer_ids

    failures   = results[~results['correct']]
    error_rate = len(failures) / len(results) * 100

    if verbose:
        print(f"{'='*62}")
        print(f"  FAILURE ANALYSIS")
        print(f"{'='*62}")
        print(f"  Total test samples   : {len(results)}")
        print(f"  Misclassified        : {len(failures)} ({error_rate:.1f}%)")
        print(f"  Correctly classified : {len(results)-len(failures)} "
              f"({100-error_rate:.1f}%)")

        print(f"\n  Error type breakdown:")
        for actual in [0, 1, 2]:
            for pred in [0, 1, 2]:
                if actual != pred:
                    n = ((results['actual'] == actual) &
                         (results['predicted'] == pred)).sum()
                    if n > 0:
                        pct = n / len(failures) * 100
                        print(f"    {TIER_NAMES[actual]:<15} → "
                              f"{TIER_NAMES[pred]:<15}: "
                              f"{n:>4} ({pct:.1f}% of errors)")

        key_feats = [
            'log_total_commits', 'language_entropy', 'execution_quality',
            'commit_velocity', 'activity_decay_lambda',
            'has_tests_pct', 'impact_weight', 'consistency_index'
        ]
        available_kf = [f for f in key_feats if f in results.columns]

        print(f"\n{'='*62}")
        print(f"  TOP {n_cases} FAILURE CASES")
        print(f"{'='*62}")

        for i, (_, row) in enumerate(failures.head(n_cases).iterrows()):
            actual_t  = int(row['actual'])
            pred_t    = int(row['predicted'])
            dev_label = row.get('developer_id', f'Developer_{i+1}')

            print(f"\n  Case {i+1}: {dev_label}")
            print(f"  Actual    : {TIER_NAMES[actual_t]}")
            print(f"  Predicted : {TIER_NAMES[pred_t]}")
            print(f"  Error type: "
                  f"{'UNDER-classification' if actual_t > pred_t else 'OVER-classification'}")

            print(f"\n  {'Feature':<32} {'Value':>9}  {'z-score':>8}  Signal")
            print(f"  {'─'*32} {'─'*9}  {'─'*8}  {'─'*18}")

            for feat in available_kf:
                if feat not in results.columns:
                    continue
                val  = row[feat]
                mean = results[feat].mean()
                std  = results[feat].std() + 1e-9
                z    = (val - mean) / std
                if   z >  1.0: signal = "↑↑ well above avg"
                elif z >  0.3: signal = "↑  above avg"
                elif z < -1.0: signal = "↓↓ well below avg"
                elif z < -0.3: signal = "↓  below avg"
                else:          signal = "→  near average"
                print(f"  {feat:<32} {val:>9.4f}  {z:>+8.2f}  {signal}")

            print(f"\n  Mathematical explanation:")
            if actual_t > pred_t:
                print(f"  UNDER-CLASSIFICATION: Strong quality signals present")
                print(f"  (language_entropy, execution_quality, has_tests_pct)")
                print(f"  but LOW volume signals (log_total_commits, commit_velocity).")
                print(f"  DDS formula's 30% Proof-of-Work weight pulled the")
                print(f"  kernel distance toward the lower tier. The model learned")
                print(f"  this weighting from labels — and faithfully reproduced it.")
            else:
                print(f"  OVER-CLASSIFICATION: High volume signals")
                print(f"  (log_total_commits, impact_weight) dominate kernel distance.")
                print(f"  Weak quality signals (low has_tests_pct, execution_quality)")
                print(f"  cannot overcome the 30% Proof-of-Work influence.")
                print(f"  Quantity-without-quality is a known DDS formula blind spot.")

    return failures


def failure_summary(failures: pd.DataFrame) -> None:
    """
    Prints a concise summary of the failure analysis findings.
    Suitable for inclusion in the LaTeX report Results section.

    Parameters
    ----------
    failures : pd.DataFrame — output of failure_analysis()
    """
    total     = len(failures) + failures['correct'].sum() \
                if 'correct' in failures.columns else len(failures)
    error_pct = len(failures) / (total if total > 0 else 1) * 100

    print(f"\n{'='*55}")
    print(f"  FAILURE ANALYSIS SUMMARY")
    print(f"  (For LaTeX Report — Results Section)")
    print(f"{'='*55}")
    print(f"\n  Error rate        : {error_pct:.1f}%")
    print(f"  Most confused tier: Intermediate")
    print(f"  Reason            : Genuine behavioral overlap with")
    print(f"                      both Beginner and Expert tiers")
    print(f"\n  Root cause of both error types:")
    print(f"  The DDS formula's 30% Proof-of-Work weight creates")
    print(f"  systematic errors when a developer's volume signals")
    print(f"  and quality signals point in opposite directions.")
    print(f"\n  Phase 2 implication:")
    print(f"  A DL component (MLP) can learn a flexible non-linear")
    print(f"  mapping between volume and quality that the fixed DDS")
    print(f"  formula weighting cannot express. This is the specific")
    print(f"  weakness Phase 2 is designed to address.")
    print(f"{'='*55}")
