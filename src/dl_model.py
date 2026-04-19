"""
dl_model.py — Developer DNA Matrix (Phase 2)
=============================================
MLP using PyTorch — works on M2 Mac, all Python versions.

Why MLP over SVM (Phase 1 winner)?
    SVM draws boundaries based on distance between points.
    MLP learns combinations of features layer by layer.
    MLP can learn "high commits + low tests = Intermediate"
    as a combined pattern — SVM cannot do this explicitly.

Architecture:
    Input(28) → Dense(128) → BatchNorm → Dropout
              → Dense(64)  → BatchNorm → Dropout
              → Dense(3)   → Softmax
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ── Constants ──────────────────────────────────────────────────────────────
TIER_NAMES  = ['Beginner', 'Intermediate', 'Expert']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Use MPS (M2 GPU) if available, else CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# 1. BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════

class MLPModel(nn.Module):
    """
    Simple MLP with 2 hidden layers.

    Layer 1 - Dense(128, ReLU):
        28 inputs expand to 128 neurons.
        Learns 128 different combinations of your features.

    BatchNorm:
        Normalizes outputs to mean=0, std=1 after each layer.
        Makes training stable — handles features at different scales.

    Dropout:
        Randomly turns off neurons during training.
        Forces the network not to rely on any single neuron.
        Result: less overfitting.

    Layer 2 - Dense(64, ReLU):
        Compresses 128 patterns into 64 higher-level patterns.

    Output - Dense(3):
        3 outputs = 3 tiers (Beginner, Intermediate, Expert).
        Softmax applied during loss calculation (CrossEntropyLoss).
    """

    def __init__(self, input_dim=28, use_batchnorm=True, use_dropout=True):
        super(MLPModel, self).__init__()

        layers = []

        # --- Hidden layer 1 ---
        layers.append(nn.Linear(input_dim, 128))
        layers.append(nn.ReLU())
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(128))
        if use_dropout:
            layers.append(nn.Dropout(0.3))

        # --- Hidden layer 2 ---
        layers.append(nn.Linear(128, 64))
        layers.append(nn.ReLU())
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(64))
        if use_dropout:
            layers.append(nn.Dropout(0.2))

        # --- Output layer ---
        layers.append(nn.Linear(64, 3))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ══════════════════════════════════════════════════════════════════════════
# 2. TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════

def get_class_weights(y_train):
    """
    Computes class weights so all tiers are treated equally.
    Intermediate is the largest class — without weights the model
    learns to over-predict Intermediate.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    print("  Class weights:")
    for tier, w in class_weight_dict.items():
        print(f"    {TIER_NAMES[tier]:<15}: {w:.4f}")

    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_model(
    X_train, y_train,
    X_val,   y_val,
    use_batchnorm = True,
    use_dropout   = True,
    epochs        = 100,
    batch_size    = 64,
    verbose       = 1
):
    """
    Trains the MLP model.

    Optimizer — Adam:
        Automatically adjusts learning rate during training.
        Better than SGD for tabular data with mixed feature scales.

    Loss — CrossEntropyLoss:
        Standard loss for multi-class classification.
        Includes Softmax internally.

    Early Stopping:
        If val loss does not improve for 15 epochs → stop training.
        Saves the best model weights automatically.
    """
    input_dim = X_train.shape[1]
    model     = MLPModel(input_dim, use_batchnorm, use_dropout).to(DEVICE)

    # Class weights
    class_weights = get_class_weights(y_train)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = optim.Adam(model.parameters(), lr=0.001)
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=7, factor=0.5
                    )

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_train.values, dtype=torch.long).to(DEVICE)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(DEVICE)
    y_vl = torch.tensor(y_val.values,   dtype=torch.long).to(DEVICE)

    # DataLoader for batching
    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training history
    history = {
        'loss': [], 'val_loss': [],
        'accuracy': [], 'val_accuracy': []
    }

    # Early stopping variables
    best_val_loss   = float('inf')
    patience_count  = 0
    patience        = 15
    best_weights    = None

    print(f"\n  Training MLP (max {epochs} epochs, early stopping patience=15)...")

    for epoch in range(epochs):

        # ── Training phase ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(y_batch)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        avg_train_loss = train_loss / train_total
        avg_train_acc  = train_correct / train_total

        # ── Validation phase ──
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_vl)
            val_loss    = criterion(val_outputs, y_vl).item()
            val_preds   = val_outputs.argmax(dim=1)
            val_acc     = (val_preds == y_vl).float().mean().item()

        scheduler.step(val_loss)

        # Save history
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(avg_train_acc)
        history['val_accuracy'].append(val_acc)

        if verbose == 1 and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3} | "
                  f"Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_weights   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"  ✓ Training complete. Best val loss: {best_val_loss:.4f}")

    return model, history


# ══════════════════════════════════════════════════════════════════════════
# 3. EVALUATE MODEL
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, model_name="MLP"):
    """
    Evaluates the trained model on test set.
    Reports F1 Macro, Accuracy, and per-class breakdown.
    """
    model.eval()
    X_ts = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(X_ts)
        y_pred  = outputs.argmax(dim=1).cpu().numpy()

    y_true = y_test.values if hasattr(y_test, 'values') else y_test

    f1  = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Set Results")
    print(f"{'='*50}")
    print(f"  F1 Macro : {f1:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"\n  Per-class breakdown:")
    print(classification_report(y_true, y_pred, target_names=TIER_NAMES))

    results = {
        'model'    : model_name,
        'f1_macro' : round(f1,  4),
        'accuracy' : round(acc, 4),
    }

    return results, y_pred


# ══════════════════════════════════════════════════════════════════════════
# 4. ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════

def run_ablation(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Tests 3 variants — removes one component at a time.
    Proves each component actually contributes to performance.
    """
    variants = [
        {"name": "No Dropout",   "use_dropout": False, "use_batchnorm": True},
        {"name": "No BatchNorm", "use_dropout": True,  "use_batchnorm": False},
        {"name": "Full Model",   "use_dropout": True,  "use_batchnorm": True},
    ]

    ablation_results = []

    for v in variants:
        print(f"\n--- Ablation: {v['name']} ---")
        model, _ = train_model(
            X_train, y_train, X_val, y_val,
            use_dropout   = v['use_dropout'],
            use_batchnorm = v['use_batchnorm'],
            epochs        = 100,
            verbose       = 0
        )
        results, _ = evaluate_model(model, X_test, y_test, v['name'])
        ablation_results.append(results)
        print(f"  → F1 Macro: {results['f1_macro']:.4f}")

    return ablation_results


# ══════════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, save_path=None):
    """
    Plots training loss and validation loss per epoch.
    Good sign: both lines drop together.
    Bad sign: train drops but val rises = overfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('MLP Training History', fontsize=13, fontweight='bold')

    axes[0].plot(history['loss'],     label='Train Loss',      color='steelblue')
    axes[0].plot(history['val_loss'], label='Validation Loss', color='darkorange')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['accuracy'],     label='Train Accuracy',      color='steelblue')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='darkorange')
    axes[1].set_title('Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """
    Rows = actual tier, Columns = predicted tier.
    Diagonal = correct. Off-diagonal = mistakes.
    """
    y_true = y_test.values if hasattr(y_test, 'values') else y_test
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=TIER_NAMES, yticklabels=TIER_NAMES,
        linewidths=0.5
    )
    plt.title('MLP Confusion Matrix\n(rows = actual, columns = predicted)',
              fontweight='bold')
    plt.ylabel('Actual Tier')
    plt.xlabel('Predicted Tier')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    plt.show()


def plot_all_models(all_results, save_path=None):
    """
    Bar chart comparing F1 scores across all models.
    """
    names  = [r['model']    for r in all_results]
    scores = [r['f1_macro'] for r in all_results]

    colors   = ['#B5C9E8'] * len(names)
    best_idx = scores.index(max(scores))
    colors[best_idx] = '#2E5FA3'

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, scores, color=colors, edgecolor='white', width=0.55)

    for bar, val in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold'
        )

    plt.title('F1 Macro — All Models (Phase 1 ML vs Phase 2 MLP)',
              fontsize=12, fontweight='bold')
    plt.ylabel('F1 Macro Score')
    plt.ylim(0.70, 0.90)
    plt.axhline(0.8116, color='gray', linestyle='--',
                linewidth=1, label='Phase 1 Best (SVM = 0.8116)')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    plt.show()


def plot_ablation(ablation_results, save_path=None):
    """
    Bar chart showing F1 for each ablation variant.
    """
    names  = [r['model']    for r in ablation_results]
    scores = [r['f1_macro'] for r in ablation_results]
    colors = ['#E8A020', '#E05C5C', '#1A7A6B']

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, scores, color=colors, edgecolor='white', width=0.4)

    for bar, val in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold'
        )

    plt.title('Ablation Study — Effect of BatchNorm and Dropout',
              fontsize=11, fontweight='bold')
    plt.ylabel('F1 Macro Score')
    plt.ylim(min(scores) - 0.02, max(scores) + 0.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    plt.show()