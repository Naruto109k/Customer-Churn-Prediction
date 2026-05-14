import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Trains a model, times it, and returns a results dict with all metrics.
    The dict is used downstream by the plotting functions and the summary table.
    """
    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 4)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Train Time: {train_time}s")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Stay', 'Leave'])}")

    return {
        "name": name,
        "model": model,
        "Accuracy": acc,
        "F1 Score": f1,
        "AUC-ROC": auc,
        "Train Time": train_time,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def plot_confusion_matrix(results, y_test):
    """
    Plots a 2x2 grid of confusion matrices, one per model.
    Rows = actual class, columns = predicted class.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, result in enumerate(results):
        cm = confusion_matrix(y_test, result["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Stay', 'Leave'],
                    yticklabels=['Stay', 'Leave'])
        axes[idx].set_title(
            f'{result["name"]}\nAccuracy: {result["Accuracy"]*100:.2f}%',
            fontweight='bold'
        )
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')

    plt.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results):
    """
    Two side-by-side bar charts:
      - Left:  Accuracy, F1, and AUC-ROC for each model
      - Right: Training time in seconds

    Separating performance from speed makes it easier to reason about
    the accuracy/efficiency trade-off across models.
    """
    names      = [r["name"] for r in results]
    accuracies = [r["Accuracy"] * 100 for r in results]
    f1_scores  = [r["F1 Score"] for r in results]
    auc_scores = [r["AUC-ROC"] for r in results]
    times      = [r["Train Time"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars1 = ax1.bar(x - width, accuracies, width, label='Accuracy (%)', color='steelblue', alpha=0.85)
    bars2 = ax1.bar(x,         f1_scores,  width, label='F1 Score',     color='darkorange', alpha=0.85)
    bars3 = ax1.bar(x + width, auc_scores, width, label='AUC-ROC',      color='mediumseagreen', alpha=0.85)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars3:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(['RF', 'ET', 'GBM', 'SVM'], fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)

    colors = ['steelblue', 'darkorange', 'mediumseagreen', 'tomato']
    ax2.bar(names, times, color=colors, alpha=0.85)
    ax2.set_title('Training Time Comparison (seconds)', fontweight='bold')
    ax2.set_ylabel('Seconds')
    ax2.set_xticklabels(['RF', 'ET', 'GBM', 'SVM'], fontsize=11)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    for i, v in enumerate(times):
        ax2.text(i, v + 0.01, f'{v}s', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()