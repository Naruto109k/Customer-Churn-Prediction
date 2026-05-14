import numpy as np
import os
from sklearn.model_selection import train_test_split

from utils.preprocessing import load_and_preprocess
from utils.metrics import evaluate_model, plot_confusion_matrix, plot_model_comparison
from models.random_forest import RandomForestClassifier
from models.extra_trees import ExtraTreesClassifier
from models.gradient_boosting import GradientBoostingClassifier
from models.svm import SVM
from explainability.shap_analysis import shap_all_models
from explainability.lime_analysis import lime_all_models

os.makedirs("outputs", exist_ok=True)

print("\n" + "="*50)
print("   CUSTOMER CHURN PREDICTION — FROM SCRATCH")
print("="*50)

# ── Data ───────────────────────────────────────────────────────

X, X_scaled, y, feature_names, scaler = load_and_preprocess("data/attrition.csv")

# Tree models work on raw features; SVM needs standardized input
X_raw    = X.values
X_scaled = X_scaled.values
y        = y.values

# Both splits use the same random state and stratification so the test set
# is identical — raw and scaled rows correspond to the same employees
X_train_raw,    X_test_raw,    y_train, y_test = train_test_split(
    X_raw,    y, test_size=0.2, random_state=42, stratify=y)
X_train_scaled, X_test_scaled, _,       _      = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ── Training ───────────────────────────────────────────────────

print("\nTraining all models from scratch...\n")

results = []

print("1. Random Forest")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
results.append(evaluate_model("Random Forest", rf,
                               X_train_raw, y_train,
                               X_test_raw,  y_test))

print("\n2. Extra Trees")
et = ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42)
results.append(evaluate_model("Extra Trees", et,
                               X_train_raw, y_train,
                               X_test_raw,  y_test))

print("\n3. Gradient Boosting")
gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                max_depth=3, random_state=42)
results.append(evaluate_model("Gradient Boosting", gb,
                               X_train_raw, y_train,
                               X_test_raw,  y_test))

print("\n4. SVM")
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
results.append(evaluate_model("SVM", svm,
                               X_train_scaled, y_train,
                               X_test_scaled,  y_test))

# ── Evaluation ─────────────────────────────────────────────────

print("\nGenerating evaluation plots...")
plot_confusion_matrix(results, y_test)
plot_model_comparison(results)

print("\n" + "="*50)
print("           FINAL RESULTS SUMMARY")
print("="*50)

best = max(results, key=lambda x: x["Accuracy"])
for r in results:
    marker = "* " if r["name"] == best["name"] else "  "
    print(f"{marker}{r['name']:<22} "
          f"Acc: {r['Accuracy']*100:.2f}%  "
          f"F1: {r['F1 Score']:.4f}  "
          f"AUC: {r['AUC-ROC']:.4f}  "
          f"Time: {r['Train Time']}s")

print(f"\nBest model: {best['name']} ({best['Accuracy']*100:.2f}% accuracy)")
print("="*50)

# ── Explainability ─────────────────────────────────────────────

models_dict = {
    "Random Forest":     rf,
    "Extra Trees":       et,
    "Gradient Boosting": gb,
    "SVM":               svm
}

# SHAP gives a global view — which features matter most across all predictions
shap_all_models(
    models_dict,
    X_train_scaled, X_test_raw,
    X_test_scaled,  feature_names
)

# LIME gives a local view — why the model made this specific prediction
lime_all_models(
    models_dict,
    X_train_raw,    X_test_raw,
    X_train_scaled, X_test_scaled,
    feature_names,  instance_idx=0
)

print("\nAll outputs saved to /outputs")