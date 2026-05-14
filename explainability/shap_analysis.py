import numpy as np
import matplotlib.pyplot as plt
import shap
import os

os.makedirs("outputs", exist_ok=True)


def shap_explain_model(model, X_train, X_test, feature_names, model_name, n_samples=50):
    """
    Computes SHAP values for a single model using KernelExplainer.

    KernelExplainer is model-agnostic — it works by perturbing the input and
    observing how the prediction changes, so it's compatible with our custom
    implementations that sklearn's TreeExplainer wouldn't support.

    The background dataset is compressed to 10 k-means centroids to keep
    computation manageable. n_samples controls how many test instances we explain.

    Returns the raw SHAP values, sorted feature indices, and mean absolute
    SHAP values — used by shap_all_models to build the combined plot.
    """
    print(f"\nComputing SHAP values for {model_name}...")

    background  = shap.kmeans(X_train, 10)
    explainer   = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test[:n_samples])

    # shap_values can come back as a list [class_0, class_1] or a 3D array
    # depending on the explainer version — we always want the class-1 slice
    sv = np.array(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    elif isinstance(shap_values, list):
        sv = np.array(shap_values[1])

    # Beeswarm plot: shows direction and magnitude of each feature's impact
    # Red = high feature value, blue = low; right = pushes toward churn
    plt.figure()
    shap.summary_plot(sv, X_test[:n_samples], feature_names=feature_names, show=False)
    plt.title(f"{model_name} — SHAP Beeswarm", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"outputs/shap_beeswarm_{model_name.replace(' ', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    # Feature importance bar: mean absolute SHAP value — how much each feature
    # moves the prediction on average, regardless of direction
    mean_shap  = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1].astype(int)

    colors = {'Random Forest': 'steelblue', 'Extra Trees': 'mediumseagreen',
              'Gradient Boosting': 'darkorange', 'SVM': 'tomato'}

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in sorted_idx],
        mean_shap[sorted_idx],
        color=colors.get(model_name, 'steelblue'), alpha=0.85
    )
    plt.title(f"{model_name} — SHAP Feature Importance", fontweight='bold')
    plt.xlabel("Mean |SHAP Value|")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"outputs/shap_importance_{model_name.replace(' ', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    return sv, sorted_idx, mean_shap


def shap_all_models(models_dict, X_train_scaled, X_test_raw,
                    X_test_scaled, feature_names):
    """
    Runs SHAP explanation for all four models and saves individual plots,
    then produces a combined side-by-side importance chart.

    Tree models (RF, ET, GBM) receive raw unscaled features — they don't
    need scaling and it keeps their SHAP values interpretable in original units.
    SVM receives scaled features because that's what it was trained on.

    Note: KernelExplainer is slow. Expect 10-15 minutes total for all models.
    """
    print("\n" + "="*50)
    print("   SHAP EXPLAINABILITY — ALL MODELS")
    print("="*50)
    print("   Using KernelExplainer (model-agnostic, works with custom models)")
    print("   This will take 10-15 minutes total...\n")

    shap_results = {}

    for name, model in models_dict.items():
        X_train = X_train_scaled
        X_test  = X_test_scaled if name == "SVM" else X_test_raw

        sv, sorted_idx, mean_shap = shap_explain_model(
            model, X_train, X_test, feature_names, name
        )
        shap_results[name] = sv

        print(f"   Top 3 features for {name}:")
        for i in sorted_idx[:3]:
            print(f"      - {feature_names[i]}: {mean_shap[i]:.4f}")

    # Combined plot: top-10 features per model in a 2x2 grid for easy comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    colors = ['steelblue', 'mediumseagreen', 'darkorange', 'tomato']

    for idx, (name, sv) in enumerate(shap_results.items()):
        mean_shap  = np.abs(sv).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)[::-1].astype(int)[:10]

        axes[idx].barh(
            [feature_names[i] for i in sorted_idx],
            mean_shap[sorted_idx],
            color=colors[idx], alpha=0.85
        )
        axes[idx].set_title(f"{name}", fontweight='bold')
        axes[idx].set_xlabel("Mean |SHAP Value|")
        axes[idx].invert_yaxis()

    plt.suptitle("SHAP Feature Importance — All Models", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/shap_combined.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\nSHAP analysis complete. All plots saved to /outputs")
    return shap_results