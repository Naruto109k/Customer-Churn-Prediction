import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import os

os.makedirs("outputs", exist_ok=True)


def lime_explain_instance(model, X_train, X_test, feature_names,
                           model_name, instance_idx=0):
    """
    Generates a LIME explanation for a single employee's prediction.

    LIME works by creating a small perturbed dataset around the instance,
    fitting a simple linear model to it, and reading off the feature weights.
    Those weights tell us which features pushed the prediction toward
    'Leave' (positive) or 'Stay' (negative) for this specific person.

    Unlike SHAP, LIME is local — the explanation only holds in the
    neighbourhood of this one instance, not across the whole dataset.
    """
    print(f"\nLIME explanation for {model_name}...")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data = np.array(X_train),
        feature_names = feature_names,
        class_names   = ['Stay', 'Leave'],
        mode          = 'classification',
        random_state  = 42
    )

    instance    = np.array(X_test)[instance_idx]
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10
    )

    pred_proba = model.predict_proba(instance.reshape(1, -1))[0]
    pred_class = "Leave" if pred_proba[1] >= 0.5 else "Stay"

    print(f"   Employee #{instance_idx} — predicted: {pred_class}")
    print(f"   Stay : {pred_proba[0]*100:.1f}%  |  Leave: {pred_proba[1]*100:.1f}%")
    print(f"\n   Feature contributions:")
    for feat, weight in explanation.as_list():
        direction = "increases" if weight > 0 else "decreases"
        print(f"   {feat:<45} {direction} churn risk  ({weight:+.4f})")

    fig = explanation.as_pyplot_figure()
    plt.title(f"{model_name} — LIME Explanation (Employee #{instance_idx})",
              fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"outputs/lime_{model_name.replace(' ', '_')}_employee{instance_idx}.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    return explanation


def lime_all_models(models_dict, X_train_raw, X_test_raw,
                    X_train_scaled, X_test_scaled,
                    feature_names, instance_idx=0):
    """
    Runs LIME on the same employee across all four models.

    Explaining the same instance with all models lets us check whether they
    agree on which features matter — if they do, that's a stronger signal
    than any single model's explanation alone.

    SVM gets the scaled versions of both train and test data since that's
    what it was trained on. Tree models get raw unscaled data.
    """
    print("\n" + "="*50)
    print("   LIME EXPLAINABILITY — ALL MODELS")
    print("="*50)
    print(f"   Explaining Employee #{instance_idx} across all models\n")

    for name, model in models_dict.items():
        if name == "SVM":
            lime_explain_instance(model, X_train_scaled, X_test_scaled,
                                  feature_names, name, instance_idx)
        else:
            lime_explain_instance(model, X_train_raw, X_test_raw,
                                  feature_names, name, instance_idx)

    print("\nLIME analysis complete. All plots saved to /outputs")