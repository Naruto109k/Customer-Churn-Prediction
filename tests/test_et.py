from utils.preprocessing import load_and_preprocess
from models.extra_trees import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time

X, X_scaled, y, feature_names, scaler = load_and_preprocess("data/attrition.csv")
X, y = X.values, y.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("⏳ Training Extra Trees from scratch...")
start = time.time()
et = ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42)
et.fit(X_train, y_train)
elapsed = round(time.time() - start, 2)

y_pred = et.predict(X_test)
y_prob = et.predict_proba(X_test)[:, 1]

print(f"\n✅ Extra Trees Results:")
print(f"   Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"   F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"   AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"   Time     : {elapsed}s")