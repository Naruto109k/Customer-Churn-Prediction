from utils.preprocessing import load_and_preprocess
from models.random_forest import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time

# Load data
X, X_scaled, y, feature_names, scaler = load_and_preprocess("data/attrition.csv")
X, y = X.values, y.values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train our Random Forest
print("⏳ Training Random Forest from scratch...")
start = time.time()
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
elapsed = round(time.time() - start, 2)

# Evaluate
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"\n✅ Random Forest Results:")
print(f"   Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"   F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"   AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"   Time     : {elapsed}s")