from utils.preprocessing import load_and_preprocess
from models.decision_tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, X_scaled, y, feature_names, scaler = load_and_preprocess("data/attrition.csv")
X, y = X.values, y.values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train our Decision Tree
print("⏳ Training Decision Tree from scratch...")
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=2)
dt.fit(X_train, y_train)

# Evaluate
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Decision Tree Accuracy: {acc*100:.2f}%")