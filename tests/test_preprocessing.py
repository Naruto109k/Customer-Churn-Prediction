from utils.preprocessing import load_and_preprocess

X, X_scaled, y, feature_names, scaler = load_and_preprocess("data/attrition.csv")

print(f"\n🎯 Final shapes:")
print(f"   X: {X.shape}")
print(f"   y: {y.shape}")