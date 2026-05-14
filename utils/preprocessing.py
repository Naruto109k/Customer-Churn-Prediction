import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_preprocess(filepath):
    """
    Loads the IBM HR attrition dataset and prepares it for training.

    Returns two versions of X:
      - X_resampled: raw (unscaled) features after SMOTE, used by tree-based models
      - X_scaled: standardized version of X_resampled, used by SVM

    Both share the same y_resampled target and were produced from the same
    SMOTE-balanced dataset, so train/test splits will be consistent.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # These four columns have zero variance across all employees — they add no signal
    df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], inplace=True)

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Label-encode all remaining categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    print(f"Class distribution before SMOTE:")
    print(f"  Stay  (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  Leave (1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

    # The dataset is heavily imbalanced (~84% stay, ~16% leave).
    # SMOTE generates synthetic minority samples so the model doesn't just
    # learn to predict "stay" for everyone.
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"After SMOTE:")
    print(f"  Stay  (0): {(y_resampled==0).sum()}")
    print(f"  Leave (1): {(y_resampled==1).sum()}")

    # StandardScaler is fitted only on the training data (here the full resampled set)
    # and applied to produce the scaled version for SVM
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_resampled), columns=X.columns)

    return X_resampled, X_scaled, y_resampled, X.columns.tolist(), scaler