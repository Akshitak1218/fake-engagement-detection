import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# 2. Basic Cleaning
train_df.drop_duplicates(inplace=True)
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# 3. Feature & Target Split
TARGET = "fake"

X = train_df.drop(TARGET, axis=1)
y = train_df[TARGET]

# Remove target column from test set if present
if "fake" in test_df.columns:
    X_test_final = test_df.drop("fake", axis=1)
else:
    X_test_final = test_df.copy()

# 4. Train–Validation Split (Proper Evaluation)
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Feature Scaling (For Logistic Regression)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

# 6. Logistic Regression (Baseline)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_val_scaled)

print("\nLogistic Regression (Validation Results):")
print(classification_report(y_val, y_pred_lr))

# 7. Random Forest (Final Model)
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_val)

print("\nRandom Forest (Validation Results):")
print(classification_report(y_val, y_pred_rf))

# 8. Confusion Matrix 
cm = confusion_matrix(y_val, y_pred_rf)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest (Validation)")
plt.show()

# 9. Feature Importance (From Random Forest)
importances = rf.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feat_df)

# 10. Final Model Training on Full Dataset
rf_final = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf_final.fit(X, y)

test_predictions = rf_final.predict(X_test_final)

test_df["predicted_fake"] = test_predictions
test_df.to_csv("test_predictions.csv", index=False)

print("\nPredictions saved as test_predictions.csv")