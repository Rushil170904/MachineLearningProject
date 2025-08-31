import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("combined_features.csv")  # replace with your combined CSV

# 2. Drop IP addresses
df = df.drop(columns=["src_ip", "dst_ip","protocol"])

# 3. Handle categorical columns

# 4. Split features and labels
X = df.drop(columns=["label"])
y = df["label"].map({"NonVPN":0,"VPN":1})  # can stay as "VPN"/"NonVPN"

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Train XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train, y_train)

# 7. Predictions
y_pred = xgb.predict(X_test)

# 8. Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Feature importance
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop Features:\n", importances.head(10))

# Optional: Plot top 10 features
plt.figure(figsize=(10,6))
plt.barh(importances['feature'][:10][::-1], importances['importance'][:10][::-1])
plt.xlabel("Importance")
plt.title("Top 10 Features - XGBoost")
plt.show()
