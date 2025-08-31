import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Load combined dataset
df = pd.read_csv("combined_features.csv")

# 2. Drop flow identifiers that can cause overfitting
df = df.drop(columns=["src_ip", "dst_ip"])

# 3. Encode categorical features (e.g., protocol, service type, etc.)
for col in df.select_dtypes(include=["object"]).columns:
    if col != "label":  # don't encode the target yet
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 4. Encode target column (VPN=1, NonVPN=0)
y = df["label"].map({"NonVPN": 0, "VPN": 1})
X = df.drop(columns=["label"])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 7. Predictions
y_pred = rf.predict(X_test)

# 8. Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Feature importance
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop Features:\n", importances.head(10))
