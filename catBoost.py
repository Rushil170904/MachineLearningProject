import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

# 1. Load combined dataset
df = pd.read_csv("combined_features.csv")   # replace with your filename

# 2. Drop flow identifiers that can cause overfitting
df = df.drop(columns=["src_ip", "dst_ip"])  

# 3. Split features and label
X = df.drop(columns=["label"])   
y = df["label"].map({"NonVPN": 0, "VPN": 1})   # encode target

# 4. Identify categorical features (string/object dtypes)
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
print("Categorical Features:", cat_features)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Train CatBoost (handles categorical data automatically)
model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100
)

model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Feature Importance
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.get_feature_importance()
}).sort_values(by="importance", ascending=False)

print("\nTop Features:\n", importances.head(10))
