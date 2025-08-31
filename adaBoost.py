import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Load combined dataset
df = pd.read_csv("combined_features.csv")

# 2. Drop flow identifiers that cause overfitting
df = df.drop(columns=["src_ip", "dst_ip"])

# 3. Encode categorical features (since AdaBoost cannot handle strings directly)
for col in df.select_dtypes(include=["object"]).columns:
    if col != "label":   # keep target separate
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# 4. Encode target
y = df["label"].map({"NonVPN": 0, "VPN": 1})
X = df.drop(columns=["label"])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Train AdaBoost (using DecisionTree as weak learner)
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # decision stumps
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)

model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
