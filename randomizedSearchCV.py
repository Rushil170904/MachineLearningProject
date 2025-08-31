import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform

# 1. Load dataset
df = pd.read_csv("combined_features.csv")
df = df.drop(columns=["src_ip", "dst_ip"])

# 2. Encode categorical features
for col in df.select_dtypes(include=["object"]).columns:
    if col != "label":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

y = df["label"].map({"NonVPN": 0, "VPN": 1})
X = df.drop(columns=["label"])

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Define AdaBoost model
base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(estimator=base_estimator, random_state=42)

# 5. Define parameter distributions
param_dist = {
    "n_estimators": randint(50, 500),      # number of weak learners
    "learning_rate": uniform(0.01, 1.0),   # shrinkage factor
    "estimator__max_depth": randint(1, 5)  # depth of weak learners
}

# 6. RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,             # number of random combinations to try
    scoring="accuracy",
    cv=5,                  # 5-fold cross validation
    random_state=42,
    n_jobs=-1,             # parallel processing
    verbose=2
)

random_search.fit(X_train, y_train)

# 7. Best model
print("Best Params:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

# 8. Evaluate on test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("✅ Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
