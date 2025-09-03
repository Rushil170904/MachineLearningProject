import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# classifiers we want to test
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# load data
data = pd.read_csv("combined_features.csv", dtype=str, low_memory=False)
data.dropna(how="all", inplace=True)

# split into features and label
X = data.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
y = data.iloc[:, -1]

# handle missing + encode labels
X = SimpleImputer(strategy="most_frequent").fit_transform(X)
y = LabelEncoder().fit_transform(y.astype(str))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# models with smaller configs so it runs faster
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=10),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(),
    "AdaBoost": AdaBoostClassifier(n_estimators=50),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, eval_metric='mlogloss', use_label_encoder=False),
    "CatBoost": CatBoostClassifier(iterations=100, depth=5, verbose=0),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,), max_iter=100)
}

# train and check accuracy for each
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc})

    print(f"===== {name} =====")
    print(classification_report(y_test, y_pred))

# final table of accuracies
results_df = pd.DataFrame(results)
print("\n  Model Comparison:\n", results_df)