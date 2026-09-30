#Credit Card Fraud Prediction w/ StratifiedKFold Cross-Validation
#Tommy Tang
#Sept 30 2026
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC


# ============================================================
# 1. FEATURES / TARGET
# ============================================================

X = df.drop(columns="target")
y = df["target"]


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# ============================================================
# 3. IDENTIFY COLUMN TYPES
# ============================================================

numeric_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

bool_cols = X_train.select_dtypes(
    include="bool"
).columns.tolist()

# Treat bools as categorical for SMOTENC
categorical_cols += bool_cols


# ============================================================
# 4. PRE-SMOTE TRANSFORMATION
# ============================================================
# Convert categorical variables to integer category codes.
# Numeric variables pass through unchanged.

pre_smote = ColumnTransformer([
    (
        "num",
        "passthrough",
        numeric_cols
    ),
    (
        "cat",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ),
        categorical_cols
    )
])


# ============================================================
# 5. SMOTENC
# ============================================================
# ColumnTransformer outputs numeric columns first,
# followed by categorical columns.
#
# Therefore:
#
# [numeric ... | categorical ...]
#
# categorical indices begin after numeric_cols.

categorical_indices = list(
    range(
        len(numeric_cols),
        len(numeric_cols) + len(categorical_cols)
    )
)

smote = SMOTENC(
    categorical_features=categorical_indices,
    random_state=42
)


# ============================================================
# 6. POST-SMOTE PREPROCESSING
# ============================================================
# After SMOTENC:
#
# numerical columns -> StandardScaler
# categorical columns -> OneHotEncoder

numeric_indices = list(range(len(numeric_cols)))

postprocessor = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        numeric_indices
    ),
    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        categorical_indices
    )
])


# ============================================================
# 7. MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42
    )
}


# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ============================================================
# 9. TEST MODELS WITH CROSS-VALIDATION
# ============================================================

results = []
pipelines = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("pre_smote", pre_smote),
        ("smote", smote),
        ("postprocessor", postprocessor),
        ("model", model)
    ])

    pipelines[name] = pipeline

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean()
    })


# ============================================================
# 10. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print(results_df)


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_pipeline = pipelines[best_model_name]

print(f"\nSelected model: {best_model_name}")


# ============================================================
# 12. FIT ON ALL TRAINING DATA
# ============================================================

best_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\nFinal Test Results")
print("------------------")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(y_test, y_pred))

print("Recall:",
      recall_score(y_test, y_pred))

print("F1:",
      f1_score(y_test, y_pred))

print("ROC-AUC:",
      roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))