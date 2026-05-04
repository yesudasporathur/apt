
# PRE PROCESSING DIRECTIVES
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns


#  LOAD DATA
input_path = "C:/apt/input.csv"
output_path = "C:/apt/output/"

os.makedirs(output_path, exist_ok=True)

df = pd.read_csv(input_path)

# Validate label
if 'Label' not in df.columns:
    raise ValueError("'Label' column not found in dataset")


#  PREPARE FEATURES
def prepare_features(df):
    print("\nPreparing features...")

    # Keep only numeric
    X = df.select_dtypes(include=['float64', 'int64'])

    # Remove label if present
    if 'Label' in X.columns:
        X = X.drop(columns=['Label'])

    y = df['Label']

    # Handle missing values
    X = X.fillna(X.mean())

    print("Feature shape:", X.shape)
    return X, y


X, y = prepare_features(df)


#  TRAIN-TEST SPLIT (BEFORE FEATURE SELECTION!)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#  REMOVE CORRELATED FEATURES (ONLY USING TRAIN DATA)
def remove_correlated_features(X, threshold=0.9):
    print("\nRemoving highly correlated features...")

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"Dropped {len(to_drop)} correlated features")
    print("Dropped columns:", to_drop)

    return X.drop(columns=to_drop), to_drop


X_train, dropped_cols = remove_correlated_features(X_train)

# Apply same columns to test set
X_test = X_test.drop(columns=dropped_cols, errors='ignore')



#  FEATURE SCALING (needed for Isolation Forest)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# RANDOM FOREST MODEL
print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_scores_rf = rf.predict_proba(X_test)[:, 1]


# ISOLATION FOREST MODEL
print("\nTraining Isolation Forest...")

iso = IsolationForest(
    n_estimators=100,
    contamination='auto',  # safer than fixed 0.1
    random_state=42
)

iso.fit(X_train_scaled)

# anomaly score (higher = more abnormal)
y_scores_iso = -iso.decision_function(X_test_scaled)

# convert to binary using threshold (optional)
threshold = np.percentile(y_scores_iso, 90)
y_pred_iso = (y_scores_iso >= threshold).astype(int)


#  EVALUATION FUNCTION
def evaluate_model(name, y_true, y_pred, y_scores):
    print(f"\n===== {name} =====")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report (5 decimal places):")
    print(classification_report(y_true, y_pred, digits=5))

    print(f"ROC-AUC: {roc_auc_score(y_true, y_scores):.5f}")

evaluate_model("Random Forest", y_test, y_pred_rf, y_scores_rf)
evaluate_model("Isolation Forest", y_test, y_pred_iso, y_scores_iso)


# VISUALIZATION
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=[0, 1],
        yticklabels=[0, 1]
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(output_path + title + ".png")
    plt.show()


plot_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")
plot_confusion_matrix(y_test, y_pred_iso, "Isolation Forest Confusion Matrix")


# ROC CURVE
def plot_roc(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.5f}")
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()

    plt.savefig(output_path + title + ".png")
    plt.show()


plot_roc(y_test, y_scores_rf, "Random Forest ROC Curve")
plot_roc(y_test, y_scores_iso, "Isolation Forest ROC Curve")



# FIGURES

from sklearn.metrics import precision_recall_curve

def plot_pr(y_true, y_scores, title):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.savefig(output_path + title + ".png",dpi=300)

    plt.show()

plot_pr(y_test, y_scores_rf, "Precision-Recall Curve Random Forest")
plot_pr(y_test, y_scores_iso, "Precision-Recall Curve Isolation Forest")


#########
df['Label'].value_counts().plot(kind='bar')
title="Class Distribution (Normal vs Attack)"
plt.title(title)
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig(output_path + title + ".png")

plt.show()
##########
results = {
    "Model": ["Random\nForest", "Isolation\nForest"],
    "Accuracy": [
        rf.score(X_test, y_test),
        (y_pred_iso == y_test).mean()
    ]
}

df_results = pd.DataFrame(results)


df_results.plot(x="Model", y="Accuracy", kind="bar", legend=False, figsize=(9,12))

title = "Model Comparison"
plt.title(title)
plt.ylabel("Accuracy")

plt.savefig(output_path + title + ".png")
plt.show()
##########
print("\nPipeline completed successfully!")
