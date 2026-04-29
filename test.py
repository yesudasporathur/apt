# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:/apt/input.csv")

# ==============================
# 5. FEATURE SELECTION
# ==============================
def prepare_features(df):
    print("\nPreparing features...")
    
    # Keep only numeric columns
    X = df.select_dtypes(include=['float64', 'int64'])
    
    # Remove label if included
    if 'Label' in X.columns:
        X = X.drop(columns=['Label'])
    
    y = df['Label']
    
    print("Feature shape:", X.shape)
    
    return X, y


X, y = prepare_features(df)


# ==============================
# 6. REMOVE HIGHLY CORRELATED FEATURES
# ==============================
def remove_correlated_features(X, threshold=0.9):
    print("\nRemoving highly correlated features...")
    
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    X = X.drop(columns=to_drop)
    
    print(f"Dropped {len(to_drop)} correlated features")
    print("New shape:", X.shape)
    
    return X


X = remove_correlated_features(X)


# ==============================
# 7. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================
# 8. FEATURE SCALING
# ==============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# 9. RANDOM FOREST MODEL
# ==============================
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


# ==============================
# 10. ISOLATION FOREST MODEL
# ==============================
print("\nTraining Isolation Forest...")

iso = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

iso.fit(X_train)

y_pred_iso = iso.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)


# ==============================
# 11. EVALUATION FUNCTION
# ==============================
def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("ROC-AUC:", roc_auc_score(y_true, y_pred))


evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("Isolation Forest", y_test, y_pred_iso)
#random forest visual


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("C:/apt/output/"+title+".png")
    plt.show()

plot_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")
plot_confusion_matrix(y_test, y_pred_iso, "Isolation Forest Confusion Matrix")
#random forest visual###


#roc curve

def plot_roc(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.savefig("C:/apt/output/"+title+".png")
    plt.show()

# Random Forest probabilities
y_scores_rf = rf.predict_proba(X_test)[:, 1]

plot_roc(y_test, y_scores_rf, "Random Forest ROC Curve")
#roc curve###


# ==============================
# 12. OPTIONAL: SAVE CLEAN DATASET
# =========================
print("\n✅ Pipeline completed successfully!")

