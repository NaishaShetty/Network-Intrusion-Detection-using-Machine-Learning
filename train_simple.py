"""
Simplified Training Script - Direct Implementation
This script runs the complete training pipeline without module imports
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Network Intrusion Detection System - Training Pipeline")
print("="*80)

# Configuration
DATASET_PATH = "KDDTrain+.txt"
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Column names
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

print("\n[1/5] Loading and preprocessing data...")
# Load data without header first to infer columns
df = pd.read_csv(DATASET_PATH, header=None)
print(f"  ✓ Loaded {len(df)} records with {len(df.columns)} columns")

# Handle 43 columns (NSL-KDD format: 41 features + 1 label + 1 difficulty)
if len(df.columns) == 43:
    print("  ✓ Detected NSL-KDD format (43 columns). Dropping difficulty column.")
    df = df.iloc[:, :42] # Keep first 42 columns
    df.columns = COLUMN_NAMES
elif len(df.columns) == 42:
    print("  ✓ Detected standard KDD format (42 columns).")
    df.columns = COLUMN_NAMES
else:
    print(f"  WARNING: Unexpected column count: {len(df.columns)}")
    # Try to assign as many names as possible
    df.columns = COLUMN_NAMES[:len(df.columns)]

# Check label column
print(f"DEBUG: Unique labels found: {df['label'].unique()[:10]}")

# Separate features and labels FIRST
X = df.drop("label", axis=1).copy()
y_labels = df["label"].copy()

print("\nDEBUG: X columns:", X.columns.tolist())

# Dynamically encode ALL object/string columns
object_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"DEBUG: Found object columns to encode: {object_cols}")

label_encoders = {}
for col in object_cols:
    # print(f"  Encoding {col}...") # Reduce verbosity
    label_encoders[col] = LabelEncoder()
    # Ensure values are strings before encoding
    X[col] = label_encoders[col].fit_transform(X[col].astype(str))

print(f"  ✓ Encoded {len(object_cols)} categorical features")

# Verify no objects remain
remaining_objects = X.select_dtypes(include=['object']).columns.tolist()
if remaining_objects:
    print(f"ERROR: Still have object columns: {remaining_objects}")
    for col in remaining_objects:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

# Binary classification for labels
# DEBUG: Print label distribution before mapping
print("DEBUG: Label value counts (top 10):")
print(y_labels.value_counts().head(10))

y_labels = y_labels.apply(
    lambda x: "normal" if str(x).strip().lower() in ["normal", "normal."] else "attack"
)

# DEBUG: Print binary distribution
print("DEBUG: Binary label distribution:")
print(y_labels.value_counts())

if len(y_labels.unique()) < 2:
    print("CRITICAL ERROR: Only one class detected! Training will fail.")
    print(f"Unique classes: {y_labels.unique()}")
    # Fallback/Debug hack: if all are normal, force one to attack to prevent crash (for debugging)
    # But ideally we fix the mapping.
    # Check if 'normal' matching is too broad or too narrow.
    
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_labels)
print(f"  ✓ Converted to binary classification")

# Features
feature_names = X.columns.tolist()

# DEBUG: Check columns and types before scaling
print("\nDEBUG: X columns before scaling:")
print(X.columns.tolist())
print("\nDEBUG: X dtypes before scaling:")
print(X.dtypes)
print("\nDEBUG: protocol_type head:")
if "protocol_type" in X.columns:
    print(X["protocol_type"].head())
    print(f"Unique values in protocol_type: {X['protocol_type'].unique()[:5]}")
else:
    print("WARNING: protocol_type NOT found in X columns!")

# Scale features
scaler = StandardScaler()
try:
    X_scaled = scaler.fit_transform(X)
except ValueError as e:
    print(f"\nERROR during scaling: {e}")
    # Print the specific column that caused the error
    for col in X.columns:
        try:
            X[col].astype(float)
        except ValueError:
            print(f"FAILED to convert column '{col}' to float. First 5 values: {X[col].head().tolist()}")
    raise
print(f"  ✓ Scaled features")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
print(f"  ✓ Attack ratio: {y_train.sum() / len(y_train):.2%}")

# Create directories
Path("models/trained").mkdir(parents=True, exist_ok=True)
Path("outputs/plots").mkdir(parents=True, exist_ok=True)

print("\n[2/5] Training models...")
results = {}

# Decision Tree
print("  Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
results['decision_tree'] = {
    'accuracy': accuracy_score(y_test, dt_pred),
    'precision': precision_score(y_test, dt_pred),
    'recall': recall_score(y_test, dt_pred),
    'f1_score': f1_score(y_test, dt_pred)
}
joblib.dump(dt, "models/trained/decision_tree.joblib")
print(f"    ✓ Accuracy: {results['decision_tree']['accuracy']:.4f}")

# SGD Classifier
print("  Training SGD Classifier...")
sgd = SGDClassifier(max_iter=1000, random_state=RANDOM_STATE)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)
results['sgd'] = {
    'accuracy': accuracy_score(y_test, sgd_pred),
    'precision': precision_score(y_test, sgd_pred),
    'recall': recall_score(y_test, sgd_pred),
    'f1_score': f1_score(y_test, sgd_pred)
}
joblib.dump(sgd, "models/trained/sgd.joblib")
print(f"    ✓ Accuracy: {results['sgd']['accuracy']:.4f}")

# Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results['random_forest'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'f1_score': f1_score(y_test, rf_pred)
}
joblib.dump(rf, "models/trained/random_forest.joblib")
print(f"    ✓ Accuracy: {results['random_forest']['accuracy']:.4f}")

# XGBoost
print("  Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
results['xgboost'] = {
    'accuracy': accuracy_score(y_test, xgb_pred),
    'precision': precision_score(y_test, xgb_pred),
    'recall': recall_score(y_test, xgb_pred),
    'f1_score': f1_score(y_test, xgb_pred)
}
joblib.dump(xgb_model, "models/trained/xgboost.joblib")
print(f"    ✓ Accuracy: {results['xgboost']['accuracy']:.4f}")

# LightGBM
print("  Training LightGBM...")
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
results['lightgbm'] = {
    'accuracy': accuracy_score(y_test, lgb_pred),
    'precision': precision_score(y_test, lgb_pred),
    'recall': recall_score(y_test, lgb_pred),
    'f1_score': f1_score(y_test, lgb_pred)
}
joblib.dump(lgb_model, "models/trained/lightgbm.joblib")
print(f"    ✓ Accuracy: {results['lightgbm']['accuracy']:.4f}")

print("\n[3/5] Saving preprocessor...")
preprocessor_artifacts = {
    'scaler': scaler,
    'label_encoders': label_encoders,
    'target_encoder': target_encoder,
    'feature_names': feature_names
}
joblib.dump(preprocessor_artifacts, "models/trained/preprocessor.joblib")
print("  ✓ Preprocessor saved")

print("\n[4/5] Generating dashboard data...")
import json
dashboard_data = {
    'models': {},
    'comparison': {
        'model_names': list(results.keys()),
        'accuracy': [results[m]['accuracy'] for m in results.keys()],
        'precision': [results[m]['precision'] for m in results.keys()],
        'recall': [results[m]['recall'] for m in results.keys()],
        'f1_score': [results[m]['f1_score'] for m in results.keys()]
    }
}

for model_name in results.keys():
    dashboard_data['models'][model_name] = {
        'metrics': results[model_name]
    }

with open("outputs/plots/dashboard_data.json", 'w') as f:
    json.dump(dashboard_data, f, indent=2)
print("  ✓ Dashboard data saved")

print("\n[5/5] Training Summary")
print("="*80)
for model_name, metrics in results.items():
    print(f"\n{model_name.upper()}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

print("\n" + "="*80)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nModels saved to: models/trained/")
print(f"Dashboard data saved to: outputs/plots/")
print(f"\nNext steps:")
print(f"  1. Start API: python -m uvicorn src.api:app --reload")
print(f"  2. Start Frontend: cd frontend && npm start")
