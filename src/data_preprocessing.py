"""
Data Preprocessing Module
Handles data loading, cleaning, encoding, and feature scaling for network intrusion detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import joblib
import logging
from pathlib import Path

from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses network traffic data for intrusion detection.
    
    Handles:
    - Loading raw data
    - Assigning column names
    - Encoding categorical features
    - Binary classification (Normal vs Attack)
    - Feature scaling
    - Train-test splitting
    """
    
    # KDD Cup 1999 column names
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
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder = LabelEncoder()
        self.feature_names = None
        self.baseline_stats = None  # To store training distribution for drift detection
        
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load network traffic data from file.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            # KDD dataset has no header
            df = pd.read_csv(file_path, header=None)
            
            # Handle 43 columns (NSL-KDD format: 41 features + 1 label + 1 difficulty)
            if len(df.columns) == 43:
                df = df.iloc[:, :42] # Keep first 42 columns
                df.columns = self.COLUMN_NAMES
            elif len(df.columns) == 42:
                df.columns = self.COLUMN_NAMES
            else:
                # Try to assign as many names as possible
                df.columns = self.COLUMN_NAMES[:len(df.columns)]
                
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        # Dynamically identify object columns if fitting, or use known cat columns
        if fit:
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            # Filter out label if present
            if 'label' in object_cols:
                object_cols.remove('label')
            
            # Update known categorical columns
            self.CATEGORICAL_COLUMNS = object_cols
        
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unknown categories by mapping to 'unknown' or robust transform
                        # Simpler robust approach: use map and fillna with extensive category
                        le = self.label_encoders[col]
                        
                        # Add 'unknown' to classes if not present
                        if 'known_unknown' not in le.classes_:
                             # This is tricky with sklearn LabelEncoder. 
                             # For simplicity, we'll strip unknown values during inference if critical
                             # or keep as strings which will crash scaling.
                             # Better approach: string match
                             pass
                        
                        # Safe transform
                        df[col] = df[col].astype(str).map(
                            lambda s: s if s in le.classes_ else le.classes_[0] 
                        )
                        df[col] = le.transform(df[col])
        
        return df
    
    def encode_labels(self, labels: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Encode target labels (Normal vs Attack).
        
        Args:
            labels: Series of labels
            fit: Whether to fit encoder
            
        Returns:
            Encoded labels as numpy array
        """
        # Convert to binary: normal vs attack
        binary_labels = labels.apply(
            lambda x: "normal" if str(x).strip().lower() in ["normal", "normal."] else "attack"
        )
        
        if fit:
            return self.target_encoder.fit_transform(binary_labels)
        else:
            return self.target_encoder.transform(binary_labels)
    
    def get_label_name(self, encoded_label: int) -> str:
        """Get original label name from encoded value"""
        return self.target_encoder.inverse_transform([encoded_label])[0]
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers
            
        Returns:
            Tuple of (features, labels)
        """
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Separate features and labels
        X = df.drop("label", axis=1)
        y = self.encode_labels(df["label"], fit=fit)
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
        
        # Scale features
        if fit:
            self.baseline_stats = {
                'mean': X.mean().to_dict(),
                'std': X.std().to_dict(),
                'count': len(df)
            }
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y

    def calculate_drift_score(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate concept drift score by comparing current data to baseline.
        Uses a simplified Population Stability Index (PSI) like comparison.
        """
        if self.baseline_stats is None:
            return {"status": "error", "message": "No baseline stats. Train models first."}
        
        try:
            # Preprocess current data without fitting
            # We only look at features present in baseline
            # For simplicity, we compare means of top features
            X_curr = current_df.copy()
            # Minimal internal encoding for numeric compatibility
            for col, le in self.label_encoders.items():
                if col in X_curr.columns:
                    X_curr[col] = X_curr[col].astype(str).map(
                        lambda s: s if s in le.classes_ else le.classes_[0]
                    )
                    X_curr[col] = le.transform(X_curr[col])
            
            # Numeric columns only
            X_curr = X_curr.select_dtypes(include=[np.number])
            
            drift_details = {}
            total_drift = 0
            count = 0
            
            for col in self.baseline_stats['mean'].keys():
                if col in X_curr.columns:
                    b_mean = self.baseline_stats['mean'][col]
                    b_std = self.baseline_stats['std'][col]
                    c_mean = X_curr[col].mean()
                    
                    # Normalize difference by baseline std
                    if b_std > 0:
                        diff = abs(c_mean - b_mean) / b_std
                        drift_details[col] = float(diff)
                        total_drift += diff
                        count += 1
            
            avg_drift = total_drift / count if count > 0 else 0
            
            return {
                "avg_drift": float(avg_drift),
                "is_drift_detected": bool(avg_drift > 0.5), # Heuristic threshold
                "feature_drift": drift_details,
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def prepare_train_test_split(
        self, 
        file_path: Path = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data and create train-test split.
        
        Args:
            file_path: Path to dataset (uses config default if None)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if file_path is None:
            file_path = config.dataset_path
        
        # Load data
        df = self.load_data(file_path)
        
        # Preprocess
        X, y = self.preprocess(df, fit=True)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.test_size, 
            random_state=config.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Attack ratio in train: {y_train.sum() / len(y_train):.2%}")
        logger.info(f"Attack ratio in test: {y_test.sum() / len(y_test):.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, path: Path):
        """Save preprocessor artifacts"""
        artifacts = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(artifacts, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: Path):
        """Load preprocessor artifacts"""
        artifacts = joblib.load(path)
        self.scaler = artifacts['scaler']
        self.label_encoders = artifacts['label_encoders']
        self.target_encoder = artifacts['target_encoder']
        self.feature_names = artifacts['feature_names']
        logger.info(f"Preprocessor loaded from {path}")
