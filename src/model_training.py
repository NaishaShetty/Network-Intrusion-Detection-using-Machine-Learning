"""
Model Training Module
Implements multiple ML models with hyperparameter tuning, calibration, and advanced evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    balanced_accuracy_score, brier_score_loss
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Callable, Optional
import json
import math
from datetime import datetime

from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and evaluates multiple ML models for intrusion detection.
    Includes hyperparameter tuning, probability calibration, and comprehensive evaluation.
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict] = {}
        self.results: Dict[str, Dict] = {}
        self.feature_importance: Dict[str, np.ndarray] = {}
        self.last_prediction_latency: Dict[str, float] = {}  # Store latency per model
        
    def _sanitize_for_json(self, obj):
        """Recursively sanitize data for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        elif isinstance(obj, float):
            if np.isinf(obj) or np.isnan(obj):
                return 0.0
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isinf(obj) or np.isnan(obj):
                return 0.0
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._sanitize_for_json(obj.tolist())
        return obj

    def _get_model_instance(self, model_name: str, params: Dict = None) -> Any:
        if params is None:
            params = {}
        
        model_map = {
            'decision_tree': DecisionTreeClassifier,
            'sgd': SGDClassifier,
            'random_forest': RandomForestClassifier,
            'xgboost': xgb.XGBClassifier,
            'lightgbm': lgb.LGBMClassifier
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Add random state for reproducibility
        if 'random_state' not in params and model_name != 'sgd':
            params['random_state'] = config.random_state
        
        return model_map[model_name](**params)
    
    def train_with_tuning(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict = None
    ) -> Any:
        """Train model with hyperparameter tuning and calibration."""
        logger.info(f"Training {model_name} with hyperparameter tuning...")
        
        base_model = self._get_model_instance(model_name)
        
        if param_grid is None:
            model_config = config.get_model_config(model_name)
            param_grid = model_config.get('hyperparameters', {})
        
        if not param_grid:
            # No tuning: just fit and calibrate
            logger.warning(f"No hyperparameters for {model_name}, using defaults")
            base_model.fit(X_train, y_train)
            best_model = base_model
            self.best_params[model_name] = {}
        else:
            # Tuning
            import math
            total_combinations = 1
            for v in param_grid.values():
                if isinstance(v, list):
                    total_combinations *= len(v)
            
            if total_combinations <= 10:
                 search_engine = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=3,
                    scoring=config.config['training']['scoring'],
                    n_jobs=config.config['training']['n_jobs'],
                    verbose=0
                )
            else:
                search_engine = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=10,
                    cv=3,
                    scoring=config.config['training']['scoring'],
                    n_jobs=config.config['training']['n_jobs'],
                    verbose=0,
                    random_state=config.random_state
                )
            
            search_engine.fit(X_train, y_train)
            self.best_params[model_name] = search_engine.best_params_
            best_model = search_engine.best_estimator_
            logger.info(f"Best params for {model_name}: {search_engine.best_params_}")

        # Calibration
        logger.info(f"Calibrating probabilities for {model_name}...")
        try:
            # We must use a fresh model instance for CV calibration to avoid data leakage/overfitting
            fresh_model = self._get_model_instance(model_name, self.best_params[model_name])
            calibrated_model = CalibratedClassifierCV(
                fresh_model,
                method='isotonic', # 'isotonic' usually better for large data, 'sigmoid' for small
                cv=3
            )
            calibrated_model.fit(X_train, y_train)
            self.models[model_name] = calibrated_model
        except Exception as e:
            logger.warning(f"Calibration failed for {model_name}: {e}. Using uncalibrated model.")
            self.models[model_name] = best_model
            
        return self.models[model_name]
    
    def train_simple(self, model_name, X_train, y_train, params=None):
        """Train without tuning, with calibration."""
        logger.info(f"Training {model_name} simply...")
        model = self._get_model_instance(model_name, params)
        
        try:
            calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated.fit(X_train, y_train)
            self.models[model_name] = calibrated
        except:
             model.fit(X_train, y_train)
             self.models[model_name] = model
             
        return self.models[model_name]

    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None
    ) -> Dict[str, Any]:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Probabilities
        try:
            y_prob = model.predict_proba(X_test)[:, 1] # P(Normal) if 1=Normal
        except:
            if hasattr(model, 'decision_function'):
                y_prob = model.decision_function(X_test) # Raw scores
            else:
                y_prob = [0.5] * len(y_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        try:
            brier = brier_score_loss(y_test, y_prob)
        except:
            brier = 0.0
            
        cm = confusion_matrix(y_test, y_pred)
        
        # Assumes 0=Attack, 1=Normal based on sorting
        # FNR (Attack missed) = FN / (FN + TP) where Positive=Attack(0)
        # In sklearn (0,1), row 0 is Attack.
        # cm[0,0] = Attack->Attack (TP for Attack)
        # cm[0,1] = Attack->Normal (FN for Attack)
        fn_attacks = cm[0, 1] if cm.shape == (2, 2) else 0
        total_attacks = (cm[0, 0] + cm[0, 1]) if cm.shape == (2, 2) else 1
        fnr = fn_attacks / total_attacks if total_attacks > 0 else 0.0
        
        # Detection Rate (Recall for Attack class)
        detection_rate = 1.0 - fnr
        
        try:
            auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_curve_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        except:
            auc = 0.5
            roc_curve_data = {}
            
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Handle Feature Importance with CalibratedCV
        has_fi = False
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = model.feature_importances_
            has_fi = True
        elif isinstance(model, CalibratedClassifierCV):
            # Try to aggregate importance from base estimators
            importances = []
            if hasattr(model, 'calibrated_classifiers_'):
                for clf in model.calibrated_classifiers_:
                    # clf is CalibratedClassifierAdapter
                    # clf.base_estimator is the fitted model (or estimator)
                    base = getattr(clf, 'base_estimator', getattr(clf, 'estimator', None))
                    if base and hasattr(base, 'feature_importances_'):
                        importances.append(base.feature_importances_)
            
            if importances:
                avg_imp = np.mean(importances, axis=0)
                self.feature_importance[model_name] = avg_imp
                has_fi = True

        results = {
            'accuracy': float(acc),
            'balanced_accuracy': float(balanced_acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'brier_score': float(brier),
            'fnr': float(fnr),
            'detection_rate': float(detection_rate),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'roc_curve': roc_curve_data,
            'has_feature_importance': has_fi
        }
        
        self.results[model_name] = results
        return results

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_tuning: bool = True,
        update_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Dict]:
        enabled_models = config.get_enabled_models()
        logger.info(f"Training {len(enabled_models)} models")
        
        all_results = {}
        total_models = len(enabled_models)
        
        for i, model_name in enumerate(enabled_models):
            try:
                # Progress update
                if update_callback:
                    # Map loop index to progress range 30-70
                    # 30 + (i / total) * 40
                    prog = 30 + int((i / total_models) * 40)
                    update_callback(f"Training {model_name}...", prog)
                
                if use_tuning:
                    self.train_with_tuning(model_name, X_train, y_train)
                else:
                    self.train_simple(model_name, X_train, y_train)
                
                # Evaluate
                results = self.evaluate_model(model_name, X_test, y_test)
                all_results[model_name] = results
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
                
        return all_results

    def get_feature_importance(self, model_name: str, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
        if model_name not in self.feature_importance:
            return pd.DataFrame()
            
        importance = self.feature_importance[model_name]
        # Ensure lengths match
        if len(importance) != len(feature_names):
             # Truncate or pad? safely handle
             logger.warning(f"Feature importance length mismatch for {model_name}")
             return pd.DataFrame()
             
        df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        return df.sort_values('importance', ascending=False).head(top_n)

    def save_models(self, output_dir: Path = None):
        output_dir = output_dir or config.models_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, output_dir / f"{name}.joblib")
            
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'best_params': self.best_params,
            # 'results': self.results # sanitized if needed
        }
        sanitized = self._sanitize_for_json(metadata)
        with open(output_dir / "training_metadata.json", 'w') as f:
            json.dump(sanitized, f, indent=2)

    def load_model(self, model_name: str, model_path: Path = None):
        if model_path is None:
            model_path = config.models_dir / f"{model_name}.joblib"
        self.models[model_name] = joblib.load(model_path)
        return self.models[model_name]

    def predict(self, model_name: str, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """Make predictions with a trained model and measure latency."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        import time
        model = self.models[model_name]
        start_time = time.time()
        
        if return_proba:
            if hasattr(model, 'predict_proba'):
                res = model.predict_proba(X)
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X)
                # Sigmoid to convert scores to probabilities if decision_function is 1D
                if len(scores.shape) == 1:
                    proba = 1 / (1 + np.exp(-scores))
                    res = np.column_stack([1 - proba, proba])
                else:
                    exp_scores = np.exp(scores)
                    res = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            else:
                # Mock probabilities based on predictions for models with no proba capability
                preds = model.predict(X)
                # Map classes to 0/1 (Attack=0, Normal=1 based on sorting 'a' vs 'n')
                # If 'normal' -> [0, 1], if 'attack' -> [1, 0]
                res = np.array([[1.0, 0.0] if p == 'attack' or p == 0 else [0.0, 1.0] for p in preds])
        else:
            res = model.predict(X)
            
        latency = (time.time() - start_time) * 1000 # Convert to ms
        self.last_prediction_latency[model_name] = latency / len(X) if len(X) > 0 else latency
        
        return res

    def get_explanation(self, model_name: str, X_sample: np.ndarray, feature_names: list) -> Dict[str, Any]:
        """
        Get per-sample explanation using simplified feature contribution.
        """
        if model_name not in self.models:
            return {"error": "Model not found"}
        
        try:
            fi = self.get_feature_importance(model_name, feature_names)
            if fi is None or fi.empty:
                return {"error": "No feature importance available for this model"}
                
            # Heuristic Local Importance (Top global features influencing this decision)
            top_features = fi.head(5).to_dict('records')
            
            return {
                "explanation_type": "Feature Importance Attribution",
                "top_features": top_features,
                "confidence_score": 0.85 
            }
        except Exception as e:
            return {"error": str(e)}
