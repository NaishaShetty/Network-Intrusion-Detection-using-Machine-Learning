"""
Visualization Module
Creates professional visualizations for model performance and data insights
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

from .config import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Visualizer:
    """Creates visualizations for intrusion detection analysis"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or config.plots_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        model_name: str,
        labels: List[str] = None,
        save: bool = True
    ) -> str:
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            labels: Class labels
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        if labels is None:
            labels = ['Normal', 'Attack']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f"{model_name}_confusion_matrix.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(output_path)
        
        return None
    
    def plot_roc_curve(
        self,
        results: Dict[str, Dict],
        save: bool = True
    ) -> str:
        """
        Plot ROC curves for all models.
        
        Args:
            results: Dictionary of model results
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            if 'roc_curve' in result:
                fpr = result['roc_curve']['fpr']
                tpr = result['roc_curve']['tpr']
                auc = result['roc_auc']
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "roc_curves.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(output_path)
        
        return None
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict],
        save: bool = True
    ) -> str:
        """
        Plot comparison of model performance metrics.
        
        Args:
            results: Dictionary of model results
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(results.keys())
        
        data = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            for metric in metrics:
                data[metric].append(results[model_name].get(metric, 0))
        
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - 1.5)
            ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title(), color=color, alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "model_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(output_path)
        
        return None
    
    def plot_feature_importance(
        self,
        feature_df: pd.DataFrame,
        model_name: str,
        top_n: int = 20,
        save: bool = True
    ) -> str:
        """
        Plot feature importance.
        
        Args:
            feature_df: DataFrame with features and importance
            model_name: Name of the model
            top_n: Number of top features to plot
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        df = feature_df.head(top_n).copy()
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        
        plt.barh(df['feature'], df['importance'], color=colors, edgecolor='black', linewidth=0.5)
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f"{model_name}_feature_importance.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(output_path)
        
        return None
    
    def create_interactive_dashboard_data(
        self,
        results: Dict[str, Dict],
        feature_importance: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Create data for interactive dashboard.
        
        Args:
            results: Dictionary of model results
            feature_importance: Dictionary of feature importance DataFrames
            
        Returns:
            Dictionary with dashboard data
        """
        dashboard_data = {
            'models': {},
            'comparison': {
                'model_names': [],
                'accuracy': [],
                'balanced_accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'roc_auc': [],
                'brier_score': [],
                'fnr': [],
                'detection_rate': []
            }
        }
        
        for model_name, result in results.items():
            # Model-specific data
            dashboard_data['models'][model_name] = {
                'metrics': {
                    'accuracy': result.get('accuracy', 0),
                    'balanced_accuracy': result.get('balanced_accuracy', 0),
                    'precision': result.get('precision', 0),
                    'recall': result.get('recall', 0),
                    'f1_score': result.get('f1_score', 0),
                    'roc_auc': result.get('roc_auc', 0),
                    'brier_score': result.get('brier_score', 0),
                    'fnr': result.get('fnr', 0),
                    'detection_rate': result.get('detection_rate', 0)
                },
                'confusion_matrix': result.get('confusion_matrix', []),
                'roc_curve': result.get('roc_curve', {}),
                'classification_report': result.get('classification_report', {})
            }
            
            # Feature importance if available
            if feature_importance and model_name in feature_importance:
                fi_df = feature_importance[model_name]
                dashboard_data['models'][model_name]['feature_importance'] = {
                    'features': fi_df['feature'].tolist(),
                    'importance': fi_df['importance'].tolist()
                }
            
            # Comparison data
            dashboard_data['comparison']['model_names'].append(model_name)
            dashboard_data['comparison']['accuracy'].append(result.get('accuracy', 0))
            dashboard_data['comparison']['balanced_accuracy'].append(result.get('balanced_accuracy', 0))
            dashboard_data['comparison']['precision'].append(result.get('precision', 0))
            dashboard_data['comparison']['recall'].append(result.get('recall', 0))
            dashboard_data['comparison']['f1_score'].append(result.get('f1_score', 0))
            dashboard_data['comparison']['roc_auc'].append(result.get('roc_auc', 0))
            dashboard_data['comparison']['brier_score'].append(result.get('brier_score', 0))
            dashboard_data['comparison']['fnr'].append(result.get('fnr', 0))
            dashboard_data['comparison']['detection_rate'].append(result.get('detection_rate', 0))
        
        return dashboard_data
    
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

    def save_dashboard_data(self, dashboard_data: Dict[str, Any], filename: str = "dashboard_data.json"):
        """Save dashboard data to JSON file"""
        output_path = self.output_dir / filename
        
        sanitized_data = self._sanitize_for_json(dashboard_data)
        
        with open(output_path, 'w') as f:
            json.dump(sanitized_data, f, indent=2)
        
        return str(output_path)
