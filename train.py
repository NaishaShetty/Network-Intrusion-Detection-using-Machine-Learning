"""
Training Script
Standalone script to train all models and generate visualizations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.visualization import Visualizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    logger.info("=" * 80)
    logger.info("Network Intrusion Detection System - Training Pipeline")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("\n[1/5] Initializing components...")
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    visualizer = Visualizer()
    
    # Load and preprocess data
    logger.info("\n[2/5] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Features: {len(preprocessor.feature_names)}")
    logger.info(f"  Attack ratio (train): {y_train.sum() / len(y_train):.2%}")
    logger.info(f"  Attack ratio (test): {y_test.sum() / len(y_test):.2%}")
    
    # Train models
    logger.info("\n[3/5] Training models with hyperparameter tuning...")
    logger.info("This may take several minutes depending on your hardware...")
    
    results = trainer.train_all_models(
        X_train, y_train, X_test, y_test,
        use_tuning=True
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING RESULTS SUMMARY")
    logger.info("=" * 80)
    
    for model_name, result in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Accuracy:  {result['accuracy']:.4f}")
        logger.info(f"  Precision: {result['precision']:.4f}")
        logger.info(f"  Recall:    {result['recall']:.4f}")
        logger.info(f"  F1 Score:  {result['f1_score']:.4f}")
        if 'roc_auc' in result:
            logger.info(f"  ROC AUC:   {result['roc_auc']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    # Generate visualizations
    logger.info("\n[4/5] Generating visualizations...")
    
    # Confusion matrices
    for model_name in results.keys():
        cm = results[model_name]['confusion_matrix']
        visualizer.plot_confusion_matrix(cm, model_name)
        logger.info(f"  ✓ Confusion matrix for {model_name}")
    
    # ROC curves
    visualizer.plot_roc_curve(results)
    logger.info(f"  ✓ ROC curves")
    
    # Model comparison
    visualizer.plot_model_comparison(results)
    logger.info(f"  ✓ Model comparison")
    
    # Feature importance
    feature_importance = {}
    for model_name in results.keys():
        if results[model_name].get('has_feature_importance'):
            fi_df = trainer.get_feature_importance(model_name, preprocessor.feature_names, top_n=20)
            visualizer.plot_feature_importance(fi_df, model_name)
            feature_importance[model_name] = fi_df
            logger.info(f"  ✓ Feature importance for {model_name}")
    
    # Save dashboard data
    dashboard_data = visualizer.create_interactive_dashboard_data(results, feature_importance)
    visualizer.save_dashboard_data(dashboard_data)
    logger.info(f"  ✓ Dashboard data saved")
    
    # Save models and preprocessor
    logger.info("\n[5/5] Saving models and preprocessor...")
    trainer.save_models()
    preprocessor.save_preprocessor(config.models_dir / "preprocessor.joblib")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\nModels saved to: {config.models_dir}")
    logger.info(f"Plots saved to: {config.plots_dir}")
    logger.info(f"\nYou can now:")
    logger.info(f"  1. Start the API server: python -m uvicorn src.api:app --reload")
    logger.info(f"  2. View the web dashboard at: http://localhost:8000")
    logger.info(f"  3. Make predictions using the API")
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nError during training: {e}", exc_info=True)
        sys.exit(1)
