"""
FastAPI Backend for Network Intrusion Detection System
Provides REST API endpoints for model training, prediction, and visualization
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime
import io

from sklearn.calibration import CalibratedClassifierCV
from .config import config
from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .visualization import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Network Intrusion Detection System",
    description="ML-powered network intrusion detection with multiple models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
preprocessor = DataPreprocessor()
trainer = ModelTrainer()
visualizer = Visualizer()

# Training status
training_status = {
    'is_training': False,
    'progress': 0,
    'message': '',
    'completed': False
}

@app.on_event("startup")
async def startup_event():
    """Load all trained models on startup"""
    try:
        available_models = [f.stem for f in config.models_dir.glob("*.joblib") if f.stem != "preprocessor"]
        for model_name in available_models:
            model_path = config.models_dir / f"{model_name}.joblib"
            trainer.load_model(model_name, model_path)
            logger.info(f"Loaded model: {model_name}")
        
        preprocessor_path = config.models_dir / "preprocessor.joblib"
        if preprocessor_path.exists():
            preprocessor.load_preprocessor(preprocessor_path)
            logger.info("Loaded preprocessor")
    except Exception as e:
        logger.error(f"Error during startup model loading: {e}")


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    model_name: str
    features: List[List[float]]


class PredictionResponse(BaseModel):
    predictions: List[str]
    probabilities: List[float]
    model_name: str


class TrainingRequest(BaseModel):
    use_tuning: bool = True
    models: Optional[List[str]] = None


class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None


# API Endpoints

@app.get("/api/status")
async def root():
    """Status endpoint"""
    return {
        "message": "Network Intrusion Detection System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get dataset information"""
    try:
        if not config.dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load a sample
        df = pd.read_csv(config.dataset_path, nrows=1000, header=None)
        
        return {
            "name": "KDD Cup 1999",
            "path": str(config.dataset_path),
            "total_features": 41,
            "sample_size": len(df),
            "features": DataPreprocessor.COLUMN_NAMES
        }
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train ML models.
    This is a long-running operation, so we run it in the background.
    """
    global training_status
    
    if training_status['is_training']:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    def train_task():
        global training_status
        try:
            training_status['is_training'] = True
            training_status['progress'] = 10
            training_status['message'] = "Loading and preprocessing data..."
            
            # Load and preprocess data
            X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()
            
            training_status['progress'] = 30
            training_status['message'] = "Starting training..."
            
            # Callback upon granular progress
            def progress_callback(msg, prog):
                training_status['message'] = msg
                training_status['progress'] = prog

            # Train models
            results = trainer.train_all_models(
                X_train, y_train, X_test, y_test,
                use_tuning=request.use_tuning,
                update_callback=progress_callback
            )
            
            training_status['progress'] = 70
            training_status['message'] = "Generating visualizations..."
            
            # Generate visualizations
            for model_name in results.keys():
                cm = np.array(results[model_name]['confusion_matrix'])
                visualizer.plot_confusion_matrix(cm, model_name)
                
                if results[model_name].get('has_feature_importance'):
                    fi_df = trainer.get_feature_importance(model_name, preprocessor.feature_names)
                    visualizer.plot_feature_importance(fi_df, model_name)
            
            visualizer.plot_roc_curve(results)
            visualizer.plot_model_comparison(results)
            
            training_status['progress'] = 90
            training_status['message'] = "Saving models..."
            
            # Save models and preprocessor
            trainer.save_models()
            preprocessor.save_preprocessor(config.models_dir / "preprocessor.joblib")
            
            # Save dashboard data
            feature_importance = {}
            for model_name in results.keys():
                if results[model_name].get('has_feature_importance'):
                    feature_importance[model_name] = trainer.get_feature_importance(
                        model_name, preprocessor.feature_names
                    )
            
            dashboard_data = visualizer.create_interactive_dashboard_data(results, feature_importance)
            visualizer.save_dashboard_data(dashboard_data)
            
            training_status['progress'] = 100
            training_status['message'] = "Training completed successfully!"
            training_status['completed'] = True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            training_status['message'] = f"Error: {str(e)}"
        finally:
            training_status['is_training'] = False
    
    background_tasks.add_task(train_task)
    
    return {
        "message": "Training started",
        "status": "in_progress"
    }


@app.get("/api/training/status")
async def get_training_status():
    """Get current training status"""
    return training_status


@app.get("/api/models")
async def get_available_models():
    """Get list of available trained models"""
    models = []
    
    for model_file in config.models_dir.glob("*.joblib"):
        if model_file.stem != "preprocessor":
            models.append(model_file.stem)
    
    return {
        "models": models,
        "count": len(models)
    }


@app.get("/api/models/{model_name}/metrics")
async def get_model_metrics(model_name: str):
    """Get metrics for a specific model"""
    metadata_path = config.models_dir / "training_metadata.json"
    
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Training metadata not found. Please train models first.")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if model_name not in metadata['results']:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return metadata['results'][model_name]


@app.get("/api/results/dashboard")
async def get_dashboard_data():
    """Get all dashboard data"""
    dashboard_path = config.plots_dir / "dashboard_data.json"
    
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard data not found. Please train models first.")
    
    def sanitize(obj):
        if isinstance(obj, float):
            if np.isinf(obj) or np.isnan(obj):
                return 0.0
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(dashboard_path, 'r') as f:
        data = json.load(f)
        return sanitize(data)


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...), 
    model_name: str = "random_forest",
    threshold: float = 0.5
):
    """
    Make predictions on uploaded CSV file.
    
    Args:
        file: CSV file with network traffic data
        model_name: Name of the model to use
        threshold: Classification threshold (0.0 to 1.0). If P(Attack) >= threshold, classify as Attack.
    """
    try:
        # Load model and preprocessor
        model_path = config.models_dir / f"{model_name}.joblib"
        preprocessor_path = config.models_dir / "preprocessor.joblib"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found. Please train first.")
        
        if not preprocessor_path.exists():
            raise HTTPException(status_code=404, detail="Preprocessor not found. Please train first.")
        
        # Load artifacts
        try:
             # Check if loaded in memory to avoid disk I/O
             if model_name not in trainer.models:
                  trainer.load_model(model_name, model_path)
        except:
             trainer.load_model(model_name, model_path)
             
        preprocessor.load_preprocessor(preprocessor_path)
        
        # Read uploaded file
        contents = await file.read()
        
        # Save to uploads for drift analysis
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            upload_path = config.uploads_dir / f"traffic_{timestamp}.csv"
            with open(upload_path, "wb") as f:
                f.write(contents)
            logger.info(f"Saved uploaded traffic to {upload_path}")
        except Exception as e:
            logger.warning(f"Failed to save upload for drift analysis: {e}")

        try:
            # First try reading without header to check first row
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')), header=None)
        except pd.errors.EmptyDataError:
             raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Check if first row is a header (contains "duration")
        if not df.empty and isinstance(df.iloc[0, 0], str) and 'duration' in str(df.iloc[0, 0]).lower():
            # Reload with header or set columns from first row
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            
        # Robust column handling
        feature_cols = DataPreprocessor.COLUMN_NAMES[:-1] # Expected features (41)
        
        # Clean up column names if they were read from header
        if df.columns[0] == 'duration':
             # Columns are already likely correct, but let's standardize
             pass
        
        if len(df.columns) == 43:
            # NSL-KDD: 43 columns (41 features + 1 label + 1 score)
            df = df.iloc[:, :42]
            df.columns = DataPreprocessor.COLUMN_NAMES
        elif len(df.columns) == 42:
            # KDD with label: 42 columns
            df.columns = DataPreprocessor.COLUMN_NAMES
        elif len(df.columns) == 41:
            # Features only: 41 columns
            df.columns = feature_cols
            df['label'] = 'unknown' # Dummy label for preprocessing
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid number of columns. Expected 41, 42, or 43. Got {len(df.columns)}"
            )
        
        # Preprocess
        X, _ = preprocessor.preprocess(df, fit=False)
        
        # Predict with threshold
        try:
            probs = trainer.predict(model_name, X, return_proba=True)
            prob_attack = probs[:, 0]
            prob_normal = probs[:, 1]
        except:
             # Fallback if probability not supported
             # Mock probabilities based on hard predictions
             raw_preds = trainer.predict(model_name, X)
             prob_attack = np.array([1.0 if p == 'attack' else 0.0 for p in raw_preds])
             prob_normal = np.array([1.0 if p == 'normal' else 0.0 for p in raw_preds])
        
        # Apply Threshold Logic
        prediction_labels = []
        logger.info(f"Applying threshold {threshold} to {len(prob_attack)} samples. Min prob: {np.min(prob_attack):.4f}, Max prob: {np.max(prob_attack):.4f}")
        for p_att in prob_attack:
            if p_att >= threshold:
                prediction_labels.append('attack')
            else:
                prediction_labels.append('normal')
        
        # Create results
        results = []
        for i, (pred_label, p_att, p_norm) in enumerate(zip(prediction_labels, prob_attack, prob_normal)):
            results.append({
                'index': i,
                'prediction': pred_label,
                'confidence': float(p_att) if pred_label == 'attack' else float(p_norm),
                'attack_probability': float(p_att)
            })
        
        # Probability Histogram for UI visualization
        hist, bin_edges = np.histogram(prob_attack, bins=10, range=(0, 1))
        hist_data = [{"bin": float(bin_edges[i]), "count": int(count)} for i, count in enumerate(hist)]
        
        return {
            'model_name': model_name,
            'threshold': threshold,
            'latency_ms': float(trainer.last_prediction_latency.get(model_name, 0)),
            'total_samples': len(results),
            'predictions': results,
            'probability_distribution': hist_data,
            'summary': {
                'normal_count': sum(1 for r in results if r['prediction'] == 'normal'),
                'attack_count': sum(1 for r in results if r['prediction'] == 'attack'),
                'avg_confidence': float(np.mean([r['confidence'] for r in results]) if results else 0.0)
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monitoring/drift")
async def get_drift_metrics():
    """Calculate and return concept drift metrics"""
    # This would typically compare the latest batch vs baseline
    # For now, we use a mock recent batch or if one exists in uploads
    uploads_dir = Path("uploads")
    if not uploads_dir.exists() or not any(uploads_dir.iterdir()):
         return {"status": "no_data", "message": "No recent traffic data for drift analysis"}
    
    # Take the most recent file
    latest_file = max(uploads_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    try:
        df = pd.read_csv(latest_file)
        drift_results = preprocessor.calculate_drift_score(df)
        return drift_results
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/monitoring/stats")
async def get_monitoring_stats():
    """Get system monitoring statistics"""
    return {
        "latencies": trainer.last_prediction_latency,
        "is_calibrated": all(isinstance(m, CalibratedClassifierCV) for m in trainer.models.values()) if trainer.models else False,
        "active_models": list(trainer.models.keys())
    }


@app.get("/api/plots/{plot_name}")
async def get_plot(plot_name: str):
    """Get a specific plot image"""
    plot_path = config.plots_dir / plot_name
    
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    
    return FileResponse(plot_path)


@app.get("/api/plots")
async def list_plots():
    """List all available plots"""
    plots = []
    
    for plot_file in config.plots_dir.glob("*.png"):
        plots.append({
            'name': plot_file.name,
            'url': f"/api/plots/{plot_file.name}"
        })
    
    return {
        'plots': plots,
        'count': len(plots)
    }


# Serve Frontend Static Files
frontend_path = Path("frontend/build")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api_host, port=config.api_port)
