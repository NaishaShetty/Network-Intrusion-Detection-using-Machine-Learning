"""
Configuration Management Module
Loads and manages application configuration from YAML and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = self.base_dir / config_path
        self.config = self._load_config()
        
        # Paths
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models" / "trained"
        self.plots_dir = self.base_dir / "outputs" / "plots"
        self.uploads_dir = self.base_dir / "uploads"
        self.logs_dir = self.base_dir / "logs"
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.models_dir, self.plots_dir, 
                         self.uploads_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # API Configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", 8000))
        
        # Dataset Configuration
        self.dataset_path = self.base_dir / self.config["dataset"]["path"]
        self.test_size = self.config["dataset"]["test_size"]
        self.random_state = self.config["dataset"]["random_state"]
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.config["models"].get(model_name, {})
    
    def get_enabled_models(self) -> list:
        """Get list of enabled models"""
        return [
            model_name for model_name, config in self.config["models"].items()
            if config.get("enabled", False)
        ]

# Global configuration instance
config = Config()
