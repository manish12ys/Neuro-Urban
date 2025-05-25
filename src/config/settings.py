"""
Configuration settings for NeuroUrban application.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

@dataclass
class DataConfig:
    """Data-related configuration."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    models_dir: str = "models"
    output_dir: str = "output"

    # City data sources
    target_cities: List[str] = None
    data_sources: Dict[str, str] = None

    def __post_init__(self):
        if self.target_cities is None:
            self.target_cities = [
                "Tokyo", "Singapore", "Zurich", "Copenhagen", "Amsterdam",
                "Vienna", "Munich", "Vancouver", "Toronto", "Melbourne",
                "Sydney", "Stockholm", "Helsinki", "Oslo", "Barcelona",
                "Paris", "London", "Berlin", "Seoul", "Hong Kong"
            ]

        if self.data_sources is None:
            self.data_sources = {
                "openstreetmap": "https://overpass-api.de/api/interpreter",
                "world_bank": "https://api.worldbank.org/v2/",
                "numbeo": "https://www.numbeo.com/api/",
                "satellite": "google_earth_engine"
            }

@dataclass
class ModelConfig:
    """ML/DL model configuration."""
    # Hardware Configuration
    use_gpu: bool = True
    gpu_device: str = "auto"  # "auto", "cuda:0", "cuda:1", etc.
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    # CNN Configuration
    cnn_input_size: tuple = (256, 256, 3)
    cnn_batch_size: int = 64  # Increased for GPU
    cnn_epochs: int = 100

    # GAN Configuration
    gan_latent_dim: int = 100
    gan_image_size: int = 256
    gan_batch_size: int = 32  # Increased for GPU
    gan_epochs: int = 200
    gan_learning_rate: float = 2e-4
    gan_beta1: float = 0.5
    gan_beta2: float = 0.999

    # Transformer Configuration
    transformer_model: str = "bert-base-uncased"
    max_sequence_length: int = 512

    # RL Configuration
    rl_algorithm: str = "PPO"
    rl_episodes: int = 1000
    rl_learning_rate: float = 3e-4
    rl_batch_size: int = 256  # Increased for GPU

    # Performance Optimization
    compile_models: bool = True  # PyTorch 2.0 compilation
    enable_cudnn_benchmark: bool = True
    gradient_accumulation_steps: int = 1

@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = False
    log_level: str = "INFO"
    port: int = 8501
    host: str = "localhost"

    # UI Configuration
    theme: str = "dark"
    page_title: str = "NeuroUrban: AI City Planner"
    page_icon: str = "🏙️"

class Config:
    """Main configuration class."""

    def __init__(self, config_file: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent.parent

        # Initialize sub-configurations
        self.data = DataConfig()
        self.model = ModelConfig()
        self.app = AppConfig()

        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)

        # Create necessary directories
        self._create_directories()

    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        config_path = self.project_root / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Update configurations
            if 'data' in config_data:
                for key, value in config_data['data'].items():
                    setattr(self.data, key, value)

            if 'model' in config_data:
                for key, value in config_data['model'].items():
                    setattr(self.model, key, value)

            if 'app' in config_data:
                for key, value in config_data['app'].items():
                    setattr(self.app, key, value)

    def _create_directories(self):
        """Create necessary project directories."""
        directories = [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.models_dir,
            self.data.output_dir,
            "logs",
            "notebooks",
            "tests"
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_data_path(self, filename: str, data_type: str = "raw") -> Path:
        """Get full path for data file."""
        if data_type == "raw":
            return self.project_root / self.data.raw_data_dir / filename
        elif data_type == "processed":
            return self.project_root / self.data.processed_data_dir / filename
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def get_model_path(self, model_name: str) -> Path:
        """Get full path for model file."""
        return self.project_root / self.data.models_dir / model_name

    def get_output_path(self, filename: str) -> Path:
        """Get full path for output file."""
        return self.project_root / self.data.output_dir / filename
