import os

# Basic paths (can be overridden by command-line args)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATABASE_DIR = os.path.join(DATA_DIR, "database")
DEFAULT_DB_PATH = os.path.join(DATABASE_DIR, "image_data.db")
DEFAULT_WARC_DIR = os.path.join(DATA_DIR, "warc")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "span_predictor")
DEFAULT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "images_output") # For downloaded images

# Default training params (can be overridden)
DEFAULT_TRAINING_PARAMS = {
    "max_len": 512,
    "batch_size": 12,
    "epochs": 10,
    "learning_rate": 5e-5,
    "validation_split": 0.2, # Changed from 0.4 for potentially more training data
}

# Default inference/enrichment params (can be overridden)
DEFAULT_INFERENCE_PARAMS = {
    "max_len": 512, # Should match training
    "max_before_tokens": 250, # Max tokens before [IMG] marker
    "min_image_size_bytes": 5 * 1024,
    "min_image_resolution": (224, 224),
    "download_timeout_seconds": 5,
    "clip_model_name": "openai/clip-vit-base-patch32",
}