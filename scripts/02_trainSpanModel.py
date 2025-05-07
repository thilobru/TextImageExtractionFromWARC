import argparse
import os
import sys
import logging
import json # For loading params from JSON

# Adjust path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.common.logging_config import setup_logging
from src.common.config import DEFAULT_DB_PATH, DEFAULT_MODELS_DIR, DEFAULT_TRAINING_PARAMS
from src.training.train import train_model # Uses the updated train.py

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Train the span prediction model using Hugging Face Transformers.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, 
                        help="Path to the SQLite database.")
    parser.add_argument("--model-save-dir", type=str, default=DEFAULT_MODELS_DIR, 
                        help="Directory to save trained models and tokenizers.")
    parser.add_argument("--params-json", type=str, default=None, 
                        help="Path to a JSON file with training parameters (overrides defaults and script arguments).")
    parser.add_argument("--no-wandb", action="store_true", 
                        help="Disable Weights & Biases logging.")
    
    # Allow overriding key parameters directly from command line
    parser.add_argument("--model-checkpoint", type=str, default=None,
                        help="Hugging Face model checkpoint to use (e.g., 'distilbert-base-uncased', 'bert-base-uncased'). Overrides value in params JSON or defaults.")
    parser.add_argument("--epochs", type=int, default=None, 
                        help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, 
                        help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, 
                        help="Override learning rate.")
    parser.add_argument("--max-len", type=int, default=None,
                        help="Override maximum sequence length.")
    parser.add_argument("--validation-split", type=float, default=None,
                        help="Override validation split ratio.")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Override patience for early stopping.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum number of training examples to load from DB (default: use all).")

    args = parser.parse_args()

    # --- Parameter Loading and Merging ---
    # Start with hardcoded defaults from config.py
    training_params = DEFAULT_TRAINING_PARAMS.copy()
    logger.info(f"Loaded default parameters: {training_params}")

    # Override with parameters from JSON file if provided
    if args.params_json:
        try:
            with open(args.params_json, 'r') as f:
                json_params = json.load(f)
                training_params.update(json_params)
                logger.info(f"Loaded and merged parameters from JSON file '{args.params_json}': {json_params}")
        except FileNotFoundError:
            logger.error(f"Parameters JSON file not found: {args.params_json}")
            sys.exit(1)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from parameters file: {args.params_json}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load parameters from {args.params_json}: {e}")
            sys.exit(1)

    # Override with command-line arguments (highest precedence)
    cli_overrides = {
        "model_checkpoint": args.model_checkpoint,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr, # Note: arg is --lr, param key is learning_rate
        "max_len": args.max_len,
        "validation_split": args.validation_split,
        "early_stopping_patience": args.early_stopping_patience,
        "max_examples": args.max_examples
    }
    # Filter out None values from CLI overrides before updating
    active_cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    if active_cli_overrides:
        training_params.update(active_cli_overrides)
        logger.info(f"Applied command-line parameter overrides: {active_cli_overrides}")
    
    logger.info(f"Final training parameters: {training_params}")

    # --- Validate Paths ---
    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        sys.exit(1)

    os.makedirs(args.model_save_dir, exist_ok=True) # Ensure model save directory exists

    # --- Start Training ---
    try:
        train_model(
            db_path=args.db_path,
            model_save_dir=args.model_save_dir,
            training_params=training_params, # Pass the fully resolved params
            run_wandb=not args.no_wandb
        )
    except Exception as e:
        logger.error(f"An error occurred during the training process: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Training script finished successfully.")

if __name__ == "__main__":
    main()
