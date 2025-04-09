import argparse
import os
import sys
import logging
import json # Or yaml if using config.yaml

# Adjust path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.common.logging_config import setup_logging
from src.common.config import DEFAULT_DB_PATH, DEFAULT_MODELS_DIR, DEFAULT_TRAINING_PARAMS
from src.training.train import train_model

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Train the span prediction model.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database.")
    parser.add_argument("--model-save-dir", type=str, default=DEFAULT_MODELS_DIR, help="Directory to save trained models.")
    parser.add_argument("--params-json", type=str, default=None, help="Path to a JSON file with training parameters (overrides defaults).")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    # Add specific param overrides if needed
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    # --- Start New Argument ---
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum number of training examples to load from DB (default: use all).")
    # --- End New Argument ---

    args = parser.parse_args()

    # Load and merge parameters
    training_params = DEFAULT_TRAINING_PARAMS.copy()
    if args.params_json:
        try:
            with open(args.params_json, 'r') as f:
                override_params = json.load(f)
                training_params.update(override_params)
                logger.info(f"Loaded training parameters from {args.params_json}")
        except Exception as e:
            logger.error(f"Failed to load parameters from {args.params_json}: {e}")
            sys.exit(1)

    # Override specific params from command line
    if args.epochs is not None: training_params['epochs'] = args.epochs
    if args.batch_size is not None: training_params['batch_size'] = args.batch_size
    if args.lr is not None: training_params['learning_rate'] = args.lr
    # --- Start Pass New Argument ---
    # Add max_examples to training_params or pass separately
    training_params['max_examples'] = args.max_examples
    # --- End Pass New Argument ---


    logger.info(f"Using training parameters: {training_params}")

    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        sys.exit(1)

    os.makedirs(args.model_save_dir, exist_ok=True)

    # Pass the max_examples limit to train_model
    train_model(
        db_path=args.db_path,
        model_save_dir=args.model_save_dir,
        training_params=training_params, # Pass the whole dict
        run_wandb=not args.no_wandb
        # max_examples=args.max_examples # Alternative: pass as separate arg
    )

    logger.info("Training script finished.")

if __name__ == "__main__":
    main()
