import argparse
import os
import sys
import logging
import json

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
    # Add specific param overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--max-len", type=int, default=None, help="Override maximum sequence length.")
    parser.add_argument("--validation-split", type=float, default=None, help="Override validation split ratio.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Override patience for early stopping.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum number of training examples to load from DB (default: use all).")
    parser.add_argument("--dropout", type=float, default=0.1, help="Override dropout rate (default: 0.1).")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Override weight decay (L2 regularization) (default: 0.01).")
    parser.add_argument("--model-checkpoint", type=str, default="roberta-base",  # Changed default
                        help="Override the pre-trained model checkpoint (e.g., roberta-base, bert-base-uncased).")

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
    if args.max_len is not None: training_params['max_len'] = args.max_len
    if args.validation_split is not None: training_params['validation_split'] = args.validation_split
    if args.early_stopping_patience is not None: training_params['early_stopping_patience'] = args.early_stopping_patience
    if args.max_examples is not None: training_params['max_examples'] = args.max_examples
    # Add new regularization and model selection parameters
    training_params['dropout'] = args.dropout
    training_params['weight_decay'] = args.weight_decay
    training_params['model_checkpoint'] = args.model_checkpoint # Use the overridden model checkpoint

    logger.info(f"Using training parameters: {training_params}")

    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        sys.exit(1)

    os.makedirs(args.model_save_dir, exist_ok=True)

    train_model(
        db_path=args.db_path,
        model_save_dir=args.model_save_dir,
        training_params=training_params,
        run_wandb=not args.no_wandb
    )

    logger.info("Training script finished.")

if __name__ == "__main__":
    main()
