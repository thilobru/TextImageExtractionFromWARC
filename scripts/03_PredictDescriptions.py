
import argparse
import os
import sys
import logging

# Adjust path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.common.logging_config import setup_logging
from src.common.config import DEFAULT_DB_PATH, DEFAULT_MODELS_DIR, DEFAULT_INFERENCE_PARAMS
from src.inference.predict_spans import run_inference

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run span prediction inference (text only).")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained Keras model (.h5 or SavedModel dir).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction.")
    parser.add_argument("--max-len", type=int, default=DEFAULT_INFERENCE_PARAMS['max_len'], help="Max sequence length for model.")
    parser.add_argument("--max-before-tokens", type=int, default=DEFAULT_INFERENCE_PARAMS['max_before_tokens'], help="Max tokens from context before [IMG].")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        logger.error(f"Model path not found: {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.db_path):
        logger.error(f"Database path not found: {args.db_path}")
        sys.exit(1)

    run_inference(
        db_path=args.db_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_before_tokens=args.max_before_tokens
    )

    logger.info("Inference script finished.")

if __name__ == "__main__":
    main()
