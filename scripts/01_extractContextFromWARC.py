import argparse
import os
import sys
import logging

# Adjust path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.common.logging_config import setup_logging
from src.common.config import DEFAULT_DB_PATH, DEFAULT_WARC_DIR
from src.common.database import initialize_schema
from src.data_extraction.warc_processor import process_warc_file

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Extract image contexts from WARC files.")
    parser.add_argument("warc_path", type=str, help="Path to a single WARC file or a directory containing WARC files.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database.")
    parser.add_argument("--mode", type=str, choices=['training', 'inference'], default='inference',
                        help="Mode: 'training' marks candidates, 'inference' extracts all contexts.")
    parser.add_argument("--recursive", action="store_true", help="Process WARC files in subdirectories if warc_path is a directory.")

    args = parser.parse_args()

    # Initialize DB schema if DB doesn't exist
    if not os.path.exists(args.db_path):
        logger.info(f"Database not found at {args.db_path}, initializing schema...")
        try:
             initialize_schema(args.db_path)
        except Exception as e:
             logger.error(f"Failed to initialize database: {e}")
             sys.exit(1)

    if os.path.isfile(args.warc_path):
        process_warc_file(args.warc_path, args.db_path, args.mode)
    elif os.path.isdir(args.warc_path):
        logger.info(f"Processing WARC files in directory: {args.warc_path}")
        for root, _, files in os.walk(args.warc_path):
            for file in files:
                if file.endswith(('.warc', '.warc.gz')):
                    full_path = os.path.join(root, file)
                    process_warc_file(full_path, args.db_path, args.mode)
            if not args.recursive: # Only process top-level directory if not recursive
                 break
    else:
        logger.error(f"Invalid WARC path: {args.warc_path}")
        sys.exit(1)

    logger.info("WARC processing finished.")

if __name__ == "__main__":
    main()
