import argparse
import os
import sys
import logging

# Adjust path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.common.logging_config import setup_logging
from src.common.config import DEFAULT_DB_PATH, DEFAULT_WARC_DIR
from src.common.database import initialize_schema # Import initialize_schema
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

    # --- Start Correction ---
    # Always attempt to initialize/verify the database schema first
    try:
        logger.info(f"Initializing/Verifying database schema at {args.db_path}...")
        # Ensure the directory for the database exists
        db_dir = os.path.dirname(args.db_path)
        if db_dir: # Check if path includes a directory
             os.makedirs(db_dir, exist_ok=True)
        initialize_schema(args.db_path) # Call this *before* processing
        logger.info("Database schema initialization/verification complete.")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}", exc_info=True)
        sys.exit(1)
    # --- End Correction ---


    # Proceed with WARC processing only if schema init succeeded
    if os.path.isfile(args.warc_path):
        process_warc_file(args.warc_path, args.db_path, args.mode)
    elif os.path.isdir(args.warc_path):
        logger.info(f"Processing WARC files in directory: {args.warc_path}")
        for root, _, files in os.walk(args.warc_path):
            # Filter for WARC files, handling potential case variations and .gz
            warc_files = [f for f in files if f.lower().endswith(('.warc', '.warc.gz'))]
            if not warc_files:
                 continue # Skip directories with no WARC files

            logger.info(f"Found {len(warc_files)} WARC files in {root}")
            for file in sorted(warc_files): # Sort for consistent order
                full_path = os.path.join(root, file)
                logger.info(f"Processing file: {file}")
                process_warc_file(full_path, args.db_path, args.mode)

            if not args.recursive: # Only process top-level directory if not recursive
                 logger.info("Non-recursive mode: Finished processing top-level directory.")
                 break
        logger.info(f"Finished processing directory: {args.warc_path}")
    else:
        logger.error(f"Invalid WARC path (not a file or directory): {args.warc_path}")
        sys.exit(1)

    logger.info("WARC processing script finished.")

if __name__ == "__main__":
    main()
