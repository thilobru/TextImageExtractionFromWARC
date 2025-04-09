import argparse
import os
import sys
import logging

# Adjust path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.common.logging_config import setup_logging
from src.common.config import DEFAULT_DB_PATH
from src.common.database import db_connect, get_final_results
# Import your analysis functions (plotting etc.) from src/analysis/report.py or use directly
# from src.analysis.report import generate_plots, generate_stats

logger = logging.getLogger(__name__)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Analyze final results from the database.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database.")
    parser.add_argument("--output-dir", type=str, default="./analysis_output", help="Directory to save plots and stats.")
    parser.add_argument("--min-clip", type=float, default=None, help="Minimum CLIP score to include in analysis.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of results to analyze.")

    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        logger.error(f"Database path not found: {args.db_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Fetching final results for analysis...")
    results_data = []
    try:
        with db_connect(args.db_path) as cursor:
            results_data = get_final_results(cursor, min_clip_score=args.min_clip, limit=args.limit)
    except Exception as e:
        logger.error(f"Failed to fetch results: {e}")
        sys.exit(1)

    if not results_data:
        logger.warning("No results found matching the criteria for analysis.")
        sys.exit(0)

    logger.info(f"Analyzing {len(results_data)} results...")

    # --- Call your analysis functions here ---
    # Example:
    # df = pd.DataFrame(results_data, columns=[desc[0] for desc in cursor.description]) # Need cursor description if using fetchall directly
    # generate_plots(df, args.output_dir)
    # generate_stats(df, args.output_dir)
    #
    # Or adapt your existing analysis script (Stage 5 code) to read from `results_data` (list of Row objects)
    # You might convert it to a pandas DataFrame first for easier plotting with seaborn/matplotlib.

    # Placeholder for your analysis logic:
    print(f"Analysis logic goes here. Data fetched. Output directory: {args.output_dir}")
    # Example: Print first 5 results
    for i, row in enumerate(results_data[:5]):
        print(f"Result {i}: Image={row['image_url']}, CLIP={row['clip_score']:.4f}, Desc='{row['predicted_text'][:50]}...'")


    logger.info("Analysis script finished.")


if __name__ == "__main__":
    main()
