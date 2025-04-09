import argparse
import os
import sys
import logging
from tqdm import tqdm
import time

# Adjust path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.common.logging_config import setup_logging
from src.common.config import DEFAULT_DB_PATH, DEFAULT_IMAGES_DIR, DEFAULT_INFERENCE_PARAMS
from src.common.database import db_connect, get_predictions_for_enrichment, add_clip_result
from src.enrichment.image_downloader import download_and_validate_image
from src.enrichment.clip_scorer import get_clip_score, load_clip_model # Import load_clip_model

logger = logging.getLogger(__name__)

def run_enrichment(db_path, images_output_dir, batch_size=100, clip_model_name=DEFAULT_INFERENCE_PARAMS['clip_model_name']):
    """Downloads images and calculates CLIP scores for pending predictions."""
    logger.info("Starting enrichment process...")

    min_size = DEFAULT_INFERENCE_PARAMS['min_image_size_bytes']
    min_res = DEFAULT_INFERENCE_PARAMS['min_image_resolution']
    timeout = DEFAULT_INFERENCE_PARAMS['download_timeout_seconds']

    # Pre-load CLIP model once
    try:
        load_clip_model(clip_model_name)
    except Exception as e:
        logger.error(f"Failed to preload CLIP model: {e}. Aborting.")
        return

    processed_count = 0
    success_count = 0
    fail_download_count = 0
    fail_clip_count = 0
    offset = 0

    while True:
        logger.info(f"Fetching enrichment batch starting from offset {offset}...")
        predictions_to_process = []
        try:
            with db_connect(db_path) as cursor:
                predictions_to_process = get_predictions_for_enrichment(cursor, limit=batch_size, offset=offset) # Use offset here
        except Exception as e:
            logger.error(f"Database error fetching predictions: {e}")
            break

        if not predictions_to_process:
            logger.info("No more pending predictions found for enrichment.")
            break

        logger.info(f"Processing {len(predictions_to_process)} predictions...")
        # Note: This loop processes one by one. For higher throughput, consider
        # parallel downloads (using asyncio or multiprocessing) and potentially
        # batching CLIP score calculation if feasible (though image loading often dominates).
        for row in tqdm(predictions_to_process, desc="Enriching Batch"):
            prediction_id = row['prediction_id']
            image_url = row['image_url']
            predicted_text = row['predicted_text']
            local_path = None
            clip_score = None
            error_msg = None

            # 1. Download Image
            try:
                local_path = download_and_validate_image(image_url, images_output_dir, min_size, min_res, timeout)
                if local_path is None:
                    error_msg = "Download/Validation failed"
                    fail_download_count += 1
            except Exception as dl_err:
                error_msg = f"Download exception: {dl_err}"
                fail_download_count += 1
                logger.warning(f"Error downloading {image_url} for prediction {prediction_id}: {dl_err}")


            # 2. Calculate CLIP Score (if download succeeded)
            if local_path:
                try:
                    clip_score = get_clip_score(local_path, predicted_text, clip_model_name)
                    if clip_score is None:
                        error_msg = "CLIP scoring failed"
                        fail_clip_count += 1
                    else:
                         success_count += 1 # Scored successfully
                except Exception as clip_err:
                     error_msg = f"CLIP exception: {clip_err}"
                     fail_clip_count += 1
                     logger.warning(f"Error calculating CLIP score for {local_path} (prediction {prediction_id}): {clip_err}")


            # 3. Update Database
            try:
                with db_connect(db_path) as cursor_update:
                     add_clip_result(cursor_update, prediction_id, local_path, clip_score, clip_model_name, error_msg)
            except Exception as db_err:
                 logger.error(f"Failed to update DB for prediction {prediction_id}: {db_err}")
                 # Log failure but continue processing others in the batch

            processed_count += 1

        # IMPORTANT: If using offset for fetching, you MUST ensure that fetching
        # `get_predictions_for_enrichment` reliably fetches only 'pending' items
        # and that the status is updated correctly by `add_clip_result`.
        # If not using offset, remove the offset logic and rely on fetching only 'pending'.
        # Using offset is generally safer if updates might fail or take time.
        # offset += len(predictions_to_process) # Advance offset for next fetch

        # Alternative: Don't use offset, just keep fetching 'pending' ones.
        # This might re-process failed updates but is simpler if DB updates are reliable.
        # If using this, add a small delay to avoid busy-waiting if no items are found.
        if not predictions_to_process:
             time.sleep(5) # Wait before checking again if no items were found

    logger.info(f"Enrichment finished. Total processed: {processed_count}, Succeeded: {success_count}, Download fails: {fail_download_count}, CLIP fails: {fail_clip_count}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Download images and calculate CLIP scores.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database.")
    parser.add_argument("--images-dir", type=str, default=DEFAULT_IMAGES_DIR, help="Directory to save downloaded images.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of predictions to fetch from DB at a time.")
    # Add arg for clip model name if needed
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        logger.error(f"Database path not found: {args.db_path}")
        sys.exit(1)

    os.makedirs(args.images_dir, exist_ok=True)

    run_enrichment(args.db_path, args.images_dir, args.batch_size)


if __name__ == "__main__":
    main()
