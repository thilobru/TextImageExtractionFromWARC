# src/data_extraction/warc_processor.py
from fastwarc.warc import ArchiveIterator, WarcRecordType
from .html_parser import parse_html_page
from ..common.database import db_connect, add_image_context, mark_training_candidate
import logging
import os

logger = logging.getLogger(__name__)

def process_warc_file(warc_path, db_path, mode='inference'):
    """
    Processes a WARC file, extracts image contexts, and saves them to the database.

    Args:
        warc_path (str): Path to the WARC file.
        db_path (str): Path to the SQLite database file.
        mode (str): 'training' or 'inference'. In 'training' mode, marks contexts
                    where alt text is found as training candidates.
    """
    count_records = 0
    count_html = 0
    count_processed = 0
    count_added_context = 0
    count_marked_training = 0
    warc_filename = os.path.basename(warc_path)
    logger.info(f"Starting processing of WARC file: {warc_filename}, mode: {mode}")

    try:
        with open(warc_path, "rb") as f, db_connect(db_path) as cursor:
            # Use stream=True for potentially large records, adjust parse_http if needed
            iterator = ArchiveIterator(f, record_types=WarcRecordType.response, parse_http=True)

            for record in iterator:
                count_records += 1
                if record.http_headers is None or record.content_type is None:
                    continue

                if record.content_type.startswith("text/html") and record.content_length >= 128:
                    count_html += 1
                    page_url = str(record.headers['WARC-Target-URI'])
                    html_bytes = record.reader.read() # Read content bytes

                    if not html_bytes:
                        continue

                    try:
                        # Parse the HTML page to get image contexts
                        image_contexts = parse_html_page(html_bytes, page_url)
                        count_processed += 1

                        for context_data in image_contexts:
                            # Add the context to the database
                            context_id = add_image_context(
                                cursor,
                                image_url=context_data['image_url'],
                                page_url=page_url,
                                warc_path=warc_filename, # Store filename for reference
                                context_before=context_data['context_before'],
                                context_after=context_data['context_after'],
                                alt_text=context_data['alt_text']
                            )

                            if context_id: # If successfully inserted (not a duplicate)
                                count_added_context += 1
                                # If in training mode and alt text was found, mark it
                                if mode == 'training' and context_data['found_alt_in_context']:
                                     if mark_training_candidate(cursor, context_id, context_data['alt_text']):
                                         count_marked_training += 1

                    except Exception as parse_err:
                        logger.error(f"Error parsing content from {page_url} in {warc_filename}: {parse_err}", exc_info=False)

                if count_records % 1000 == 0:
                    logger.info(f"[{warc_filename}] Processed {count_records} records...")

    except FileNotFoundError:
        logger.error(f"WARC file not found: {warc_path}")
    except Exception as e:
        logger.error(f"Failed processing {warc_filename}: {e}", exc_info=True)
    finally:
        logger.info(f"Finished processing WARC file: {warc_filename}")
        logger.info(f"  Total Records: {count_records}")
        logger.info(f"  HTML Responses: {count_html}")
        logger.info(f"  Successfully Parsed: {count_processed}")
        logger.info(f"  Contexts Added/Updated: {count_added_context}")
        if mode == 'training':
            logger.info(f"  Marked for Training: {count_marked_training}")