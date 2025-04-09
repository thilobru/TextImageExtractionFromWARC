import sqlite3
import json
import logging
import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def db_connect(db_path):
    """Provides a transactional scope around a series of operations."""
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=10) # Increased timeout
        # Enable Write-Ahead Logging for better read/write concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row # Access columns by name
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise # Re-raise the exception
    finally:
        if conn:
            conn.close()

def initialize_schema(db_path):
    """Creates the database tables if they don't exist."""
    schema = """
    CREATE TABLE IF NOT EXISTS ImageContext (
        context_id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_url TEXT NOT NULL,
        page_url TEXT,
        warc_path TEXT,
        context_before TEXT,
        context_after TEXT,
        alt_text TEXT,
        is_training_candidate BOOLEAN DEFAULT 0,
        status_extraction TEXT DEFAULT 'pending', -- e.g., 'pending', 'processed', 'failed'
        extraction_timestamp DATETIME,
        UNIQUE(image_url, page_url) -- Avoid duplicate entries for the same image on the same page
    );

    CREATE INDEX IF NOT EXISTS idx_image_url ON ImageContext(image_url);
    CREATE INDEX IF NOT EXISTS idx_page_url ON ImageContext(page_url);
    CREATE INDEX IF NOT EXISTS idx_extraction_status ON ImageContext(status_extraction);
    CREATE INDEX IF NOT EXISTS idx_is_training_candidate ON ImageContext(is_training_candidate);

    CREATE TABLE IF NOT EXISTS TrainingData (
        training_id INTEGER PRIMARY KEY AUTOINCREMENT,
        context_id INTEGER NOT NULL,
        input_tokens_json TEXT NOT NULL,
        attention_mask_json TEXT NOT NULL,
        true_start_token INTEGER NOT NULL,
        true_end_token INTEGER NOT NULL,
        alt_text_used TEXT,
        created_timestamp DATETIME,
        FOREIGN KEY (context_id) REFERENCES ImageContext(context_id)
    );

    CREATE TABLE IF NOT EXISTS PredictedDescriptions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        context_id INTEGER NOT NULL,
        model_path TEXT NOT NULL,
        predicted_text TEXT,
        start_token INTEGER,
        end_token INTEGER,
        start_confidence REAL,
        end_confidence REAL,
        average_confidence REAL,
        span_confidence REAL,
        status_enrichment TEXT DEFAULT 'pending', -- e.g., 'pending', 'downloaded', 'clip_scored', 'failed_download', 'failed_clip'
        prediction_timestamp DATETIME,
        FOREIGN KEY (context_id) REFERENCES ImageContext(context_id)
    );

    CREATE INDEX IF NOT EXISTS idx_pred_context_id ON PredictedDescriptions(context_id);
    CREATE INDEX IF NOT EXISTS idx_pred_model_path ON PredictedDescriptions(model_path);
    CREATE INDEX IF NOT EXISTS idx_enrichment_status ON PredictedDescriptions(status_enrichment);


    CREATE TABLE IF NOT EXISTS ClipResults (
        clip_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL UNIQUE, -- Ensure one result per prediction
        image_local_path TEXT,
        clip_score REAL,
        clip_model_name TEXT,
        enrichment_timestamp DATETIME,
        error_message TEXT,
        FOREIGN KEY (prediction_id) REFERENCES PredictedDescriptions(prediction_id)
    );
    """
    try:
        with db_connect(db_path) as cursor:
            cursor.executescript(schema)
        logger.info(f"Database schema initialized/verified at {db_path}")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise

# --- Functions for Stage 1 / Stage 3a ---

def add_image_context(cursor, image_url, page_url, warc_path, context_before, context_after, alt_text=None):
    """Adds an image context entry. Returns the context_id."""
    sql = """
    INSERT INTO ImageContext (image_url, page_url, warc_path, context_before, context_after, alt_text, status_extraction, extraction_timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(image_url, page_url) DO NOTHING;
    """
    # ON CONFLICT... prevents duplicates for the same image on the same page
    timestamp = datetime.datetime.now()
    try:
        cursor.execute(sql, (image_url, page_url, warc_path, context_before, context_after, alt_text, 'processed', timestamp))
        return cursor.lastrowid # May be 0 if conflict occurred and nothing was inserted
    except sqlite3.IntegrityError:
         logger.warning(f"Duplicate context skipped: {image_url} on {page_url}")
         return None
    except sqlite3.Error as e:
        logger.error(f"Failed to insert image context for {image_url}: {e}")
        return None

def mark_training_candidate(cursor, context_id, alt_text_used):
    """Marks an ImageContext entry as suitable for training."""
    sql = """
    UPDATE ImageContext
    SET is_training_candidate = 1
    WHERE context_id = ?;
    """
    sql_td = """
    INSERT INTO TrainingData (context_id, alt_text_used, created_timestamp)
    VALUES (?, ?, ?);
    """
    # We add to TrainingData here conceptually, tokenization happens in the training script
    timestamp = datetime.datetime.now()
    try:
        cursor.execute(sql, (context_id,))
        # cursor.execute(sql_td, (context_id, alt_text_used, timestamp)) # Let's decouple this
        logger.debug(f"Marked context_id {context_id} as training candidate.")
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Failed to mark context {context_id} as training candidate: {e}")
        return False

# --- Functions for Stage 2 (Training) ---

def get_training_contexts(cursor, limit=None, offset=0):
    """Fetches ImageContext entries marked as training candidates."""
    sql = """
    SELECT context_id, context_before, context_after, alt_text
    FROM ImageContext
    WHERE is_training_candidate = 1
    """
    params = []
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    try:
        cursor.execute(sql, params)
        return cursor.fetchall() # Returns list of Row objects
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch training contexts: {e}")
        return []

# TrainingData table is populated conceptually; actual token data added during training prep if needed
# or generated on the fly by the data loader.

# --- Functions for Stage 3b (Inference) ---

def get_contexts_for_inference(cursor, limit=None, offset=0):
    """Fetches ImageContext entries that haven't been processed by the prediction model yet."""
    # This fetches all contexts, assuming prediction handles duplicates if needed
    # Or add a status field like 'prediction_status' to ImageContext
    sql = """
    SELECT context_id, context_before, context_after, alt_text, image_url, page_url
    FROM ImageContext
    -- WHERE status_prediction = 'pending' -- Example if adding status
    ORDER BY context_id -- Ensure consistent processing order
    """
    params = []
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    try:
        cursor.execute(sql, params)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch contexts for inference: {e}")
        return []

def add_predicted_description(cursor, context_id, model_path, predicted_text, start_token, end_token, confidences):
    """Adds a prediction result."""
    sql = """
    INSERT INTO PredictedDescriptions (
        context_id, model_path, predicted_text, start_token, end_token,
        start_confidence, end_confidence, average_confidence, span_confidence,
        status_enrichment, prediction_timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    timestamp = datetime.datetime.now()
    try:
        cursor.execute(sql, (
            context_id, model_path, predicted_text, start_token, end_token,
            confidences['start'], confidences['end'], confidences['average'], confidences['span'],
            'pending', timestamp
        ))
        return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Failed to insert prediction for context_id {context_id}: {e}")
        return None

# --- Functions for Stage 4 (Enrichment) ---

def get_predictions_for_enrichment(cursor, limit=None, offset=0):
    """Fetches predictions that need CLIP scoring."""
    sql = """
    SELECT pd.prediction_id, pd.predicted_text, ic.image_url
    FROM PredictedDescriptions pd
    JOIN ImageContext ic ON pd.context_id = ic.context_id
    WHERE pd.status_enrichment = 'pending'
    ORDER BY pd.prediction_id -- Consistent order
    """
    params = []
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    try:
        cursor.execute(sql, params)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch predictions for enrichment: {e}")
        return []

def add_clip_result(cursor, prediction_id, image_local_path, clip_score, clip_model_name, error_message=None):
    """Adds or updates the CLIP enrichment result for a prediction."""
    timestamp = datetime.datetime.now()
    enrich_status = 'clip_scored' if error_message is None else 'failed_clip' # Or 'failed_download' if applicable

    # Insert into ClipResults
    sql_clip = """
    INSERT INTO ClipResults (prediction_id, image_local_path, clip_score, clip_model_name, enrichment_timestamp, error_message)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(prediction_id) DO UPDATE SET
        image_local_path=excluded.image_local_path,
        clip_score=excluded.clip_score,
        clip_model_name=excluded.clip_model_name,
        enrichment_timestamp=excluded.enrichment_timestamp,
        error_message=excluded.error_message;
    """
    # Update status in PredictedDescriptions
    sql_pred_update = """
    UPDATE PredictedDescriptions
    SET status_enrichment = ?
    WHERE prediction_id = ?;
    """
    try:
        cursor.execute(sql_clip, (prediction_id, image_local_path, clip_score, clip_model_name, timestamp, error_message))
        cursor.execute(sql_pred_update, (enrich_status, prediction_id))
        logger.debug(f"Added/Updated CLIP result for prediction_id {prediction_id}, status: {enrich_status}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to add CLIP result for prediction_id {prediction_id}: {e}")
        return False

# --- Functions for Stage 5 (Analysis) ---

def get_final_results(cursor, min_clip_score=None, limit=None):
    """Fetches final results including predictions and CLIP scores."""
    sql = """
    SELECT
        pd.prediction_id,
        ic.image_url,
        ic.page_url,
        cr.image_local_path,
        pd.predicted_text,
        pd.start_confidence,
        pd.end_confidence,
        pd.average_confidence,
        pd.span_confidence,
        cr.clip_score
    FROM PredictedDescriptions pd
    JOIN ImageContext ic ON pd.context_id = ic.context_id
    JOIN ClipResults cr ON pd.prediction_id = cr.prediction_id
    WHERE cr.error_message IS NULL -- Only successful enrichments
    """
    params = []
    if min_clip_score is not None:
        sql += " AND cr.clip_score >= ?"
        params.append(min_clip_score)

    sql += " ORDER BY cr.clip_score DESC" # Example ordering

    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    try:
        cursor.execute(sql, params)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch final results: {e}")
        return []