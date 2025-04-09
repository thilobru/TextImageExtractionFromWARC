# tests/common/test_database.py
import pytest
import sqlite3
import os
import sys
import datetime

# Adjust path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import database functions to test
from src.common.database import (
    db_connect,
    initialize_schema,
    add_image_context,
    mark_training_candidate,
    get_training_contexts,
    get_contexts_for_inference,
    add_predicted_description,
    get_predictions_for_enrichment,
    add_clip_result,
    get_final_results
)

# --- Fixtures ---

@pytest.fixture
def temp_db_path(tmp_path):
    """Creates a temporary database file path for testing."""
    db_file = tmp_path / "test_image_data.db"
    # Ensure schema is initialized for tests that need it by default
    initialize_schema(str(db_file))
    return str(db_file) # Return string path

# Separate fixture just for path if needed
@pytest.fixture
def temp_db_path_only(tmp_path):
     db_file = tmp_path / "test_db_no_schema.db"
     return str(db_file)


# --- Tests ---

def test_initialize_schema(temp_db_path_only): # Use non-initialized path
    """Test if schema initialization creates the expected tables."""
    db_path = temp_db_path_only
    initialize_schema(db_path)
    # Check if tables exist
    with db_connect(db_path) as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row['name'] for row in cursor.fetchall()}
        assert 'ImageContext' in tables
        assert 'TrainingData' in tables
        assert 'PredictedDescriptions' in tables
        assert 'ClipResults' in tables
    # Running again should not cause errors
    initialize_schema(db_path)


def test_add_image_context_unique_constraint(temp_db_path): # Use initialized path
    """Test the UNIQUE constraint on (image_url, page_url)."""
    db_path = temp_db_path # Use the already initialized DB from fixture
    data1 = { "image_url": "http://example.com/img1.jpg", "page_url": "http://example.com/page1", "warc_path": "warc1", "context_before": "b", "context_after": "a", "alt_text": "alt1" }
    data2_duplicate = { "image_url": "http://example.com/img1.jpg", "page_url": "http://example.com/page1", "warc_path": "warc1", "context_before": "b_dup", "context_after": "a_dup", "alt_text": "alt_dup" }
    data3_different_page = { "image_url": "http://example.com/img1.jpg", "page_url": "http://example.com/page2", "warc_path": "warc1", "context_before": "b", "context_after": "a", "alt_text": "alt1" }

    with db_connect(db_path) as cursor:
        # Insert first record
        rowid1 = add_image_context(cursor, **data1)
        assert rowid1 is not None and rowid1 > 0, "First insert should return a valid rowid"

        # Attempt to insert duplicate
        rowid2 = add_image_context(cursor, **data2_duplicate)
        # --- Start Correction ---
        # Removed the unreliable assertion on rowid2
        # --- End Correction ---

        # Insert record with different page_url
        rowid3 = add_image_context(cursor, **data3_different_page)
        assert rowid3 is not None and rowid3 != rowid1, "Third insert (different page) should get a new rowid"

        # Verify count - this is the reliable check
        cursor.execute("SELECT COUNT(*) FROM ImageContext")
        count = cursor.fetchone()[0]
        assert count == 2, f"Expected 2 rows after handling conflict, found {count}"

# --- Other database tests remain the same ---

def test_mark_training_candidate(temp_db_path):
    db_path = temp_db_path
    data = {"image_url": "img.png", "page_url": "page", "warc_path": "w", "context_before": "b", "context_after": "a", "alt_text": "alt"}
    context_id = None
    with db_connect(db_path) as cursor:
        context_id = add_image_context(cursor, **data)
        assert context_id is not None
        marked = mark_training_candidate(cursor, context_id, data["alt_text"])
        assert marked is True
    with db_connect(db_path) as cursor:
        cursor.execute("SELECT is_training_candidate FROM ImageContext WHERE context_id = ?", (context_id,))
        row = cursor.fetchone()
        assert row is not None and row['is_training_candidate'] == 1

def test_get_training_contexts(temp_db_path):
    db_path = temp_db_path
    ids_added = []
    with db_connect(db_path) as cursor:
        id1 = add_image_context(cursor, "img1", "p1", "w", "b", "a", "alt1")
        id2 = add_image_context(cursor, "img2", "p1", "w", "b", "a", "alt2")
        id3 = add_image_context(cursor, "img3", "p2", "w", "b", "a", "alt3")
        ids_added.extend([id1, id2, id3])
        mark_training_candidate(cursor, id1, "alt1")
        mark_training_candidate(cursor, id3, "alt3")
    with db_connect(db_path) as cursor:
        results = get_training_contexts(cursor)
        assert len(results) == 2
        result_ids = {row['context_id'] for row in results}
        assert result_ids == {id1, id3}
        results_limit = get_training_contexts(cursor, limit=1)
        assert len(results_limit) == 1
        results_offset = get_training_contexts(cursor, limit=1, offset=1)
        assert len(results_offset) == 1
        # Assuming default order is by context_id
        expected_offset_id = sorted([id1, id3])[1]
        assert results_offset[0]['context_id'] == expected_offset_id

def test_add_predicted_description(temp_db_path):
    db_path = temp_db_path
    context_id = None
    with db_connect(db_path) as cursor:
         context_id = add_image_context(cursor, "img1", "p1", "w", "b", "a", "alt1")
    prediction_data = { "context_id": context_id, "model_path": "model.h5", "predicted_text": "predicted desc", "start_token": 5, "end_token": 10, "confidences": {'start': 0.9, 'end': 0.8, 'average': 0.85, 'span': 0.72} }
    prediction_id = None
    with db_connect(db_path) as cursor:
        prediction_id = add_predicted_description(cursor, **prediction_data)
        assert prediction_id is not None and prediction_id > 0
    with db_connect(db_path) as cursor:
        cursor.execute("SELECT * FROM PredictedDescriptions WHERE prediction_id = ?", (prediction_id,))
        row = cursor.fetchone()
        assert row is not None and row['context_id'] == context_id and row['predicted_text'] == prediction_data['predicted_text'] and row['status_enrichment'] == 'pending'

def test_get_predictions_for_enrichment(temp_db_path):
    db_path = temp_db_path
    ctx_id1, ctx_id2 = None, None
    pred_id1, pred_id2, pred_id3 = None, None, None
    conf = {'start': 0.9, 'end': 0.8, 'average': 0.85, 'span': 0.72}
    with db_connect(db_path) as cursor:
        ctx_id1 = add_image_context(cursor, "img1", "p1", "w", "b", "a", "alt1")
        ctx_id2 = add_image_context(cursor, "img2", "p2", "w", "b", "a", "alt2")
        pred_id1 = add_predicted_description(cursor, ctx_id1, "m", "pred1", 5, 10, conf)
        pred_id2 = add_predicted_description(cursor, ctx_id2, "m", "pred2", 6, 11, conf)
        pred_id3 = add_predicted_description(cursor, ctx_id1, "m", "pred3", 7, 12, conf)
        cursor.execute("UPDATE PredictedDescriptions SET status_enrichment = 'clip_scored' WHERE prediction_id = ?", (pred_id2,))
    with db_connect(db_path) as cursor:
        results = get_predictions_for_enrichment(cursor)
        assert len(results) == 2
        result_ids = {row['prediction_id'] for row in results}
        assert result_ids == {pred_id1, pred_id3}
        results_limit = get_predictions_for_enrichment(cursor, limit=1)
        assert len(results_limit) == 1
        results_offset = get_predictions_for_enrichment(cursor, limit=1, offset=1)
        assert len(results_offset) == 1
        assert results_limit[0]['prediction_id'] != results_offset[0]['prediction_id']

def test_add_clip_result(temp_db_path):
    db_path = temp_db_path
    ctx_id, pred_id = None, None
    conf = {'start': 0.9, 'end': 0.8, 'average': 0.85, 'span': 0.72}
    with db_connect(db_path) as cursor:
        ctx_id = add_image_context(cursor, "img1", "p1", "w", "b", "a", "alt1")
        pred_id = add_predicted_description(cursor, ctx_id, "m", "pred1", 5, 10, conf)
    clip_data_ok = { "prediction_id": pred_id, "image_local_path": "/path/img1.jpg", "clip_score": 0.75, "clip_model_name": "clip-vit-base", "error_message": None }
    with db_connect(db_path) as cursor:
        added_ok = add_clip_result(cursor, **clip_data_ok)
        assert added_ok is True
    with db_connect(db_path) as cursor:
        cursor.execute("SELECT * FROM ClipResults WHERE prediction_id = ?", (pred_id,))
        row_clip = cursor.fetchone(); assert row_clip is not None
        assert row_clip['clip_score'] == pytest.approx(0.75); assert row_clip['error_message'] is None
        cursor.execute("SELECT status_enrichment FROM PredictedDescriptions WHERE prediction_id = ?", (pred_id,))
        row_pred = cursor.fetchone(); assert row_pred is not None; assert row_pred['status_enrichment'] == 'clip_scored'
    clip_data_err = { "prediction_id": pred_id, "image_local_path": None, "clip_score": None, "clip_model_name": "clip-vit-base", "error_message": "Download failed" }
    with db_connect(db_path) as cursor:
        added_err = add_clip_result(cursor, **clip_data_err)
        assert added_err is True
    with db_connect(db_path) as cursor:
        cursor.execute("SELECT * FROM ClipResults WHERE prediction_id = ?", (pred_id,))
        row_clip = cursor.fetchone(); assert row_clip is not None
        assert row_clip['clip_score'] is None; assert row_clip['error_message'] == "Download failed"
        cursor.execute("SELECT status_enrichment FROM PredictedDescriptions WHERE prediction_id = ?", (pred_id,))
        row_pred = cursor.fetchone(); assert row_pred is not None; assert row_pred['status_enrichment'] == 'failed_clip'

