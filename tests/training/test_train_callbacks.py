# tests/training/test_train_callbacks.py
import pytest
import numpy as np
import os
import sys
from unittest.mock import MagicMock, PropertyMock

# Adjust path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import callbacks from train module
from src.training.train import ExactMatch, IoUCallback

# --- Fixtures ---

@pytest.fixture
def mock_model():
    """Creates a mock Keras model with a predict method."""
    model = MagicMock()
    # Define what model.predict returns based on input size
    def mock_predict(inputs):
        input_ids, _ = inputs # We only need shape from input_ids
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        # Return dummy logits (shape: batch_size, seq_len)
        # Make values small so argmax is predictable
        start_logits = np.random.rand(batch_size, seq_len) * 0.1
        end_logits = np.random.rand(batch_size, seq_len) * 0.1
        # Set predictable argmax for testing
        if batch_size >= 1:
            start_logits[0, 5] = 1.0 # Make index 5 max for first sample start
            end_logits[0, 10] = 1.0 # Make index 10 max for first sample end
        if batch_size >= 2:
            start_logits[1, 2] = 1.0 # index 2 for second sample start
            end_logits[1, 2] = 1.0 # index 2 for second sample end (exact match, IoU=1)
        if batch_size >= 3:
             start_logits[2, 7] = 1.0 # index 7
             end_logits[2, 9] = 1.0 # index 9
        return [start_logits, end_logits]

    model.predict = MagicMock(side_effect=mock_predict)
    return model

@pytest.fixture
def validation_data():
    """Creates sample validation data."""
    batch_size = 3
    max_len = 20
    # Dummy inputs (content doesn't matter, just shape)
    input_ids = np.zeros((batch_size, max_len), dtype=np.int32)
    attn_mask = np.ones((batch_size, max_len), dtype=np.int32)
    # True labels matching the predictable argmax in mock_model
    true_starts = np.array([5, 2, 7], dtype=np.int32)
    true_ends = np.array([10, 2, 8], dtype=np.int32) # Note: end for sample 3 is 8 (off by 1 from pred)
    return ([input_ids, attn_mask], [true_starts, true_ends])

# --- Tests for ExactMatch ---

def test_exact_match_callback(mock_model, validation_data):
    history_log = {}
    callback = ExactMatch(validation_data, history_log)
    callback.model = mock_model # Assign the mock model

    logs = {}
    callback.on_epoch_end(0, logs=logs)

    # Predictions: (start, end) -> (5, 10), (2, 2), (7, 9)
    # True Labels: (start, end) -> (5, 10), (2, 2), (7, 8)
    # Matches: Sample 0 (Yes), Sample 1 (Yes), Sample 2 (No)
    # Expected Accuracy = 2 / 3
    expected_acc = 2.0 / 3.0

    assert 'val_exact_match' in logs
    assert logs['val_exact_match'] == pytest.approx(expected_acc)
    assert 'val_exact_match' in history_log
    assert history_log['val_exact_match'] == [pytest.approx(expected_acc)]

# --- Tests for IoUCallback ---

def test_iou_callback(mock_model, validation_data):
    history_log = {}
    callback = IoUCallback(validation_data, history_log)
    callback.model = mock_model # Assign the mock model

    logs = {}
    callback.on_epoch_end(0, logs=logs)

    # Predictions: (start, end) -> (5, 10), (2, 2), (7, 9)
    # True Labels: (start, end) -> (5, 10), (2, 2), (7, 8)

    # IoU Calculation:
    # Sample 0: Pred=[5..10], True=[5..10]. Inter=6, Union=6. IoU = 1.0
    # Sample 1: Pred=[2..2], True=[2..2]. Inter=1, Union=1. IoU = 1.0
    # Sample 2: Pred=[7..9], True=[7..8]. Inter=2 (7,8), Union=3 (7,8,9). IoU = 2/3
    # Average IoU = (1.0 + 1.0 + 2/3) / 3 = (8/3) / 3 = 8/9
    expected_iou = 8.0 / 9.0

    assert 'val_iou' in logs
    assert logs['val_iou'] == pytest.approx(expected_iou)
    assert 'val_iou' in history_log
    assert history_log['val_iou'] == [pytest.approx(expected_iou)]

def test_iou_callback_invalid_span(mock_model, validation_data):
    """Test IoU calculation when prediction or label is invalid (start > end)."""
    history_log = {}
    # Modify validation data to have an invalid true label
    invalid_val_data = (validation_data[0], [validation_data[1][0], np.array([10, 5, 8])]) # end < start for sample 1

    callback = IoUCallback(invalid_val_data, history_log)
    callback.model = mock_model

    logs = {}
    callback.on_epoch_end(0, logs=logs)

    # IoU Calculation:
    # Sample 0: Pred=[5..10], True=[5..10]. IoU = 1.0
    # Sample 1: Pred=[2..2], True=[invalid]. IoU = 0.0 (as per callback logic)
    # Sample 2: Pred=[7..9], True=[7..8]. IoU = 2/3
    # Average IoU = (1.0 + 0.0 + 2/3) / 3 = (5/3) / 3 = 5/9
    expected_iou = 5.0 / 9.0

    assert 'val_iou' in logs
    assert logs['val_iou'] == pytest.approx(expected_iou)

