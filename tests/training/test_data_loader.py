# tests/training/test_data_loader.py
import pytest
import numpy as np
import os
import sys
from unittest.mock import MagicMock

# Adjust path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from transformers import DistilBertTokenizer
from src.training.data_loader import prepare_training_data

# --- Fixtures (remain the same) ---
@pytest.fixture(scope="module")
def tokenizer():
    tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    special_token = "[IMG]"
    if special_token not in tok.additional_special_tokens:
        tok.add_special_tokens({'additional_special_tokens': [special_token]})
    return tok

@pytest.fixture
def sample_db_row_basic():
    return { 'context_id': 1, 'context_before': "Some text before.", 'context_after': "Some text after.", 'alt_text': "the alt text" }
@pytest.fixture
def sample_db_row_long_before():
    return { 'context_id': 2, 'context_before': " ".join([f"word{i}" for i in range(300)]), 'context_after': "Short after.", 'alt_text': "alt text here" }
@pytest.fixture
def sample_db_row_long_after():
    return { 'context_id': 3, 'context_before': "Short before.", 'context_after': " ".join([f"word{i}" for i in range(300)]), 'alt_text': "find this alt" }
@pytest.fixture
def sample_db_row_alt_truncated():
    return { 'context_id': 4, 'context_before': "Very short before.", 'context_after': "Very short after.", 'alt_text': "This alt text is quite long and might be cut off depending on max_len" }

# --- Tests for prepare_training_data ---

def test_prepare_basic(tokenizer, sample_db_row_basic):
    # (No changes needed - should pass now)
    max_len = 64
    db_rows = [sample_db_row_basic]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert x_tokens is not None; assert len(x_tokens) == 1; assert x_tokens.shape == (1, max_len)
    assert x_masks is not None; assert len(x_masks) == 1; assert x_masks.shape == (1, max_len)
    assert y_starts is not None; assert len(y_starts) == 1
    assert y_ends is not None; assert len(y_ends) == 1
    before_tokens = tokenizer.encode(sample_db_row_basic['context_before'], add_special_tokens=False)
    alt_tokens = tokenizer.encode(sample_db_row_basic['alt_text'], add_special_tokens=False)
    expected_start = 1 + len(before_tokens)
    expected_end = expected_start + len(alt_tokens) - 1
    assert y_starts[0] == expected_start
    assert y_ends[0] == expected_end
    assert x_masks[0, expected_end] == 1
    assert x_masks[0, -1] == 0

def test_prepare_truncation_before(tokenizer, sample_db_row_long_before):
    # (No changes needed - should pass now)
    max_len = 128
    db_rows = [sample_db_row_long_before]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert x_tokens is not None and len(x_tokens) == 1
    assert x_tokens.shape == (1, max_len); assert x_masks.shape == (1, max_len)
    assert np.sum(x_masks[0]) <= max_len
    assert 0 <= y_starts[0] < max_len; assert 0 <= y_ends[0] < max_len
    assert y_starts[0] <= y_ends[0]

def test_prepare_truncation_after(tokenizer, sample_db_row_long_after):
    # (No changes needed - should pass now)
    max_len = 128
    db_rows = [sample_db_row_long_after]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert x_tokens is not None and len(x_tokens) == 1
    assert x_tokens.shape == (1, max_len); assert x_masks.shape == (1, max_len)
    assert np.sum(x_masks[0]) <= max_len
    assert 0 <= y_starts[0] < max_len; assert 0 <= y_ends[0] < max_len
    assert y_starts[0] <= y_ends[0]

def test_prepare_alt_gets_truncated(tokenizer, sample_db_row_alt_truncated):
    # (No changes needed - should pass now)
    max_len = 32
    db_rows = [sample_db_row_alt_truncated]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert len(x_tokens) == 0

def test_prepare_empty_input(tokenizer):
    max_len = 64
    db_rows = []
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    # --- Start Correction ---
    # Check for empty arrays, not None
    assert isinstance(x_tokens, np.ndarray) and len(x_tokens) == 0
    assert isinstance(x_masks, np.ndarray) and len(x_masks) == 0
    assert isinstance(y_starts, np.ndarray) and len(y_starts) == 0
    assert isinstance(y_ends, np.ndarray) and len(y_ends) == 0
    # --- End Correction ---

    # Test row with empty alt_text
    db_rows = [{ 'context_id': 6, 'context_before': "Some", 'context_after': "Text", 'alt_text': ""}]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert len(x_tokens) == 0 # Should be rejected

    # Test row with empty context but valid alt_text
    db_rows = [{ 'context_id': 7, 'context_before': "", 'context_after': "", 'alt_text': "valid alt"}]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert len(x_tokens) == 1 # Should be processed
    assert y_starts[0] == 1
    alt_tokens = tokenizer.encode("valid alt", add_special_tokens=False)
    assert y_ends[0] == 1 + len(alt_tokens) - 1

