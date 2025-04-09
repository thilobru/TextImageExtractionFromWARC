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
from src.training.data_loader import prepare_training_data, find_sublist_indices

# --- Fixtures ---

@pytest.fixture(scope="module")
def tokenizer():
    """Provides a DistilBertTokenizer instance with the special [IMG] token."""
    tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    special_token = "[IMG]"
    if special_token not in tok.additional_special_tokens:
        tok.add_special_tokens({'additional_special_tokens': [special_token]})
    return tok

@pytest.fixture
def sample_db_row_basic():
    """A basic sample row mimicking database output."""
    return {
        'context_id': 1,
        'context_before': "Some text before.",
        'context_after': "Some text after.",
        'alt_text': "the alt text" # Target span
    }

@pytest.fixture
def sample_db_row_long_before():
    """Sample row with long before context causing truncation."""
    return {
        'context_id': 2,
        'context_before': " ".join([f"word{i}" for i in range(300)]), # Long before
        'context_after': "Short after.",
        'alt_text': "alt text here"
    }

@pytest.fixture
def sample_db_row_long_after():
    """Sample row with long after context causing truncation."""
    return {
        'context_id': 3,
        'context_before': "Short before.",
        'context_after': " ".join([f"word{i}" for i in range(300)]), # Long after
        'alt_text': "find this alt"
    }

@pytest.fixture
def sample_db_row_alt_truncated():
    """Sample row where alt text might get truncated."""
    return {
        'context_id': 4,
        'context_before': "Very short before.",
        'context_after': "Very short after.",
        'alt_text': "This alt text is quite long and might be cut off depending on max_len"
    }

@pytest.fixture
def sample_db_row_no_alt_found():
    """Sample row where alt text isn't actually in context (for find_sublist testing)."""
    # Note: prepare_training_data relies on find_sublist_indices internally
    return {
        'context_id': 5,
        'context_before': "Some text before.",
        'context_after': "Some text after.",
        'alt_text': "alt text not present" # This text doesn't appear
    }

# --- Tests for find_sublist_indices ---

def test_find_sublist_found():
    assert find_sublist_indices([1, 2, 3, 4, 5], [3, 4]) == (2, 3)
    assert find_sublist_indices([1, 2, 3, 4, 3, 4, 5], [3, 4]) == (2, 3) # First occurrence

def test_find_sublist_not_found():
    assert find_sublist_indices([1, 2, 3, 4, 5], [6, 7]) == (None, None)

def test_find_sublist_empty_sublist():
    assert find_sublist_indices([1, 2, 3], []) == (None, None)

def test_find_sublist_empty_mainlist():
     assert find_sublist_indices([], [1]) == (None, None)

# --- Tests for prepare_training_data ---

def test_prepare_basic(tokenizer, sample_db_row_basic):
    max_len = 64 # Keep max_len reasonable for testing
    db_rows = [sample_db_row_basic]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)

    assert x_tokens is not None and len(x_tokens) == 1
    assert x_masks is not None and len(x_masks) == 1
    assert y_starts is not None and len(y_starts) == 1
    assert y_ends is not None and len(y_ends) == 1

    # Decode to verify content and find indices manually
    # [CLS] Some text before . [IMG] Some text after . [SEP] [PAD]...
    # Expected tokens (approx): [101, 2070, 3793, 2077, 1012, *IMG_ID*, 2070, 3793, 2044, 1012, 102]
    # Expected alt tokens: [1996, 7561, 3793] ('the', 'alt', 'text')
    decoded = tokenizer.decode(x_tokens[0], skip_special_tokens=False)
    print(f"\nDecoded basic: {decoded}")
    print(f"Tokens basic: {x_tokens[0].tolist()}")
    print(f"Mask basic: {x_masks[0].tolist()}")

    # Manually find expected indices (this depends heavily on exact tokenization)
    # Example: If tokens are [101, 2070, 3793, 2077, 1012, *IMG*, 1996, 7561, 3793, 2070, 3793, 2044, 1012, 102, 0...]
    # And alt is [1996, 7561, 3793] ('the', 'alt', 'text')
    # Then start should be index 6, end should be index 8
    # This needs verification based on actual tokenizer output
    alt_tokens = tokenizer.encode(sample_db_row_basic['alt_text'], add_special_tokens=False)
    start_expected, end_expected = find_sublist_indices(x_tokens[0].tolist(), alt_tokens)

    assert start_expected is not None, "Alt tokens not found in prepared tokens"
    assert y_starts[0] == start_expected
    assert y_ends[0] == end_expected
    assert x_tokens.shape == (1, max_len)
    assert x_masks.shape == (1, max_len)
    assert x_masks[0, end_expected] == 1 # Check mask covers span
    assert x_masks[0, -1] == 0 # Check padding mask is 0

def test_prepare_truncation_before(tokenizer, sample_db_row_long_before):
    max_len = 128
    db_rows = [sample_db_row_long_before]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)

    assert x_tokens is not None and len(x_tokens) == 1
    # Check length constraint
    assert np.sum(x_masks[0]) <= max_len

    # Decode and check if alt text is present and indices are correct
    decoded = tokenizer.decode(x_tokens[0, y_starts[0]:y_ends[0]+1])
    print(f"\nDecoded truncated before span: {decoded}")
    assert sample_db_row_long_before['alt_text'] in decoded # Basic check

    # Verify indices are within bounds
    assert 0 <= y_starts[0] < max_len
    assert 0 <= y_ends[0] < max_len
    assert y_starts[0] <= y_ends[0]

def test_prepare_truncation_after(tokenizer, sample_db_row_long_after):
    max_len = 128
    db_rows = [sample_db_row_long_after]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)

    assert x_tokens is not None and len(x_tokens) == 1
    assert np.sum(x_masks[0]) <= max_len

    decoded = tokenizer.decode(x_tokens[0, y_starts[0]:y_ends[0]+1])
    print(f"\nDecoded truncated after span: {decoded}")
    assert sample_db_row_long_after['alt_text'] in decoded

    assert 0 <= y_starts[0] < max_len
    assert 0 <= y_ends[0] < max_len
    assert y_starts[0] <= y_ends[0]

def test_prepare_alt_gets_truncated(tokenizer, sample_db_row_alt_truncated):
    """Test that if alt text doesn't fit, the example is rejected."""
    max_len = 32 # Force truncation
    db_rows = [sample_db_row_alt_truncated]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)

    # Expect rejection because alt_tokens likely won't be found in truncated sequence
    assert x_tokens is None or len(x_tokens) == 0

def test_prepare_alt_not_in_context(tokenizer, sample_db_row_no_alt_found):
    """Test rejection if alt_tokens aren't found (simulates check)."""
    max_len = 64
    db_rows = [sample_db_row_no_alt_found]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)

    # Expect rejection because find_sublist_indices should fail
    assert x_tokens is None or len(x_tokens) == 0

def test_prepare_empty_input(tokenizer):
    max_len = 64
    db_rows = []
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert x_tokens is None

    db_rows = [{ 'context_id': 6, 'context_before': "", 'context_after': "", 'alt_text': ""}]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert x_tokens is None or len(x_tokens) == 0 # Should be rejected
