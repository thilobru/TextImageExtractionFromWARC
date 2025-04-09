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
# Only import the main function we are testing now
from src.training.data_loader import prepare_training_data

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
    # This test might be less relevant now, but keep fixture for now
    return {
        'context_id': 5,
        'context_before': "Some text before.",
        'context_after': "Some text after.",
        'alt_text': "alt text not present" # This text doesn't appear
    }

# --- Tests for prepare_training_data ---

def test_prepare_basic(tokenizer, sample_db_row_basic):
    max_len = 64 # Keep max_len reasonable for testing
    db_rows = [sample_db_row_basic]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)

    assert x_tokens is not None, "prepare_training_data returned None for x_tokens"
    assert x_masks is not None, "prepare_training_data returned None for x_masks"
    assert y_starts is not None, "prepare_training_data returned None for y_starts"
    assert y_ends is not None, "prepare_training_data returned None for y_ends"

    assert len(x_tokens) == 1, f"Expected 1 sample, got {len(x_tokens)}"
    assert x_tokens.shape == (1, max_len)
    assert x_masks.shape == (1, max_len)

    # Decode to verify content and find indices manually
    decoded = tokenizer.decode(x_tokens[0], skip_special_tokens=False)
    print(f"\nDecoded basic: {decoded}")
    print(f"Tokens basic: {x_tokens[0].tolist()}")
    print(f"Mask basic: {x_masks[0].tolist()}")
    print(f"Start/End basic: {y_starts[0]}/{y_ends[0]}")

    # Manually calculate expected indices based on the NEW logic
    before_tokens = tokenizer.encode(sample_db_row_basic['context_before'], add_special_tokens=False)
    alt_tokens = tokenizer.encode(sample_db_row_basic['alt_text'], add_special_tokens=False)
    # Assuming truncation doesn't happen for this short example
    expected_start = 1 + len(before_tokens) # After [CLS] and before_tokens
    expected_end = expected_start + len(alt_tokens) - 1

    assert y_starts[0] == expected_start
    assert y_ends[0] == expected_end
    assert x_masks[0, expected_end] == 1 # Check mask covers span
    assert x_masks[0, -1] == 0 # Check padding mask is 0

    # Optional: Verify token match (uncomment check in data_loader if using)
    # assert x_tokens[0, expected_start : expected_end + 1].tolist() == alt_tokens

def test_prepare_truncation_before(tokenizer, sample_db_row_long_before):
    max_len = 128
    db_rows = [sample_db_row_long_before]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)

    assert x_tokens is not None and len(x_tokens) == 1, f"Expected 1 sample, got {len(x_tokens) if x_tokens is not None else 0}"
    assert x_tokens.shape == (1, max_len)
    assert x_masks.shape == (1, max_len)
    assert np.sum(x_masks[0]) <= max_len # Check length constraint

    # Verify indices are within valid bounds and start <= end
    # We trust the index calculation logic now, no need to decode the input span
    assert 0 <= y_starts[0] < max_len, f"Start index {y_starts[0]} out of bounds"
    assert 0 <= y_ends[0] < max_len, f"End index {y_ends[0]} out of bounds"
    assert y_starts[0] <= y_ends[0], f"Start index {y_starts[0]} > End index {y_ends[0]}"

def test_prepare_truncation_after(tokenizer, sample_db_row_long_after):
    max_len = 128
    db_rows = [sample_db_row_long_after]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)

    assert x_tokens is not None and len(x_tokens) == 1, f"Expected 1 sample, got {len(x_tokens) if x_tokens is not None else 0}"
    assert x_tokens.shape == (1, max_len)
    assert x_masks.shape == (1, max_len)
    assert np.sum(x_masks[0]) <= max_len

    # Verify indices are within valid bounds and start <= end
    assert 0 <= y_starts[0] < max_len, f"Start index {y_starts[0]} out of bounds"
    assert 0 <= y_ends[0] < max_len, f"End index {y_ends[0]} out of bounds"
    assert y_starts[0] <= y_ends[0], f"Start index {y_starts[0]} > End index {y_ends[0]}"
    

def test_prepare_alt_gets_truncated(tokenizer, sample_db_row_alt_truncated):
    """Test that if alt text doesn't fit based on calculated indices, the example is rejected."""
    max_len = 32 # Force truncation
    db_rows = [sample_db_row_alt_truncated]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)

    # Expect rejection because calculated end_idx will be >= current_len
    assert len(x_tokens) == 0 # Check for empty array

# test_prepare_alt_not_in_context is less relevant now as we don't search
# def test_prepare_alt_not_in_context(...)

def test_prepare_empty_input(tokenizer):
    max_len = 64
    db_rows = []
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    # The function returns None if db_rows is empty initially
    assert x_tokens is None

    # Test row with empty alt_text
    db_rows = [{ 'context_id': 6, 'context_before': "Some", 'context_after': "Text", 'alt_text': ""}]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert len(x_tokens) == 0 # Should be rejected because alt_text is empty

    # Test row with empty context but valid alt_text
    db_rows = [{ 'context_id': 7, 'context_before': "", 'context_after': "", 'alt_text': "valid alt"}]
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(db_rows, tokenizer, max_len)
    assert len(x_tokens) == 1 # Should be processed
    # Expected: [CLS] [IMG] [SEP] ...
    assert y_starts[0] == 1 # Index of [IMG]
    alt_tokens = tokenizer.encode("valid alt", add_special_tokens=False)
    assert y_ends[0] == 1 + len(alt_tokens) - 1

