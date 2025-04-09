# tests/training/test_model.py
import pytest
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys

# Adjust path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from transformers import DistilBertTokenizer
from src.training.model import create_span_prediction_model

# --- Fixtures ---

@pytest.fixture(scope="module")
def tokenizer_info():
    """Provides tokenizer and its vocab size."""
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    special_token = "[IMG]"
    if special_token not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
    return tokenizer, len(tokenizer)

# --- Tests ---

def test_create_model_builds(tokenizer_info):
    """Test if the model can be created without errors."""
    _, vocab_size = tokenizer_info
    max_len = 128
    learning_rate = 5e-5
    try:
        model = create_span_prediction_model(max_len, learning_rate, vocab_size)
        assert isinstance(model, keras.Model)
    except Exception as e:
        pytest.fail(f"create_span_prediction_model raised an exception: {e}")

def test_model_output_shapes(tokenizer_info):
    """Test the output shapes of the model."""
    _, vocab_size = tokenizer_info
    max_len = 128
    learning_rate = 5e-5
    batch_size = 4

    model = create_span_prediction_model(max_len, learning_rate, vocab_size)

    # Create dummy input
    dummy_input_ids = np.random.randint(0, vocab_size, size=(batch_size, max_len), dtype=np.int32)
    dummy_attention_mask = np.ones((batch_size, max_len), dtype=np.int32)
    # Set some padding in mask for realism
    dummy_attention_mask[:, max_len//2:] = 0

    inputs = [tf.constant(dummy_input_ids), tf.constant(dummy_attention_mask)]

    # Predict
    outputs = model(inputs, training=False) # Use training=False for inference mode if needed

    assert isinstance(outputs, list)
    assert len(outputs) == 2 # Start and End logits

    start_logits, end_logits = outputs
    assert isinstance(start_logits, tf.Tensor)
    assert isinstance(end_logits, tf.Tensor)

    # Output shape should be (batch_size, max_len) for flattened logits
    assert start_logits.shape == (batch_size, max_len)
    assert end_logits.shape == (batch_size, max_len)

# Potential future test: Check if embedding layer size matches vocab size after resizing
# Requires inspecting the model layers, might be more involved.
