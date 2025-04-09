# tests/training/test_model.py
import pytest
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import tempfile

# Adjust path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering, TFAutoModelForQuestionAnswering
# Import the creator function
from src.training.model import create_span_prediction_model

# --- Fixtures ---
@pytest.fixture(scope="module")
def tokenizer_info():
    """Provides tokenizer and its vocab size."""
    # Use the specific checkpoint name expected by the model creator
    checkpoint = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
    special_token = "[IMG]"
    if special_token not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
    return tokenizer, len(tokenizer), checkpoint

# --- Tests ---

def test_create_model_loads(tokenizer_info):
    """Test if the QA model can be loaded without errors."""
    tokenizer, vocab_size, checkpoint = tokenizer_info
    max_len = 128
    try:
        # Pass necessary args (learning_rate isn't used by create_... anymore)
        model = create_span_prediction_model(max_len, 0.0, vocab_size, model_checkpoint=checkpoint)
        # Check type
        assert isinstance(model, TFDistilBertForQuestionAnswering)
        # Check if embeddings were resized (optional)
        base_model = getattr(model, model.base_model_prefix, None)
        if base_model:
             assert base_model.config.vocab_size == vocab_size
        else:
             assert model.config.vocab_size == vocab_size

    except Exception as e:
        pytest.fail(f"create_span_prediction_model raised an exception: {e}")

def test_model_output_shapes_and_type(tokenizer_info):
    """Test the output shapes and type of the QA model."""
    tokenizer, vocab_size, checkpoint = tokenizer_info
    max_len = 128
    batch_size = 4

    model = create_span_prediction_model(max_len, 0.0, vocab_size, model_checkpoint=checkpoint)

    # Create dummy input
    dummy_input_ids = tf.constant(np.random.randint(0, vocab_size, size=(batch_size, max_len), dtype=np.int32))
    dummy_attention_mask = tf.constant(np.ones((batch_size, max_len), dtype=np.int32))

    # Call model using dictionary input (standard for HF models)
    inputs = {'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask}
    outputs = model(inputs) # training=False is default

    # Output is a specific HF output object
    assert hasattr(outputs, 'start_logits')
    assert hasattr(outputs, 'end_logits')
    assert isinstance(outputs.start_logits, tf.Tensor)
    assert isinstance(outputs.end_logits, tf.Tensor)
    assert outputs.start_logits.shape == (batch_size, max_len)
    assert outputs.end_logits.shape == (batch_size, max_len)

# --- New Test for save_pretrained / from_pretrained ---
def test_model_save_load_pretrained(tokenizer_info, tmp_path):
    """Tests saving and reloading using save_pretrained/from_pretrained."""
    tokenizer, vocab_size, checkpoint = tokenizer_info
    max_len = 64

    # 1. Create the model
    model = create_span_prediction_model(max_len, 0.0, vocab_size, model_checkpoint=checkpoint)

    # 2. Define save directory
    save_directory = tmp_path / "hf_qa_model"
    save_directory_str = str(save_directory)

    # 3. Save model and tokenizer
    try:
        model.save_pretrained(save_directory_str)
        tokenizer.save_pretrained(save_directory_str)
        print(f"Model and tokenizer saved to {save_directory_str}")
        assert os.path.exists(os.path.join(save_directory_str, "tf_model.h5")) # Or config.json etc.
        assert os.path.exists(os.path.join(save_directory_str, "tokenizer_config.json"))
    except Exception as e:
        pytest.fail(f"save_pretrained failed: {e}")

    # 4. Reload model and tokenizer
    try:
        loaded_model = TFAutoModelForQuestionAnswering.from_pretrained(save_directory_str)
        loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_directory_str)
        print(f"Model and tokenizer loaded from {save_directory_str}")
        assert loaded_model is not None
        assert loaded_tokenizer is not None
        # Check if tokenizer has the special token
        assert "[IMG]" in loaded_tokenizer.get_vocab()
        # Check if model vocab size matches reloaded tokenizer
        base_model = getattr(loaded_model, loaded_model.base_model_prefix, None)
        if base_model: assert base_model.config.vocab_size == len(loaded_tokenizer)
        else: assert loaded_model.config.vocab_size == len(loaded_tokenizer)

    except Exception as e:
        pytest.fail(f"from_pretrained failed: {e}")

    # 5. Check inference with reloaded model (optional basic check)
    dummy_input_ids = tf.constant(np.random.randint(0, vocab_size, size=(2, max_len), dtype=np.int32))
    dummy_attention_mask = tf.constant(np.ones((2, max_len), dtype=np.int32))
    inputs = {'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask}
    try:
        outputs = loaded_model(inputs)
        assert hasattr(outputs, 'start_logits') and hasattr(outputs, 'end_logits')
        assert outputs.start_logits.shape == (2, max_len)
        print("Prediction successful with loaded model.")
    except Exception as e:
        pytest.fail(f"Prediction with loaded model failed: {e}")

