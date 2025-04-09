# src/training/model.py
import tensorflow as tf
from tensorflow import keras # Keep for type hints if needed elsewhere
# --- Start Change ---
# Use TFAutoModelForQuestionAnswering for flexibility or specific class
from transformers import TFAutoModelForQuestionAnswering, TFDistilBertForQuestionAnswering, DistilBertTokenizer
# --- End Change ---
import logging

logger = logging.getLogger(__name__)

# Remove the SpanPredictionModel class definition

def create_span_prediction_model(max_len, learning_rate, tokenizer_vocab_size,
                                 model_checkpoint="distilbert-base-uncased"):
    """
    Loads a pre-trained Transformer model for Question Answering.

    Args:
        max_len (int): Maximum sequence length (used for info, not directly here).
        learning_rate (float): Optimizer learning rate (compilation happens in train.py).
        tokenizer_vocab_size (int): Size of the tokenizer vocabulary (incl. special tokens).
        model_checkpoint (str): The Hugging Face model identifier.

    Returns:
        TFPreTrainedModel: Loaded Hugging Face model for Question Answering.
                           Model is NOT compiled here.
    """
    logger.info(f"Loading pre-trained QA model: {model_checkpoint}")
    try:
        # Load the specified QA model
        # Use the specific class for clarity or TFAutoModelForQuestionAnswering
        model = TFDistilBertForQuestionAnswering.from_pretrained(model_checkpoint)

        # --- Handle Tokenizer Resizing ---
        # Get the underlying base model (e.g., model.distilbert) to resize embeddings
        base_model = getattr(model, model.base_model_prefix, None)
        if base_model and hasattr(base_model, 'resize_token_embeddings'):
             if base_model.config.vocab_size != tokenizer_vocab_size:
                  base_model.resize_token_embeddings(tokenizer_vocab_size)
                  # Also update the main model's config if necessary
                  model.config.vocab_size = tokenizer_vocab_size
                  logger.info(f"Resized model token embeddings to {tokenizer_vocab_size}")
        elif model.config.vocab_size != tokenizer_vocab_size:
             # Fallback if base_model attribute isn't standard (less common for TF)
             logger.warning("Could not directly access base model to resize embeddings, attempting on main model.")
             try:
                  model.resize_token_embeddings(tokenizer_vocab_size)
                  logger.info(f"Resized model token embeddings to {tokenizer_vocab_size}")
             except AttributeError:
                   logger.error("Failed to resize token embeddings. Ensure model architecture supports it or handle manually.")
                   raise RuntimeError("Embedding resizing failed.") # Make it a fatal error

    except Exception as e:
        logger.error(f"Failed to load/resize pre-trained QA model '{model_checkpoint}': {e}")
        raise

    # Note: Compilation (optimizer, loss) is now handled in the training script
    logger.info(f"Successfully loaded QA model: {model_checkpoint}")
    model.summary(print_fn=logger.info) # Print summary

    return model