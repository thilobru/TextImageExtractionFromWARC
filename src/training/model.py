# src/training/model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFDistilBertModel, DistilBertTokenizer
import logging

logger = logging.getLogger(__name__)

def create_span_prediction_model(max_len, learning_rate, tokenizer_vocab_size):
    """
    Creates the DistilBERT model for span prediction.

    Args:
        max_len (int): Maximum sequence length.
        learning_rate (float): Optimizer learning rate.
        tokenizer_vocab_size (int): Size of the tokenizer vocabulary (incl. special tokens).

    Returns:
        keras.Model: Compiled Keras model.
    """
    # Load the DistilBERT encoder
    try:
        encoder = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
        # Resize token embeddings if tokenizer has added special tokens
        if encoder.config.vocab_size != tokenizer_vocab_size:
             encoder.resize_token_embeddings(tokenizer_vocab_size)
             logger.info(f"Resized DistilBERT token embeddings to {tokenizer_vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load DistilBERT model: {e}")
        raise

    # Define input layers
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # Get DistilBERT embeddings
    # Use training=False to disable dropout during inference if needed, but usually keep default for training
    embedding = encoder(input_ids, attention_mask=attention_mask)[0] # Last hidden state

    # Create start and end logits for span prediction
    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten(name="start_flatten")(start_logits) # Shape: (batch_size, max_len)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten(name="end_flatten")(end_logits) # Shape: (batch_size, max_len)

    # Using logits directly with SparseCategoricalCrossentropy (from_logits=True) is often more stable
    # Softmax is implicitly handled by the loss function.
    # If using softmax activation here, set from_logits=False in the loss.

    # Build the model
    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[start_logits, end_logits], # Output logits
    )

    # Compile the model
    # Use from_logits=True as we are outputting raw logits
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Add metrics like Exact Match and IoU later via callbacks if needed
    model.compile(optimizer=optimizer, loss=[loss, loss]) # Separate loss for start and end

    logger.info("Span prediction model created and compiled.")
    model.summary(print_fn=logger.info) # Log summary
    return model