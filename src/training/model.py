# src/training/model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFDistilBertModel, DistilBertTokenizer
import logging

logger = logging.getLogger(__name__)

# Define the model by subclassing tf.keras.Model
class SpanPredictionModel(keras.Model):
    def __init__(self, max_len, tokenizer_vocab_size, name="span_prediction_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_len = max_len

        # Load the DistilBERT encoder
        try:
            self.encoder = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
            # Resize token embeddings if tokenizer has added special tokens
            if self.encoder.config.vocab_size != tokenizer_vocab_size:
                 self.encoder.resize_token_embeddings(tokenizer_vocab_size)
                 logger.info(f"Resized DistilBERT token embeddings to {tokenizer_vocab_size}")
            self.hidden_size = self.encoder.config.dim
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}")
            raise

        # Define subsequent layers
        self.start_dense = layers.Dense(1, name="start_logit", use_bias=False)
        self.start_flatten = layers.Flatten(name="start_flatten")

        self.end_dense = layers.Dense(1, name="end_logit", use_bias=False)
        self.end_flatten = layers.Flatten(name="end_flatten")

    def call(self, inputs, training=False):
        # Inputs should be a dictionary or a tuple/list [input_ids, attention_mask]
        if isinstance(inputs, (list, tuple)):
            input_ids, attention_mask = inputs
        elif isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
        else:
            raise ValueError("Inputs should be a list, tuple, or dict containing 'input_ids' and 'attention_mask'")

        # Pass inputs through the encoder
        # Pass training argument to control dropout etc.
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training # Pass the training flag
        )[0] # Get the last hidden state

        # Pass through dense layers
        start_logits = self.start_dense(encoder_output)
        start_logits = self.start_flatten(start_logits) # Shape: (batch_size, max_len)

        end_logits = self.end_dense(encoder_output)
        end_logits = self.end_flatten(end_logits) # Shape: (batch_size, max_len)

        return [start_logits, end_logits] # Return logits

    # Optional: Define compute_output_shape if needed, but usually inferred correctly in subclassing
    # def compute_output_shape(self, input_shape):
    #     # input_shape is expected to be a tuple of shapes [(batch, max_len), (batch, max_len)]
    #     batch_size = input_shape[0][0]
    #     seq_len = input_shape[0][1]
    #     return [(batch_size, seq_len), (batch_size, seq_len)]


def create_span_prediction_model(max_len, learning_rate, tokenizer_vocab_size):
    """
    Creates and compiles the DistilBERT model for span prediction using subclassing.

    Args:
        max_len (int): Maximum sequence length.
        learning_rate (float): Optimizer learning rate.
        tokenizer_vocab_size (int): Size of the tokenizer vocabulary (incl. special tokens).

    Returns:
        keras.Model: Compiled Keras model instance.
    """
    # Instantiate the subclassed model
    model = SpanPredictionModel(max_len=max_len, tokenizer_vocab_size=tokenizer_vocab_size)

    # Compile the model
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=[loss, loss])

    logger.info("Span prediction model (subclassed) created and compiled.")

    # Build the model by calling it with dummy input to print summary correctly
    # (Subclassed models need a call to be built)
    dummy_input_ids = tf.zeros((1, max_len), dtype=tf.int32)
    dummy_attention_mask = tf.zeros((1, max_len), dtype=tf.int32)
    try:
        model([dummy_input_ids, dummy_attention_mask], training=False)
        model.summary(print_fn=logger.info) # Log summary after building
    except Exception as e:
        logger.error(f"Failed to build the subclassed model for summary: {e}")


    return model
