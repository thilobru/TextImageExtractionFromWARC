# src/training/model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFDistilBertModel, DistilBertTokenizer
import logging

logger = logging.getLogger(__name__)

class SpanPredictionModel(keras.Model):
    def __init__(self, max_len, tokenizer_vocab_size, name="span_prediction_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_len = max_len

        try:
            self.encoder = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
            if self.encoder.config.vocab_size != tokenizer_vocab_size:
                 self.encoder.resize_token_embeddings(tokenizer_vocab_size)
                 logger.info(f"Resized DistilBERT token embeddings to {tokenizer_vocab_size}")
            self.hidden_size = self.encoder.config.dim
        except Exception as e:
            logger.error(f"Failed to load DistilBert model: {e}")
            raise

        self.start_dense = layers.Dense(1, name="start_logit", use_bias=False)
        # Flatten layer removed
        self.end_dense = layers.Dense(1, name="end_logit", use_bias=False)
        # Flatten layer removed

    # No @tf.function decorator for now
    def call(self, input_ids, attention_mask, training=False): # Use named args
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )[0] # Shape: (batch_size, sequence_length, hidden_size)

        start_logits = self.start_dense(encoder_output) # Shape: (batch_size, sequence_length, 1)
        # Replace Flatten layer with tf.squeeze
        start_logits = tf.squeeze(start_logits, axis=-1, name="start_squeeze") # Shape: (batch_size, sequence_length)

        end_logits = self.end_dense(encoder_output) # Shape: (batch_size, sequence_length, 1)
        # Replace Flatten layer with tf.squeeze
        end_logits = tf.squeeze(end_logits, axis=-1, name="end_squeeze") # Shape: (batch_size, sequence_length)

        return [start_logits, end_logits]


def create_span_prediction_model(max_len, learning_rate, tokenizer_vocab_size):
    """
    Creates and compiles the DistilBERT model for span prediction using subclassing.
    Model is built on first call.
    """
    model = SpanPredictionModel(max_len=max_len, tokenizer_vocab_size=tokenizer_vocab_size)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=[loss, loss])

    logger.info("Span prediction model (subclassed) created and compiled.")

    # --- REMOVED explicit model.build() and model.summary() ---
    # The model will be built when first called.
    # Summary can be printed after building if needed.

    return model
