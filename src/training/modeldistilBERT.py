# src/training/model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
from transformers import TFDistilBertModel, DistilBertTokenizer
import logging

logger = logging.getLogger(__name__)

class SpanPredictionModel(keras.Model):
    # __init__ and call methods remain the same as the version that
    # uses tf.squeeze and has no @tf.function decorator on call
    def __init__(self, max_len, tokenizer_vocab_size, name="span_prediction_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_len = max_len
        self.tokenizer_vocab_size = tokenizer_vocab_size

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
        self.start_squeeze = layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="start_squeeze")

        self.end_dense = layers.Dense(1, name="end_logit", use_bias=False)
        self.end_squeeze = layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="end_squeeze")

    def call(self, inputs, training=False):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            input_ids, attention_mask = inputs
        elif isinstance(inputs, dict):
             input_ids = inputs.get("input_ids")
             attention_mask = inputs.get("attention_mask")
             if input_ids is None or attention_mask is None:
                  raise ValueError("Inputs dict must contain 'input_ids' and 'attention_mask'")
        else:
             raise ValueError("Inputs should be a list/tuple [input_ids, attention_mask] or dict.")

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )[0]

        start_logits = self.start_dense(encoder_output)
        start_logits = self.start_squeeze(start_logits)

        end_logits = self.end_dense(encoder_output)
        end_logits = self.end_squeeze(end_logits)

        return [start_logits, end_logits]


def create_span_prediction_model(max_len, learning_rate, tokenizer_vocab_size):
    """
    Creates and compiles the DistilBERT model for span prediction using subclassing.
    """
    model = SpanPredictionModel(max_len=max_len, tokenizer_vocab_size=tokenizer_vocab_size)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = LegacyAdam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=[loss, loss])

    logger.info("Span prediction model (subclassed) created and compiled.")

    # Build call removed - model builds on first call/fit

    return model
