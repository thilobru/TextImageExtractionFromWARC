# src/inference/predict_spans.py
import tensorflow as tf
from transformers import DistilBertTokenizer
import numpy as np
import logging
import os
import json
import argparse
from tqdm import tqdm # Added tqdm for progress
from ..common.database import db_connect, get_contexts_for_inference, add_predicted_description
from ..common.config import DEFAULT_DB_PATH, DEFAULT_MODELS_DIR, DEFAULT_INFERENCE_PARAMS
from ..common.logging_config import setup_logging

logger = logging.getLogger(__name__)

def prepare_inference_input(context_before, context_after, tokenizer: DistilBertTokenizer, max_len, max_before_tokens):
    """
    Prepares a single input for inference, excluding alt_tag tokens.

    Args:
        context_before (str): Text before the image.
        context_after (str): Text after the image.
        tokenizer: The tokenizer instance.
        max_len: Maximum sequence length for the model.
        max_before_tokens: Max tokens to keep from context_before.

    Returns:
        tuple: (input_ids, attention_mask, original_tokens) or None if invalid.
               original_tokens are the actual tokens fed to the model (before padding).
    """
    img_token = "[IMG]"
    # Ensure the special token is handled correctly
    if img_token not in tokenizer.vocab:
        logger.warning(f"Special token '{img_token}' not found in tokenizer vocab during inference preparation.")
        # Attempt to add it if missing, though it should be loaded with the tokenizer
        tokenizer.add_special_tokens({'additional_special_tokens': [img_token]})

    img_token_id = tokenizer.convert_tokens_to_ids(img_token)
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    before_tokens = tokenizer.encode(context_before or "", add_special_tokens=False)
    after_tokens = tokenizer.encode(context_after or "", add_special_tokens=False)

    # Truncate context, prioritizing tokens around [IMG]
    available_len = max_len - 3 # Space for [CLS], [IMG], [SEP]
    if available_len <= 0:
         logger.error(f"max_len ({max_len}) is too small to accommodate special tokens.")
         return None, None, None

    truncated_before = before_tokens[-max_before_tokens:] # Keep end of 'before' context

    remaining_len = available_len - len(truncated_before)
    if remaining_len < 0: # If before_tokens alone exceeds limit
         truncated_before = truncated_before[-available_len:] # Further truncate before_tokens
         truncated_after = []
         logger.warning(f"Before context exceeded available length ({available_len}), truncated.")
         remaining_len = 0 # No space left for after_tokens
    else:
        truncated_after = after_tokens[:remaining_len] # Keep start of 'after' context

    original_tokens = [cls_token_id] + truncated_before + [img_token_id] + truncated_after + [sep_token_id]
    current_len = len(original_tokens)

    if current_len > max_len:
        # This shouldn't happen with correct logic, but double-check
        logger.error(f"Sequence length {current_len} exceeds max_len {max_len} after construction.")
        original_tokens = original_tokens[:max_len-1] + [sep_token_id] # Force fit
        current_len = max_len
    elif current_len == 0:
         logger.warning("Constructed sequence has zero length.")
         return None, None, None


    # Pad sequence
    padding_len = max_len - current_len
    input_ids = original_tokens + [pad_token_id] * padding_len
    attention_mask = [1] * current_len + [0] * padding_len

    if len(input_ids) != max_len or len(attention_mask) != max_len:
        logger.error("Padding resulted in incorrect length.")
        return None, None, None # Indicate error

    return np.array(input_ids, dtype=np.int32), np.array(attention_mask, dtype=np.int32), original_tokens


def run_inference(db_path, model_path, batch_size=32, max_len=512, max_before_tokens=250):
    """
    Runs span prediction inference on data from the database.

    Args:
        db_path (str): Path to the SQLite database.
        model_path (str): Path to the trained Keras model file (.h5 or SavedModel dir).
        batch_size (int): Batch size for model prediction.
        max_len (int): Max sequence length model expects.
        max_before_tokens (int): Max tokens from 'before' context to use.
    """
    logger.info(f"Starting inference using model: {model_path}")

    # --- Load Model and Tokenizer ---
    try:
        # Check if path is SavedModel directory or H5 file
        if os.path.isdir(model_path) or model_path.endswith('.keras'):
             model = tf.keras.models.load_model(model_path, compile=False)
        elif model_path.endswith('.h5'):
             # Loading H5 might require custom objects if model uses custom layers/losses not standard in TF
             # Assuming standard layers for now
             model = tf.keras.models.load_model(model_path, compile=False)
        else:
             raise ValueError(f"Model path is not a recognized format (directory, .keras, or .h5): {model_path}")

        logger.info(f"Loaded model from {model_path}")
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    try:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        special_token = "[IMG]"
        if special_token not in tokenizer.additional_special_tokens:
            # Ensure tokenizer used for inference matches the one used for training
            num_added = tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
            if num_added > 0:
                 logger.info(f"Added special token '{special_token}' to tokenizer for inference.")
            # IMPORTANT: If tokens were added, the model's embedding layer size
            # must match len(tokenizer). Loading a saved model usually handles this,
            # but it's crucial they are consistent.
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # --- Process Data in Batches ---
    processed_count = 0
    added_count = 0
    offset = 0
    # Fetch a larger chunk from DB to reduce DB query frequency
    db_fetch_limit = batch_size * 10

    while True:
        logger.info(f"Fetching inference batch starting from offset {offset}...")
        batch_input_ids = []
        batch_attn_masks = []
        batch_original_tokens = []
        batch_context_ids = []

        try:
            with db_connect(db_path) as cursor:
                 # Modify get_contexts_for_inference if status tracking is added
                 # e.g., add WHERE status_prediction = 'pending'
                 contexts = get_contexts_for_inference(cursor, limit=db_fetch_limit, offset=offset)
        except Exception as e:
            logger.error(f"Database error fetching contexts: {e}")
            break # Stop processing if DB fails

        if not contexts:
            logger.info("No more contexts found for inference.")
            break # End of data

        logger.info(f"Preparing {len(contexts)} contexts for prediction...")
        for row in contexts:
            context_id = row['context_id']
            # Prepare single input
            input_ids, attention_mask, original_tokens = prepare_inference_input(
                row['context_before'], row['context_after'], tokenizer, max_len, max_before_tokens
            )

            if input_ids is not None:
                batch_input_ids.append(input_ids)
                batch_attn_masks.append(attention_mask)
                batch_original_tokens.append(original_tokens) # Keep original tokens for decoding
                batch_context_ids.append(context_id)

        if not batch_input_ids:
            logger.warning(f"No valid inputs prepared from DB fetch starting at offset {offset}.")
            offset += len(contexts) # Advance offset even if no valid inputs prepared
            continue # Skip to next fetch

        logger.info(f"Predicting on batch of size {len(batch_input_ids)}...")
        try:
            # Convert lists to tensors for the batch
            input_ids_tf = tf.convert_to_tensor(batch_input_ids, dtype=tf.int32)
            attention_mask_tf = tf.convert_to_tensor(batch_attn_masks, dtype=tf.int32)

            # Get model predictions (logits)
            start_logits_batch, end_logits_batch = model.predict([input_ids_tf, attention_mask_tf], batch_size=batch_size)

            # Get the most probable start and end positions for each item in the batch
            start_indices = tf.argmax(start_logits_batch, axis=-1).numpy()
            end_indices = tf.argmax(end_logits_batch, axis=-1).numpy()

            # Get confidence scores (probabilities after softmax, if model outputs logits use tf.nn.softmax)
            # Assuming model outputs logits, apply softmax
            start_probs_batch = tf.nn.softmax(start_logits_batch, axis=-1).numpy()
            end_probs_batch = tf.nn.softmax(end_logits_batch, axis=-1).numpy()


        except Exception as e:
            logger.error(f"Error during model prediction: {e}", exc_info=True)
            # Decide how to handle: skip batch, stop? For now, skip batch.
            offset += len(contexts) # Advance offset
            continue

        logger.info(f"Saving {len(start_indices)} predictions to database...")
        num_added_in_batch = 0
        try:
            with db_connect(db_path) as cursor: # Reconnect for batch write
                for i in range(len(start_indices)):
                    start_idx = start_indices[i]
                    end_idx = end_indices[i]
                    context_id = batch_context_ids[i]
                    original_toks = batch_original_tokens[i] # Use original tokens before padding

                    # Ensure start <= end and indices are within the *original* sequence length
                    original_len = sum(batch_attn_masks[i]) # Length before padding
                    if start_idx >= original_len or end_idx >= original_len or start_idx > end_idx:
                        logger.warning(f"Invalid span indices predicted for context_id {context_id}: start={start_idx}, end={end_idx}, original_len={original_len}. Skipping.")
                        continue

                    # Decode the description using original tokens (before padding)
                    # Add skip_special_tokens=True to remove [CLS], [SEP] if they get included
                    predicted_description = tokenizer.decode(original_toks[start_idx : end_idx + 1], skip_special_tokens=True)

                    # If decoded description is empty or just whitespace, skip
                    if not predicted_description or predicted_description.isspace():
                         logger.warning(f"Empty description decoded for context_id {context_id}. Skipping.")
                         continue

                    # Confidence scores for the predicted indices
                    start_confidence = float(start_probs_batch[i][start_idx])
                    end_confidence = float(end_probs_batch[i][end_idx])
                    average_confidence = (start_confidence + end_confidence) / 2.0
                    span_confidence = start_confidence * end_confidence # Geometric mean proxy

                    confidences = {
                        'start': start_confidence,
                        'end': end_confidence,
                        'average': average_confidence,
                        'span': span_confidence
                    }

                    # Add prediction to database
                    pred_id = add_predicted_description(
                        cursor, context_id, model_path, predicted_description,
                        int(start_idx), int(end_idx), confidences
                    )
                    if pred_id:
                        num_added_in_batch += 1

                logger.info(f"Added {num_added_in_batch} predictions from this batch.")
                added_count += num_added_in_batch
                processed_count += len(start_indices)

        except Exception as e:
            logger.error(f"Database error saving predictions: {e}", exc_info=True)
            # Decide how to handle: stop? For now, continue to next fetch.

        # Advance offset for the next DB fetch
        offset += len(contexts)

    logger.info(f"Inference finished. Total contexts processed (attempted prediction): {processed_count}, Predictions added: {added_count}")


if __name__ == "__main__":
    setup_logging() # Setup basic logging

    parser = argparse.ArgumentParser(description="Run span prediction inference.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH,
                        help=f"Path to the SQLite database file (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained Keras model (.h5 file or SavedModel directory)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for prediction (default: 32)")
    parser.add_argument("--max-len", type=int, default=DEFAULT_INFERENCE_PARAMS['max_len'],
                        help=f"Max sequence length for model (default: {DEFAULT_INFERENCE_PARAMS['max_len']})")
    parser.add_argument("--max-before-tokens", type=int, default=DEFAULT_INFERENCE_PARAMS['max_before_tokens'],
                        help=f"Max tokens from context before [IMG] (default: {DEFAULT_INFERENCE_PARAMS['max_before_tokens']})")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        logger.error(f"Model path not found: {args.model_path}")
    elif not os.path.exists(args.db_path):
        logger.error(f"Database path not found: {args.db_path}")
    else:
        run_inference(
            db_path=args.db_path,
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            max_before_tokens=args.max_before_tokens
        )
