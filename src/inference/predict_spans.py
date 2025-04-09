# src/inference/predict_spans.py
import tensorflow as tf
# --- Start Change ---
from transformers import DistilBertTokenizer, TFAutoModelForQuestionAnswering # Use AutoModel
# --- End Change ---
import numpy as np
import logging
import os
import json
import argparse
from tqdm import tqdm
from ..common.database import db_connect, get_contexts_for_inference, add_predicted_description
from ..common.config import DEFAULT_DB_PATH, DEFAULT_MODELS_DIR, DEFAULT_INFERENCE_PARAMS
from ..common.logging_config import setup_logging

logger = logging.getLogger(__name__)

# prepare_inference_input remains the same as it just prepares tokens/masks
def prepare_inference_input(context_before, context_after, tokenizer: DistilBertTokenizer, max_len, max_before_tokens):
    # ... (Keep implementation from predict_spans_complete) ...
    img_token = "[IMG]"
    if img_token not in tokenizer.vocab:
        tokenizer.add_special_tokens({'additional_special_tokens': [img_token]})
    img_token_id = tokenizer.convert_tokens_to_ids(img_token)
    cls_token_id = tokenizer.cls_token_id; sep_token_id = tokenizer.sep_token_id; pad_token_id = tokenizer.pad_token_id
    before_tokens = tokenizer.encode(context_before or "", add_special_tokens=False)
    after_tokens = tokenizer.encode(context_after or "", add_special_tokens=False)
    available_len_for_context = max_len - 3
    if available_len_for_context <= 0: return None, None, None
    truncated_before = before_tokens[-max_before_tokens:]
    remaining_len = available_len_for_context - len(truncated_before)
    if remaining_len < 0: truncated_before = truncated_before[-available_len_for_context:]; truncated_after = []
    else: truncated_after = after_tokens[:remaining_len]
    original_tokens = [cls_token_id] + truncated_before + [img_token_id] + truncated_after + [sep_token_id]
    current_len = len(original_tokens)
    if current_len > max_len: original_tokens = original_tokens[:max_len-1] + [sep_token_id]; current_len = max_len
    elif current_len == 0: return None, None, None
    padding_len = max_len - current_len
    input_ids = original_tokens + [pad_token_id] * padding_len
    attention_mask = [1] * current_len + [0] * padding_len
    if len(input_ids) != max_len or len(attention_mask) != max_len: return None, None, None
    return np.array(input_ids, dtype=np.int32), np.array(attention_mask, dtype=np.int32), original_tokens


def run_inference(db_path, model_path, batch_size=32, max_len=512, max_before_tokens=250):
    """
    Runs span prediction inference using a pre-trained HF QA model.
    """
    logger.info(f"Starting inference using model from directory: {model_path}")

    # --- Load Model and Tokenizer ---
    try:
        # --- Start Change ---
        # Load model using from_pretrained
        model = TFAutoModelForQuestionAnswering.from_pretrained(model_path)
        # Load tokenizer from the same path (saved with model)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        logger.info(f"Loaded QA model and tokenizer from {model_path}")
        # --- End Change ---
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer from {model_path}: {e}", exc_info=True)
        return

    # --- Process Data in Batches (Loop structure remains similar) ---
    processed_count = 0
    added_count = 0
    offset = 0
    db_fetch_limit = batch_size * 10

    while True:
        logger.info(f"Fetching inference batch starting from offset {offset}...")
        contexts = []
        try:
            with db_connect(db_path) as cursor:
                 contexts = get_contexts_for_inference(cursor, limit=db_fetch_limit, offset=offset)
        except Exception as e: logger.error(f"DB error fetching contexts: {e}"); break
        if not contexts: logger.info("No more contexts for inference."); break

        batch_input_ids = []
        batch_attn_masks = []
        batch_original_tokens = []
        batch_context_ids = []

        for row in contexts:
            context_id = row['context_id']
            input_ids, attention_mask, original_tokens = prepare_inference_input(
                row['context_before'], row['context_after'], tokenizer, max_len, max_before_tokens
            )
            if input_ids is not None:
                batch_input_ids.append(input_ids)
                batch_attn_masks.append(attention_mask)
                batch_original_tokens.append(original_tokens)
                batch_context_ids.append(context_id)

        if not batch_input_ids:
            logger.warning(f"No valid inputs prepared from DB fetch starting at offset {offset}.")
            offset += len(contexts); continue

        logger.info(f"Predicting on batch of size {len(batch_input_ids)}...")
        try:
            # Prepare input dictionary for HF model
            inputs_tf = {
                'input_ids': tf.convert_to_tensor(batch_input_ids, dtype=tf.int32),
                'attention_mask': tf.convert_to_tensor(batch_attn_masks, dtype=tf.int32)
            }
            # --- Start Change ---
            # Get model predictions (output object)
            outputs = model(inputs_tf) # No training=False needed usually for inference call
            start_logits_batch = outputs.start_logits
            end_logits_batch = outputs.end_logits
            # --- End Change ---

            start_indices = tf.argmax(start_logits_batch, axis=-1).numpy()
            end_indices = tf.argmax(end_logits_batch, axis=-1).numpy()

            # Calculate probabilities if needed (apply softmax)
            start_probs_batch = tf.nn.softmax(start_logits_batch, axis=-1).numpy()
            end_probs_batch = tf.nn.softmax(end_logits_batch, axis=-1).numpy()

        except Exception as e:
            logger.error(f"Error during model prediction: {e}", exc_info=True)
            offset += len(contexts); continue

        logger.info(f"Saving {len(start_indices)} predictions to database...")
        num_added_in_batch = 0
        try:
            with db_connect(db_path) as cursor:
                for i in range(len(start_indices)):
                    start_idx = int(start_indices[i]) # Convert numpy int64
                    end_idx = int(end_indices[i])
                    context_id = batch_context_ids[i]
                    original_toks = batch_original_tokens[i]
                    original_len = sum(batch_attn_masks[i])

                    # Filter invalid spans
                    if start_idx >= original_len or end_idx >= original_len or start_idx > end_idx:
                        logger.debug(f"Invalid span indices predicted for context_id {context_id}: start={start_idx}, end={end_idx}, original_len={original_len}. Skipping.")
                        continue

                    # Decode description (ensure start/end are within original length)
                    # Add +1 because slicing is exclusive at the end
                    predicted_description = tokenizer.decode(original_toks[start_idx : end_idx + 1], skip_special_tokens=True)

                    if not predicted_description or predicted_description.isspace():
                         logger.debug(f"Empty description decoded for context_id {context_id}. Skipping.")
                         continue

                    # Confidence scores
                    start_confidence = float(start_probs_batch[i][start_idx])
                    end_confidence = float(end_probs_batch[i][end_idx])
                    average_confidence = (start_confidence + end_confidence) / 2.0
                    span_confidence = start_confidence * end_confidence

                    confidences = {'start': start_confidence, 'end': end_confidence, 'average': average_confidence, 'span': span_confidence}

                    pred_id = add_predicted_description(cursor, context_id, model_path, predicted_description, start_idx, end_idx, confidences)
                    if pred_id: num_added_in_batch += 1

                logger.info(f"Added {num_added_in_batch} predictions from this batch.")
                added_count += num_added_in_batch
                processed_count += len(start_indices) # Count attempted predictions

        except Exception as e:
            logger.error(f"Database error saving predictions: {e}", exc_info=True)

        offset += len(contexts) # Advance offset

    logger.info(f"Inference finished. Total contexts processed: {processed_count}, Predictions added: {added_count}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Run span prediction inference.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help=f"Path to the SQLite database file (default: {DEFAULT_DB_PATH})")
    # --- Start Change ---
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved Hugging Face model directory (output of training stage).")
    # --- End Change ---
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction (default: 32)")
    parser.add_argument("--max-len", type=int, default=DEFAULT_INFERENCE_PARAMS['max_len'], help=f"Max sequence length for model (default: {DEFAULT_INFERENCE_PARAMS['max_len']})")
    parser.add_argument("--max-before-tokens", type=int, default=DEFAULT_INFERENCE_PARAMS['max_before_tokens'], help=f"Max tokens from context before [IMG] (default: {DEFAULT_INFERENCE_PARAMS['max_before_tokens']})")
    args = parser.parse_args()

    if not os.path.isdir(args.model_path): # Check if it's a directory
        logger.error(f"Model path not found or not a directory: {args.model_path}")
    elif not os.path.exists(args.db_path):
        logger.error(f"Database path not found: {args.db_path}")
    else:
        run_inference(db_path=args.db_path, model_path=args.model_path, batch_size=args.batch_size, max_len=args.max_len, max_before_tokens=args.max_before_tokens)

