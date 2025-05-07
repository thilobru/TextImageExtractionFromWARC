# src/inference/predict_spans.py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering # Use AutoModel and AutoTokenizer
import numpy as np
import logging
import os
import json # For saving confidences if needed as JSON string in DB
import argparse
from tqdm import tqdm
from ..common.database import db_connect, get_contexts_for_inference, add_predicted_description
from ..common.config import DEFAULT_DB_PATH, DEFAULT_INFERENCE_PARAMS # Using DEFAULT_MODELS_DIR is removed as model_path is required
from ..common.logging_config import setup_logging

logger = logging.getLogger(__name__)

# prepare_inference_input remains largely the same, but ensure it uses the passed tokenizer
def prepare_inference_input(context_before, context_after, tokenizer: AutoTokenizer, max_len, max_before_tokens):
    """
    Prepares a single input for inference, excluding alt_tag tokens.
    Uses the provided Hugging Face tokenizer.
    """
    img_token = "[IMG]"
    # Ensure the special token is handled correctly by the loaded tokenizer
    # It should have been added during tokenizer loading if it was new.
    if img_token not in tokenizer.vocab:
        logger.warning(f"Special token '{img_token}' not found in tokenizer vocab during inference preparation. This might indicate a mismatch with training tokenizer.")
        # Depending on policy, you might add it, but it's better if it's already there from model_path
        # tokenizer.add_special_tokens({'additional_special_tokens': [img_token]})

    img_token_id = tokenizer.convert_tokens_to_ids(img_token)
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    # Handle cases where special tokens might not exist (e.g., for some model types)
    if cls_token_id is None or sep_token_id is None:
        logger.error(f"Tokenizer {tokenizer.name_or_path} is missing CLS or SEP token. Cannot prepare input.")
        return None, None, None
    if pad_token_id is None: # Some tokenizers might not have a PAD token by default, set it if not
        logger.warning(f"Tokenizer {tokenizer.name_or_path} is missing PAD token. Using EOS token as PAD.")
        tokenizer.pad_token = tokenizer.eos_token # Common practice
        pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None: # If still no PAD token
            logger.error(f"Tokenizer {tokenizer.name_or_path} has no PAD or EOS token. Cannot prepare input.")
            return None, None, None


    before_tokens = tokenizer.encode(context_before or "", add_special_tokens=False)
    after_tokens = tokenizer.encode(context_after or "", add_special_tokens=False)

    # Truncate context, prioritizing tokens around [IMG]
    # Max length for context tokens = max_len - (CLS + IMG + SEP)
    available_len_for_context = max_len - 3 
    if available_len_for_context <= 0:
         logger.error(f"max_len ({max_len}) is too small to accommodate special tokens ([CLS], [IMG], [SEP]).")
         return None, None, None

    truncated_before = before_tokens[-max_before_tokens:] 

    remaining_len_for_after = available_len_for_context - len(truncated_before)
    if remaining_len_for_after < 0: # If before_tokens alone already exceed available_len_for_context
         truncated_before = truncated_before[-available_len_for_context:] # Further truncate before_tokens
         truncated_after = []
         logger.debug(f"Before context (len {len(before_tokens)}) exceeded available length ({available_len_for_context}), truncated to {len(truncated_before)}.")
    else:
        truncated_after = after_tokens[:remaining_len_for_after]

    # Construct the sequence: [CLS] truncated_before [IMG] truncated_after [SEP]
    original_tokens_ids = [cls_token_id] + truncated_before + [img_token_id] + truncated_after + [sep_token_id]
    current_len = len(original_tokens_ids)

    if current_len > max_len:
        # This should ideally not happen if logic above is correct, but as a safeguard:
        logger.warning(f"Constructed sequence length {current_len} exceeds max_len {max_len}. Truncating forcefully.")
        original_tokens_ids = original_tokens_ids[:max_len-1] + [sep_token_id] # Ensure SEP is last
        current_len = max_len
    elif current_len == 0: # Should not happen with CLS, IMG, SEP
         logger.warning("Constructed sequence has zero length. Skipping.")
         return None, None, None

    # Pad sequence
    padding_len = max_len - current_len
    input_ids = original_tokens_ids + [pad_token_id] * padding_len
    attention_mask = [1] * current_len + [0] * padding_len

    if len(input_ids) != max_len or len(attention_mask) != max_len:
        logger.error("Padding resulted in incorrect final length. This indicates an issue in truncation/padding logic.")
        return None, None, None 

    return np.array(input_ids, dtype=np.int32), np.array(attention_mask, dtype=np.int32), original_tokens_ids


def run_inference(db_path, model_path, batch_size=32, max_len=512, max_before_tokens=250):
    """
    Runs span prediction inference using a fine-tuned Hugging Face QA model.
    Args:
        db_path (str): Path to the SQLite database.
        model_path (str): Path to the directory containing the saved model and tokenizer
                          (output of `save_pretrained`).
        batch_size (int): Batch size for model prediction.
        max_len (int): Max sequence length model expects.
        max_before_tokens (int): Max tokens from 'before' context to use.
    """
    logger.info(f"Starting inference using model from directory: {model_path}")

    # --- Load Model and Tokenizer ---
    try:
        # Load model using TFAutoModelForQuestionAnswering from the specified path
        model = TFAutoModelForQuestionAnswering.from_pretrained(model_path)
        # Load tokenizer from the same path (it should have been saved with the model)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Loaded QA model and tokenizer from {model_path}")
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer from {model_path}: {e}", exc_info=True)
        return

    # --- Process Data in Batches ---
    processed_count = 0 # Number of contexts for which prediction was attempted
    added_count = 0     # Number of valid predictions added to DB
    offset = 0
    db_fetch_limit = batch_size * 10 # Fetch a larger chunk from DB to reduce query frequency

    while True:
        logger.info(f"Fetching inference batch from database, starting from offset {offset}...")
        contexts_from_db = []
        try:
            with db_connect(db_path) as cursor:
                 # Modify get_contexts_for_inference if you add a status field 
                 # to ImageContext to track prediction status (e.g., 'pending_prediction').
                 contexts_from_db = get_contexts_for_inference(cursor, limit=db_fetch_limit, offset=offset)
        except Exception as e: 
            logger.error(f"Database error fetching contexts: {e}", exc_info=True)
            break # Stop processing if DB fails

        if not contexts_from_db:
            logger.info("No more contexts found for inference.")
            break # End of data

        batch_input_ids_list = []
        batch_attn_masks_list = []
        batch_original_tokens_ids_list = [] # To store original token IDs for decoding
        batch_context_ids_list = []         # To link predictions back to DB contexts

        logger.info(f"Preparing {len(contexts_from_db)} contexts for prediction...")
        for row in contexts_from_db:
            context_id = row['context_id']
            # Prepare single input using the loaded tokenizer
            input_ids_arr, attention_mask_arr, original_tokens_ids = prepare_inference_input(
                row['context_before'], row['context_after'], tokenizer, max_len, max_before_tokens
            )

            if input_ids_arr is not None and attention_mask_arr is not None:
                batch_input_ids_list.append(input_ids_arr)
                batch_attn_masks_list.append(attention_mask_arr)
                batch_original_tokens_ids_list.append(original_tokens_ids) 
                batch_context_ids_list.append(context_id)
            else:
                logger.warning(f"Skipping context_id {context_id} due to input preparation error.")


        if not batch_input_ids_list:
            logger.warning(f"No valid inputs prepared from DB fetch starting at offset {offset}. Advancing offset.")
            offset += len(contexts_from_db) # Advance offset even if no valid inputs prepared from this chunk
            continue # Skip to next fetch

        logger.info(f"Predicting on batch of size {len(batch_input_ids_list)}...")
        try:
            # Prepare input dictionary for HF model
            inputs_tf_dict = {
                'input_ids': tf.convert_to_tensor(batch_input_ids_list, dtype=tf.int32),
                'attention_mask': tf.convert_to_tensor(batch_attn_masks_list, dtype=tf.int32)
            }
            
            # Get model predictions (output object)
            # No training=False needed for HF model's __call__ or predict method during inference
            outputs = model(inputs_tf_dict) 
            start_logits_batch = outputs.start_logits
            end_logits_batch = outputs.end_logits
            
            # Get the most probable start and end positions for each item in the batch
            pred_start_indices = tf.argmax(start_logits_batch, axis=-1).numpy()
            pred_end_indices = tf.argmax(end_logits_batch, axis=-1).numpy()

            # Get confidence scores (probabilities after softmax)
            start_probs_batch = tf.nn.softmax(start_logits_batch, axis=-1).numpy()
            end_probs_batch = tf.nn.softmax(end_logits_batch, axis=-1).numpy()

        except Exception as e:
            logger.error(f"Error during model prediction: {e}", exc_info=True)
            offset += len(contexts_from_db) # Advance offset
            continue

        logger.info(f"Saving {len(pred_start_indices)} predictions to database...")
        num_added_in_batch = 0
        try:
            with db_connect(db_path) as cursor: # Reconnect for batch write
                for i in range(len(pred_start_indices)):
                    start_idx = int(pred_start_indices[i]) # Convert numpy int64 to Python int
                    end_idx = int(pred_end_indices[i])
                    current_context_id = batch_context_ids_list[i]
                    original_tok_ids_for_item = batch_original_tokens_ids_list[i] # Original tokens before padding
                    
                    # original_len is the number of non-padded tokens
                    original_len = sum(batch_attn_masks_list[i]) 

                    # --- Span Validation ---
                    # 1. Ensure start_idx and end_idx are within the valid range of *original* (unpadded) tokens
                    # 2. Ensure start_idx <= end_idx
                    if not (0 <= start_idx < original_len and 0 <= end_idx < original_len and start_idx <= end_idx):
                        logger.debug(f"Invalid span indices predicted for context_id {current_context_id}: start={start_idx}, end={end_idx}, original_len={original_len}. Skipping.")
                        continue

                    # Decode the description using original token IDs (before padding)
                    # Slicing is [start_idx : end_idx + 1] to include the token at end_idx
                    predicted_token_ids = original_tok_ids_for_item[start_idx : end_idx + 1]
                    predicted_description = tokenizer.decode(predicted_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    # If decoded description is empty or just whitespace, skip
                    if not predicted_description or predicted_description.isspace():
                         logger.debug(f"Empty or whitespace-only description decoded for context_id {current_context_id} (span {start_idx}-{end_idx}). Skipping.")
                         continue

                    # Confidence scores for the predicted indices
                    start_confidence = float(start_probs_batch[i][start_idx])
                    end_confidence = float(end_probs_batch[i][end_idx])
                    average_confidence = (start_confidence + end_confidence) / 2.0
                    # Span confidence can be geometric mean or product, ensure it's meaningful
                    span_confidence = np.sqrt(start_confidence * end_confidence) # Geometric mean

                    confidences_dict = {
                        'start': start_confidence,
                        'end': end_confidence,
                        'average': average_confidence,
                        'span': span_confidence # Store as float
                    }

                    # Add prediction to database
                    # The model_path argument to add_predicted_description should be the path to the model *directory*
                    pred_id = add_predicted_description(
                        cursor, current_context_id, model_path, predicted_description,
                        start_idx, end_idx, confidences_dict
                    )
                    if pred_id:
                        num_added_in_batch += 1
                
                logger.info(f"Added {num_added_in_batch} predictions to database from this batch.")
                added_count += num_added_in_batch
        
        except Exception as e:
            logger.error(f"Database error while saving predictions: {e}", exc_info=True)
            # Decide how to handle: stop? For now, continue to next fetch.

        processed_count += len(batch_input_ids_list) # Count successfully prepared and predicted items
        offset += len(contexts_from_db) # Advance offset for the next DB fetch

    logger.info(f"Inference finished. Total contexts for which prediction was attempted: {processed_count}. Valid predictions added to DB: {added_count}")


if __name__ == "__main__":
    setup_logging() # Setup basic logging

    parser = argparse.ArgumentParser(description="Run span prediction inference using a fine-tuned Hugging Face QA model.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH,
                        help=f"Path to the SQLite database file (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the directory containing the saved Hugging Face model and tokenizer (output of training stage's save_pretrained).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INFERENCE_PARAMS.get('prediction_batch_size', 32),
                        help=f"Batch size for prediction (default: {DEFAULT_INFERENCE_PARAMS.get('prediction_batch_size', 32)})")
    parser.add_argument("--max-len", type=int, default=DEFAULT_INFERENCE_PARAMS['max_len'],
                        help=f"Max sequence length for model (default: {DEFAULT_INFERENCE_PARAMS['max_len']})")
    parser.add_argument("--max-before-tokens", type=int, default=DEFAULT_INFERENCE_PARAMS['max_before_tokens'],
                        help=f"Max tokens from context before [IMG] (default: {DEFAULT_INFERENCE_PARAMS['max_before_tokens']})")

    args = parser.parse_args()

    if not os.path.isdir(args.model_path): # Model path must be a directory
        logger.error(f"Model path not found or is not a directory: {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.db_path):
        logger.error(f"Database path not found: {args.db_path}")
        sys.exit(1)
    
    run_inference(
        db_path=args.db_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_before_tokens=args.max_before_tokens
    )
    logger.info("Inference script finished.")
