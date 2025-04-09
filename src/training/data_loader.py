# src/training/data_loader.py
import numpy as np
import json
from transformers import DistilBertTokenizer
import logging
import re # Keep re if needed elsewhere, maybe not needed now

logger = logging.getLogger(__name__)

# find_sublist_indices is no longer needed for the core logic here
# def find_sublist_indices(main_list, sub_list): ...

def prepare_training_data(db_rows, tokenizer: DistilBertTokenizer, max_len):
    """
    Prepares training data (tokens, masks, labels) from database rows.
    Uses calculated indices based on token lengths instead of sublist search.
    """
    if not db_rows:
        logger.warning("No database rows provided for training data preparation.")
        return None, None, None, None

    img_token = "[IMG]"
    # Ensure token exists
    if img_token not in tokenizer.additional_special_tokens:
         logger.error(f"Tokenizer missing special token: {img_token}. Add it before calling.")
         return None, None, None, None # Fail if token missing

    img_token_id = tokenizer.convert_tokens_to_ids(img_token)
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    x_tokens_list = []
    x_masks_list = []
    y_starts_list = []
    y_ends_list = []
    rejected_count = 0

    for row in db_rows:
        # Use .get() for safety, although columns should exist
        context_id = row.get('context_id', 'UNKNOWN')
        before_text = row.get('context_before', "") or ""
        after_text = row.get('context_after', "") or ""
        alt_text = row.get('alt_text', "") or ""

        if not alt_text:
            logger.debug(f"Skipping context {context_id}: Missing alt_text.")
            rejected_count += 1
            continue

        # Tokenize components separately
        before_tokens = tokenizer.encode(before_text, add_special_tokens=False)
        after_tokens = tokenizer.encode(after_text, add_special_tokens=False)
        alt_tokens = tokenizer.encode(alt_text, add_special_tokens=False) # Target span

        if not alt_tokens:
            logger.debug(f"Skipping context {context_id}: Alt text '{alt_text}' tokenized to empty list.")
            rejected_count += 1
            continue

        # Define available length for context tokens (excluding CLS, IMG, SEP)
        available_len_for_context = max_len - 3
        if available_len_for_context <= 0:
             logger.warning(f"max_len {max_len} too small. Skipping context {context_id}")
             rejected_count += 1
             continue

        # Prioritize context around IMG. Truncate intelligently.
        # Calculate how much space needed for alt_tokens if they were included (for conceptual planning)
        # However, the actual input uses [IMG], so we just split available context space.
        max_before_len = available_len_for_context // 2
        max_after_len = available_len_for_context - max_before_len

        truncated_before = before_tokens[-max_before_len:] # Take end part of 'before'
        truncated_after = after_tokens[:max_after_len]    # Take start part of 'after'

        # Construct the final input sequence with the [IMG] marker
        input_ids_list = [cls_token_id] + truncated_before + [img_token_id] + truncated_after + [sep_token_id]
        current_len = len(input_ids_list)

        # --- Start Revised Index Calculation & Verification ---
        # Calculate where the alt text *should* start and end.
        # The target span corresponds conceptually to the [IMG] marker's location + the alt_text length.
        # Start index is the position of the [IMG] marker.
        start_idx_final = 1 + len(truncated_before) # Index after [CLS] and truncated_before
        # End index is start index + length of alt_tokens - 1
        end_idx_final = start_idx_final + len(alt_tokens) - 1

        # --- Verification Step ---
        # Check 1: Does the calculated span END within the actual sequence length (before padding)?
        # If end_idx_final is too large, it means the alt_text itself wouldn't fit even if context was minimal.
        if end_idx_final >= current_len:
            logger.warning(f"Context {context_id}: Calculated end index ({end_idx_final}) is out of bounds for actual sequence length ({current_len}). Alt text likely too long or context too short. Skipping.")
            rejected_count += 1
            continue

        # Check 2: Are the start/end indices valid? (Should always be true if Check 1 passes, but good sanity check)
        if not (0 <= start_idx_final < current_len and 0 <= end_idx_final < current_len and start_idx_final <= end_idx_final):
             logger.error(f"Context {context_id}: Invalid calculated indices: start={start_idx_final}, end={end_idx_final}, len={current_len}. Skipping.")
             rejected_count += 1
             continue

        # Check 3 (Optional but Recommended): Verify token match at the calculated span.
        # This catches subtle tokenization differences.
        # span_tokens_in_input = input_ids_list[start_idx_final : end_idx_final + 1]
        # if span_tokens_in_input != alt_tokens:
        #      logger.warning(f"Context {context_id}: Token mismatch between calculated span and alt_tokens. Skipping.")
        #      logger.debug(f"Expected Alt Tokens: {alt_tokens}")
        #      logger.debug(f"Found Span Tokens: {span_tokens_in_input}")
        #      rejected_count += 1
        #      continue
        # --- End Verification Step ---

        # Pad sequence
        padding_len = max_len - current_len
        attention_mask = [1] * current_len + [0] * padding_len
        input_ids = input_ids_list + [pad_token_id] * padding_len

        # Final validation checks on padded sequence length
        if len(input_ids) != max_len or len(attention_mask) != max_len:
             logger.error(f"Padding error for context {context_id}: Final length mismatch.")
             rejected_count +=1
             continue

        x_tokens_list.append(input_ids)
        x_masks_list.append(attention_mask)
        y_starts_list.append(start_idx_final)
        y_ends_list.append(end_idx_final)

    logger.info(f"Prepared training data: {len(x_tokens_list)} examples, {rejected_count} rejected.")

    if not x_tokens_list:
        # Return empty arrays instead of None if no valid data was prepared
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.array(x_tokens_list, dtype=np.int32),
        np.array(x_masks_list, dtype=np.int32),
        np.array(y_starts_list, dtype=np.int32),
        np.array(y_ends_list, dtype=np.int32)
    )
