# src/training/data_loader.py
import numpy as np
import json
from transformers import DistilBertTokenizer
import logging
import re

logger = logging.getLogger(__name__)

def prepare_training_data(db_rows, tokenizer: DistilBertTokenizer, max_len):
    """
    Prepares training data (tokens, masks, labels) from database rows.
    Uses calculated indices based on token lengths and corrected truncation.
    """
    if db_rows is None:
         logger.warning("Received None for db_rows.")
         return np.array([]), np.array([]), np.array([]), np.array([])
    if not db_rows:
        logger.warning("No database rows provided for training data preparation.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    img_token = "[IMG]"
    if img_token not in tokenizer.additional_special_tokens:
         logger.error(f"Tokenizer missing special token: {img_token}.")
         return np.array([]), np.array([]), np.array([]), np.array([])

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
        try:
            context_id = row['context_id']
            before_text = row['context_before'] or ""
            after_text = row['context_after'] or ""
            alt_text = row['alt_text'] or ""
        except (KeyError, IndexError) as e:
             logger.error(f"Error accessing expected column in row: {e}. Row data: {dict(row)}")
             rejected_count += 1
             continue

        if not alt_text:
            logger.debug(f"Skipping context {context_id}: Missing alt_text.")
            rejected_count += 1
            continue

        # Tokenize components separately
        before_tokens = tokenizer.encode(before_text, add_special_tokens=False)
        after_tokens = tokenizer.encode(after_text, add_special_tokens=False)
        alt_tokens = tokenizer.encode(alt_text, add_special_tokens=False)

        if not alt_tokens:
            logger.debug(f"Skipping context {context_id}: Alt text '{alt_text}' tokenized to empty list.")
            rejected_count += 1
            continue

        # --- Start Corrected Truncation Logic ---
        # Calculate available length for context (before + after)
        # Need space for [CLS], [IMG], [SEP] AND the alt_tokens themselves if we verify span content later
        # For index calculation, we only need space for CLS, IMG, SEP
        len_alt_tokens = len(alt_tokens)
        # Check if alt_tokens alone are too long (won't fit even with no context)
        if len_alt_tokens > max_len - 3:
             logger.warning(f"Context {context_id}: Alt tokens ({len_alt_tokens}) longer than max_len-3 ({max_len-3}). Skipping.")
             rejected_count += 1
             continue

        # Available length for actual before/after context tokens
        available_len_for_context = max_len - 3 # Space for CLS, IMG, SEP
        if available_len_for_context < 0: available_len_for_context = 0 # Handle very small max_len

        # Split available context space, prioritizing context around IMG
        max_before_len = available_len_for_context // 2
        max_after_len = available_len_for_context - max_before_len

        truncated_before = before_tokens[-max_before_len:] # Take end part of 'before'
        truncated_after = after_tokens[:max_after_len]    # Take start part of 'after'
        # --- End Corrected Truncation Logic ---


        # Construct the final input sequence with the [IMG] marker
        input_ids_list = [cls_token_id] + truncated_before + [img_token_id] + truncated_after + [sep_token_id]
        current_len = len(input_ids_list)

        # Calculate indices
        start_idx_final = 1 + len(truncated_before) # Index of [IMG] marker
        end_idx_final = start_idx_final + len_alt_tokens - 1

        # --- Verification Step ---
        # Check 1: Does the calculated span END within the actual sequence length (before padding)?
        # This check should fail much less often now with corrected truncation logic.
        if end_idx_final >= current_len:
            # This might still happen if alt_tokens is very long and pushes the end beyond SEP
            # even when available_len_for_context was calculated correctly.
            logger.warning(f"Context {context_id}: Calculated end index ({end_idx_final}) >= current_len ({current_len}) even after truncation adjustment. Skipping.")
            rejected_count += 1
            continue

        # Check 2: Are the start/end indices valid?
        if not (0 <= start_idx_final < current_len and 0 <= end_idx_final < current_len and start_idx_final <= end_idx_final):
             logger.error(f"Context {context_id}: Invalid calculated indices: start={start_idx_final}, end={end_idx_final}, len={current_len}. Skipping.")
             rejected_count += 1
             continue

        # Check 3 (Optional): Verify token match - uncomment if needed
        # span_tokens_in_input = input_ids_list[start_idx_final : end_idx_final + 1]
        # if span_tokens_in_input != alt_tokens: ...

        # Pad sequence
        padding_len = max_len - current_len
        attention_mask = [1] * current_len + [0] * padding_len
        input_ids = input_ids_list + [pad_token_id] * padding_len

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
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.array(x_tokens_list, dtype=np.int32),
        np.array(x_masks_list, dtype=np.int32),
        np.array(y_starts_list, dtype=np.int32),
        np.array(y_ends_list, dtype=np.int32)
    )
