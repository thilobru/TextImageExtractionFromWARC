# src/training/data_loader.py
import numpy as np
import json
from transformers import DistilBertTokenizer
import logging
import re

logger = logging.getLogger(__name__)

def find_sublist_indices(main_list, sub_list):
    """Finds the start and end indices of the first occurrence of sub_list in main_list."""
    n = len(main_list)
    m = len(sub_list)
    if m == 0: return None, None # Cannot find empty sublist
    for i in range(n - m + 1):
        if main_list[i:i + m] == sub_list:
            return i, i + m - 1
    return None, None

def prepare_training_data(db_rows, tokenizer: DistilBertTokenizer, max_len):
    """
    Prepares training data (tokens, masks, labels) from database rows.

    Args:
        db_rows (list): List of Row objects from the ImageContext table.
        tokenizer (DistilBertTokenizer): The tokenizer instance.
        max_len (int): Maximum sequence length.

    Returns:
        tuple: (x_tokens, x_masks, y_starts, y_ends) as numpy arrays.
               Returns None if input is empty or invalid.
    """
    if not db_rows:
        logger.warning("No database rows provided for training data preparation.")
        return None, None, None, None

    img_token = "[IMG]"
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
        before_text = row['context_before'] or ""
        after_text = row['context_after'] or ""
        alt_text = row['alt_text'] or ""

        if not alt_text: # Should not happen for training candidates, but check
            rejected_count += 1
            continue

        # Tokenize components separately first
        # Add space before/after IMG? Depends on how tokenizer handles it
        before_tokens = tokenizer.encode(before_text, add_special_tokens=False)
        after_tokens = tokenizer.encode(after_text, add_special_tokens=False)
        alt_tokens = tokenizer.encode(alt_text, add_special_tokens=False) # Target span

        if not alt_tokens:
            rejected_count += 1
            continue

        # Construct the sequence: [CLS] before [IMG] after [SEP]
        # Adjust truncation to prioritize context around [IMG]
        # Simple truncation for now:
        # Allow space for [CLS], [IMG], [SEP] -> max_len - 3 tokens available
        available_len = max_len - 3
        max_before_len = available_len // 2 # Rough split
        max_after_len = available_len - max_before_len

        truncated_before = before_tokens[-max_before_len:] # Take end part of 'before'
        truncated_after = after_tokens[:max_after_len]    # Take start part of 'after'

        input_ids = [cls_token_id] + truncated_before + [img_token_id] + truncated_after + [sep_token_id]

        # Find the token indices of the alt_text within the *original combined text*
        # This is tricky because we truncated. A safer way: check if alt_tokens
        # are fully contained within the *truncated* sequence used.
        # For simplicity now, we assume the original DB check was sufficient.
        # Re-find alt_tokens within input_ids (excluding special tokens for search)
        search_sequence = truncated_before + [img_token_id] + truncated_after

        logger.debug(f"Context ID: {row['context_id']}")
        logger.debug(f"Alt Text: '{alt_text}'")
        logger.debug(f"Alt Tokens: {alt_tokens}")
        # Decode search sequence for easier reading (might be long)
        try:
            decoded_search = tokenizer.decode(search_sequence)
            logger.debug(f"Search Sequence (decoded): '{decoded_search[:500]}...'") # Log prefix
        except: # Handle potential decoding errors
            logger.debug(f"Search Sequence (tokens): {search_sequence}")

        start_idx_in_search, end_idx_in_search = find_sublist_indices(search_sequence, alt_tokens)

        if start_idx_in_search is None:
            logger.warning(f"Alt tokens NOT FOUND in search sequence for context {row['context_id']}")
            # Maybe log original sequence too for comparison if needed
            original_search = before_tokens + [img_token_id] + after_tokens
            logger.debug(f"Original search sequence had length: {len(original_search)}")

            # Alt text might have been truncated away or wasn't found by regex originally
            # Let's try searching the original, untruncated sequence (less reliable for labels)
            # original_search = before_tokens + [img_token_id] + after_tokens
            # start_orig, end_orig = find_sublist_indices(original_search, alt_tokens)
            # if start_orig is not None:
            #     logger.warning(f"Alt text found in original but not truncated sequence for context {row['context_id']}")
            # else:
            #      logger.warning(f"Alt text tokens not found even in original sequence for context {row['context_id']}")
            rejected_count += 1
            continue # Skip if alt text tokens not found in the used context

        # Adjust indices to account for the [CLS] token at the beginning
        start_idx_final = start_idx_in_search + 1 # +1 for [CLS]
        end_idx_final = end_idx_in_search + 1

        # Ensure indices are within the sequence length *before* padding
        current_len = len(input_ids)
        if start_idx_final >= current_len or end_idx_final >= current_len:
             logger.warning(f"Calculated indices out of bounds before padding for context {row['context_id']}")
             rejected_count += 1
             continue

        # Pad sequence
        padding_len = max_len - current_len
        attention_mask = [1] * current_len + [0] * padding_len
        input_ids = input_ids + [pad_token_id] * padding_len

        # Validation (optional but good)
        if len(input_ids) != max_len or len(attention_mask) != max_len:
             logger.error(f"Padding error for context {row['context_id']}")
             rejected_count +=1
             continue
        if not (0 <= start_idx_final < max_len and 0 <= end_idx_final < max_len):
            logger.error(f"Final indices out of bounds for context {row['context_id']}")
            rejected_count += 1
            continue
        # Check if decoded span matches roughly
        # decoded_span = tokenizer.decode(input_ids[start_idx_final : end_idx_final + 1])
        # if alt_text.lower() not in decoded_span.lower():
        #      logger.warning(f"Decoded span '{decoded_span}' doesn't match alt '{alt_text}' for {row['context_id']}")
             # This might happen due to tokenization artifacts, decide if it's critical

        x_tokens_list.append(input_ids)
        x_masks_list.append(attention_mask)
        y_starts_list.append(start_idx_final)
        y_ends_list.append(end_idx_final)

    logger.info(f"Prepared training data: {len(x_tokens_list)} examples, {rejected_count} rejected.")

    if not x_tokens_list:
        return None, None, None, None

    return (
        np.array(x_tokens_list, dtype=np.int32),
        np.array(x_masks_list, dtype=np.int32),
        np.array(y_starts_list, dtype=np.int32),
        np.array(y_ends_list, dtype=np.int32)
    )