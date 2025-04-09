from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
from resiliparse.parse.lang import detect_fast
from resiliparse.parse.html import HTMLTree
from urllib.parse import urljoin
import re
import string
import logging

logger = logging.getLogger(__name__)

def _normalize_for_match(text):
    """Helper to normalize text for loose substring matching."""
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse ALL whitespace (spaces, tabs, newlines etc.) to single spaces
    text = re.sub(r'\s+', ' ', text).strip() # Use regex and strip ends
    return text

def parse_html_page(html_bytes, page_url, min_alt_len=10, max_images_per_page=20, min_context_words=30, max_context_chars=2500):
    """
    Parses HTML, extracts images, alt text, and surrounding context.

    Args:
        html_bytes: Raw HTML content as bytes.
        page_url: The URL of the HTML page.
        min_alt_len: Minimum character length for alt text to be considered.
        max_images_per_page: Max images to process from a single page.
        min_context_words: Min words required in combined before/after context.
        max_context_chars: Max characters for before/after context segments.

    Returns:
        A list of dictionaries, each containing:
        {
            'image_url': str,
            'alt_text': str or None,
            'context_before': str,
            'context_after': str,
            'found_alt_in_context': bool # For Stage 1 training data identification
        }
        Returns an empty list if parsing fails or no suitable images/context found.
    """
    results = []
    try:
        encoding = detect_encoding(html_bytes)
        tree = HTMLTree.parse_from_bytes(html_bytes, encoding)

        if tree.body is None:
            logger.debug(f"No body tag found in {page_url}")
            return []

        # --- Step 1: Identify images and original alt text ---
        images_data = []
        img_elements = tree.body.get_elements_by_tag_name("img")

        if not img_elements:
            logger.debug(f"No img tags found in {page_url}")
            return []

        limited_img_elements = img_elements[:max_images_per_page]
        placeholder_map = {} # Maps placeholder back to img_info index

        for i, img_tag in enumerate(limited_img_elements):
            src = img_tag.getattr("src")
            # Skip images without src OR data URIs for now (can be very long)
            if not src or src.startswith('data:image'):
                continue

            img_url = urljoin(page_url, src)
            original_alt = img_tag.getattr("alt") or ""

            placeholder = f"###IMG{i}###"
            img_tag.setattr("alt", placeholder) # Replace alt before text extraction

            img_info = {
                'id': i,
                'placeholder': placeholder,
                'image_url': img_url,
                'original_alt': original_alt
            }
            images_data.append(img_info)
            placeholder_map[placeholder] = i # Map placeholder to index in images_data

        if not images_data:
            return [] # No valid images found

        # --- Step 2: Extract plain text with placeholders ---
        text = extract_plain_text(tree, preserve_formatting=False,
                                  main_content=False, list_bullets=False, # Use main_content=False to get all text
                                  alt_texts=True, # Includes the placeholders
                                  links=False, form_fields=False, noscript=False)

        if not text or text.isspace():
            logger.debug(f"No extractable text found in {page_url}")
            return []

        # --- Step 3: Language Detection (Optional here, can be done later) ---
        # lang_code, lang_score = detect_fast(text)
        # if lang_code != 'en':
        #     logger.debug(f"Skipping non-English page {page_url} (Detected: {lang_code})")
        #     return [] # Filter early if desired

        # --- Step 4: Split text by placeholders and assign context ---
        # Create regex pattern dynamically based on placeholders actually used
        placeholder_pattern = "|".join(re.escape(img['placeholder']) for img in images_data)
        if not placeholder_pattern: return []

        # Split text using the placeholders as delimiters. Capturing keeps delimiters.
        segments = re.split(f'({placeholder_pattern})', text)

        # Assign context based on segments
        contexts = [{} for _ in images_data] # Initialize context dict for each image
        current_text_segment = ""

        for segment in segments:
            if not segment: continue # Skip empty segments

            if segment in placeholder_map:
                # This segment is a placeholder
                img_index = placeholder_map[segment]
                # Assign the accumulated text as the 'before' context for this image
                contexts[img_index]['before'] = current_text_segment.strip()[-max_context_chars:]

                # Reset for the next segment (which will be 'after' this image)
                current_text_segment = ""

                # Assign this placeholder as the 'start' for the 'after' context of the *previous* image (if any)
                # This requires iterating differently or post-processing, let's simplify:
                # We will assign 'after' context based on the text *following* the placeholder.
            else:
                # This segment is regular text
                current_text_segment += segment # Keep accumulating text

        # Post-process to assign 'after' context
        # Iterate through images_data, find placeholder in segments, look ahead
        for i, img_info in enumerate(images_data):
             placeholder = img_info['placeholder']
             try:
                 placeholder_segment_index = segments.index(placeholder)
             except ValueError:
                 # Placeholder wasn't found in split text (shouldn't happen if parsing worked)
                 contexts[i]['after'] = "" # Assign empty after context
                 continue

             after_text_list = []
             # Look at segments *after* the placeholder
             for k in range(placeholder_segment_index + 1, len(segments)):
                 segment = segments[k]
                 if segment in placeholder_map: # Stop if we hit the next placeholder
                     break
                 if segment: # Append non-empty text segments
                     after_text_list.append(segment.strip())

             # Join stripped segments with a single space
             contexts[i]['after'] = " ".join(after_text_list)[:max_context_chars]

             # If 'before' context wasn't assigned (e.g., image at start), set empty
             if 'before' not in contexts[i]:
                  contexts[i]['before'] = ""


        # --- Step 5: Assemble results and apply final filters ---
        for i, img_info in enumerate(images_data):
            context = contexts[i]
            # Ensure context was actually assigned (might not be if placeholder logic failed)
            if 'before' not in context or 'after' not in context:
                 continue

            context_before = context['before']
            context_after = context['after']
            original_alt = img_info['original_alt']

            # Check minimum context word count
            combined_context = context_before + " " + context_after
            word_count = len(re.findall(r'\w+', combined_context))
            if word_count < min_context_words:
                logger.debug(f"Context word count {word_count} below threshold for {img_info['image_url']}")
                continue # Skip if not enough surrounding words

            # Check if original alt text (if significant) is present in context
            # Use normalized versions for a more robust check
            found_alt = False
            if original_alt and len(original_alt) >= min_alt_len:
                 normalized_alt = _normalize_for_match(original_alt) # Uses module-level helper
                 normalized_context = _normalize_for_match(combined_context) # Uses module-level helper
                 # Check if non-empty normalized alt is in normalized context
                 if normalized_alt and normalized_alt in normalized_context:
                     found_alt = True

            results.append({
                'image_url': img_info['image_url'],
                'alt_text': original_alt,
                'context_before': context_before,
                'context_after': context_after,
                'found_alt_in_context': found_alt
            })

    except Exception as e:
        logger.error(f"Error parsing {page_url}: {e}", exc_info=False) # Set exc_info=True for full traceback
        return []

    return results
