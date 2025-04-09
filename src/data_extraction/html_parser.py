# src/data_extraction/html_parser.py
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
from resiliparse.parse.lang import detect_fast
from resiliparse.parse.html import HTMLTree
from urllib.parse import urljoin
import re
import logging

logger = logging.getLogger(__name__)

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

        for i, img_tag in enumerate(limited_img_elements):
            src = img_tag.getattr("src")
            if not src:
                continue # Skip images without src

            img_url = urljoin(page_url, src)
            original_alt = img_tag.getattr("alt") or ""

            # Replace alt attribute content with placeholder *before* text extraction
            placeholder = f"###IMG{i}###"
            img_tag.setattr("alt", placeholder)

            images_data.append({
                'id': i,
                'placeholder': placeholder,
                'image_url': img_url,
                'original_alt': original_alt
            })

        if not images_data:
            return [] # No valid images found

        # --- Step 2: Extract plain text with placeholders ---
        # Note: Using main_content=True might sometimes exclude headers/footers
        # where relevant context could be. Consider main_content=False if needed.
        text = extract_plain_text(tree, preserve_formatting=False,
                                  main_content=False, list_bullets=False,
                                  alt_texts=True, # Includes the placeholders we inserted
                                  links=False, form_fields=False, noscript=False)

        if not text or text.isspace():
            logger.debug(f"No extractable text found in {page_url}")
            return []

        # --- Step 3: Language Detection ---
        lang_code, lang_score = detect_fast(text) # Using detect_fast instead of detect
        # Adjust threshold? Lower score is better/more confident.
        # Maybe check score < 1000? Depends on resiliparse version/details.
        if lang_code != 'en': # or lang_score > 1200: # Removed score check for now
            logger.debug(f"Skipping non-English page {page_url} (Detected: {lang_code}, Score: {lang_score})")
            return []

        # --- Step 4: Split text by placeholders and expand context ---
        # Create regex pattern dynamically based on placeholders found
        placeholder_pattern = "|".join(re.escape(img['placeholder']) for img in images_data)
        if not placeholder_pattern:
             return [] # Should not happen if images_data is populated

        # Split text using the placeholders as delimiters
        # The regex captures text segments *between* placeholders (or start/end)
        # This approach is simpler than finditer and manually managing segments
        segments = re.split(f'({placeholder_pattern})', text)

        # Reconstruct context around each image
        context_map = {} # placeholder -> {'before': str, 'after': str}
        current_before = ""
        for k in range(len(segments)):
            segment = segments[k].strip()
            is_placeholder = any(segment == img['placeholder'] for img in images_data)

            if is_placeholder:
                 # The text *before* this placeholder is `current_before`
                 # The text *after* this placeholder starts with the *next* segment
                 after_context_list = []
                 for l in range(k + 1, len(segments)):
                     next_segment = segments[l].strip()
                     next_is_placeholder = any(next_segment == img['placeholder'] for img in images_data)
                     if next_is_placeholder:
                         break # Stop at the next image
                     if next_segment:
                        after_context_list.append(next_segment)

                 context_map[segment] = {
                     'before': current_before[-max_context_chars:], # Limit length
                     'after': " ".join(after_context_list)[:max_context_chars] # Limit length
                 }
                 current_before = "" # Reset for text after this image
            else:
                 # Append non-placeholder text to the current 'before' context
                 if segment:
                     current_before += " " + segment if current_before else segment


        # --- Step 5: Assemble results and check conditions ---
        for img_info in images_data:
            placeholder = img_info['placeholder']
            if placeholder in context_map:
                context = context_map[placeholder]
                context_before = context['before']
                context_after = context['after']
                original_alt = img_info['original_alt']

                # Basic check for minimum context length
                combined_context = context_before + " " + context_after
                if len(re.findall(r'\w+', combined_context)) < min_context_words:
                    continue # Skip if not enough surrounding words

                # Check if original alt text (if decent length) is present
                # This check is primarily for identifying training candidates (Stage 1)
                found_alt = False
                if original_alt and len(original_alt) >= min_alt_len:
                     # Simple substring check (case-insensitive)
                     if original_alt.lower() in combined_context.lower():
                         found_alt = True

                results.append({
                    'image_url': img_info['image_url'],
                    'alt_text': original_alt, # Store original alt text
                    'context_before': context_before,
                    'context_after': context_after,
                    'found_alt_in_context': found_alt
                })

    except Exception as e:
        logger.error(f"Error parsing {page_url}: {e}", exc_info=False) # Set exc_info=True for full traceback
        return []

    return results