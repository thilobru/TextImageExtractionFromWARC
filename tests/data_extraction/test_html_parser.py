import pytest
import os
import sys

# Make sure src directory is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Module to test
from src.data_extraction.html_parser import parse_html_page

# --- Test Cases for Intention B ---
# Intention B: 'found_alt_in_context' should only be True if the alt text
#              appears naturally in the surrounding text nodes (ignoring
#              any text inserted via alt_texts=True).

def test_alt_only_in_attribute():
    """
    Tests case B1: Alt text is ONLY in the attribute, not surrounding tags.
    EXPECT: found_alt_in_context == False
    """
    html_content = b"""
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
        <p>Some text before the image.</p>
        <img src="image1.jpg" alt="A unique description only in alt." />
        <p>Some text after the image.</p>
    </body>
    </html>
    """
    page_url = "http://example.com/page"
    # EXPECTATION CHANGED HERE: found_alt_in_context is now False
    expected_output = [
        {
            'image_url': 'http://example.com/image1.jpg',
            'alt_text': 'A unique description only in alt.',
            'context_before': 'Some text before the image.', # Context still extracted
            'context_after': 'Some text after the image.',  # Context still extracted
            'found_alt_in_context': False # <<<< EXPECTATION CHANGED
        }
    ]
    # Assuming min_alt_len=10, min_context_words=5
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=5)

    assert len(result) == 1
    assert result[0]['image_url'] == expected_output[0]['image_url']
    assert result[0]['alt_text'] == expected_output[0]['alt_text']
    # Basic context check (implementation might vary slightly)
    assert "text before" in result[0]['context_before']
    assert "text after" in result[0]['context_after']
    # Assert the new expectation for the flag
    assert result[0]['found_alt_in_context'] == expected_output[0]['found_alt_in_context'], \
        "Test Failed: Expected found_alt_in_context to be False when alt is only in attribute."


def test_alt_present_in_context():
    """
    Tests case B2: Alt text is in attribute AND naturally in surrounding text.
    EXPECT: found_alt_in_context == True
    """
    alt_description = "The specific description is here."
    html_content = f"""
    <!DOCTYPE html>
    <html><body>
        <p>Text before. {alt_description} This is related text.</p>
        <img src="image2.png" alt="{alt_description}" />
        <p>Text after.</p>
    </body></html>
    """.encode('utf-8') # Encode to bytes
    page_url = "http://example.com/alt_in_context"
    expected_output = [
        {
            'image_url': 'http://example.com/image2.png',
            'alt_text': alt_description,
            'context_before': f'Text before. {alt_description} This is related text.', # Expected context
            'context_after': 'Text after.', # Expected context
            'found_alt_in_context': True # <<<< EXPECTATION
        }
    ]
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=5)

    assert len(result) == 1
    assert result[0]['image_url'] == expected_output[0]['image_url']
    assert result[0]['alt_text'] == expected_output[0]['alt_text']
    # Check if the flag is correctly True
    assert result[0]['found_alt_in_context'] == expected_output[0]['found_alt_in_context'], \
        "Test Failed: Expected found_alt_in_context to be True when alt is also in surrounding text."


def test_alt_present_in_context_normalized():
    """
    Tests case B4: Alt text in attribute and context differ slightly (case, punctuation).
    EXPECT: found_alt_in_context == True (due to normalization in the check)
    """
    alt_description = "Description with Title Case and Punctuation!"
    context_text = "description with title case and punctuation" # Lowercase, no punctuation
    html_content = f"""
    <!DOCTYPE html>
    <html><body>
        <p>Before... {context_text} ...more text.</p>
        <img src="image3.gif" alt="{alt_description}" />
        <p>After.</p>
    </body></html>
    """.encode('utf-8')
    page_url = "http://example.com/alt_normalized"
    expected_output = [
        {
            'image_url': 'http://example.com/image3.gif',
            'alt_text': alt_description,
            'context_before': f'Before... {context_text} ...more text.',
            'context_after': 'After.',
            'found_alt_in_context': True # <<<< EXPECTATION (should match after normalization)
        }
    ]
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=5)

    assert len(result) == 1
    assert result[0]['image_url'] == expected_output[0]['image_url']
    assert result[0]['alt_text'] == expected_output[0]['alt_text']
    assert result[0]['found_alt_in_context'] == expected_output[0]['found_alt_in_context'], \
        "Test Failed: Expected found_alt_in_context to be True when normalized alt matches context."


# --- Keep other tests, modify expectations if needed ---

def test_no_images():
    """Tests HTML with no images. EXPECT: Empty list."""
    html_content = b"""
    <!DOCTYPE html>
    <html><body><p>Just text.</p></body></html>
    """
    page_url = "http://example.com/noimg"
    result = parse_html_page(html_content, page_url)
    assert result == []

def test_image_no_alt():
    """
    Tests an image with no alt attribute.
    EXPECT: found_alt_in_context == False
    """
    html_content = b"""
    <!DOCTYPE html>
    <html><body>
    <p>Before</p><img src="/images/pic.png" /><p>After</p>
    </body></html>
    """
    page_url = "http://example.com/noalt"
    result = parse_html_page(html_content, page_url, min_context_words=1)
    assert len(result) == 1 # Should still extract context
    assert result[0]['alt_text'] == ''
    assert result[0]['found_alt_in_context'] == False

def test_image_short_alt():
    """
    Tests an image with alt text shorter than min_alt_len.
    EXPECT: found_alt_in_context == False
    """
    html_content = b"""
    <!DOCTYPE html>
    <html><body>
    <p>Before</p><img src="short.gif" alt="short" /><p>After</p>
    </body></html>
    """
    page_url = "http://example.com/shortalt"
    # Assuming min_alt_len is 10
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=1)
    assert len(result) == 1
    assert result[0]['alt_text'] == 'short'
    assert result[0]['found_alt_in_context'] == False # Fails length check

def test_multiple_images_intention_b():
    """
    Tests multiple images, checking Intention B.
    EXPECT: found_alt_in_context is False for img1 (alt only in attr), True for img2 (alt in attr and context)
    """
    alt1 = "Alt text for first image unique"
    alt2 = "Alt text for second image also in text"
    html_content = f"""
    <!DOCTYPE html>
    <html><body>
    <h1>Title</h1>
    <p>Text A</p>
    <img src="img1.png" alt="{alt1}">
    <p>Text B is between images. It mentions: {alt2}.</p>
    <img src="img2.jpg" alt="{alt2}">
    <p>Text C</p>
    </body></html>
    """.encode('utf-8')
    page_url = "http://example.com/multi_b"
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=1)

    assert len(result) == 2

    # Check first image (img1.png) - Alt only in attribute
    img1_result = next((r for r in result if "img1.png" in r['image_url']), None)
    assert img1_result is not None
    assert img1_result['alt_text'] == alt1
    assert "Text A" in img1_result['context_before']
    assert "Text B is between images" in img1_result['context_after']
    assert img1_result['found_alt_in_context'] == False # <<<< EXPECTATION CHANGED

    # Check second image (img2.jpg) - Alt in attribute AND context
    img2_result = next((r for r in result if "img2.jpg" in r['image_url']), None)
    assert img2_result is not None
    assert img2_result['alt_text'] == alt2
    assert "Text B is between images" in img2_result['context_before']
    assert "Text C" in img2_result['context_after']
    assert img2_result['found_alt_in_context'] == True # <<<< EXPECTATION


def test_min_context_words_filter_intention_b():
    """Tests filtering based on minimum context words (should still work)."""
    html_content = b"""
    <!DOCTYPE html>
    <html><body>
    <p>Short</p>
    <img src="image.png" alt="Valid alt text here not in context">
    <p>Context</p>
    </body></html>
    """
    page_url = "http://example.com/shortcontext_b"
    # Test with min_context_words=5, the context "Short Context" has 2 words
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=5)
    assert result == [] # Should be filtered out by word count

    # Test with min_context_words=2
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=2)
    assert len(result) == 1 # Should pass word count
    assert result[0]['found_alt_in_context'] == False # Alt wasn't in context anyway
