import pytest
import os
import sys

# Make sure src directory is in path for imports
# Adjust based on how you run pytest (e.g., from project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Module to test
from src.data_extraction.html_parser import parse_html_page

# --- Test Cases ---

def test_basic_image_extraction():
    """Tests extracting a single image with alt text and context."""
    html_content = b"""
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
        <p>Some text before the image.</p>
        <img src="image1.jpg" alt="A description of image one, long enough." />
        <p>Some text after the image.</p>
    </body>
    </html>
    """
    page_url = "http://example.com/page"
    expected_output = [
        {
            'image_url': 'http://example.com/image1.jpg',
            'alt_text': 'A description of image one, long enough.',
            'context_before': 'Some text before the image.',
            'context_after': 'Some text after the image.',
            'found_alt_in_context': True # Alt text is present in the combined context
        }
    ]
    # Use default parameters from html_parser if possible, or specify here
    # Assuming min_alt_len=10, min_context_words=5 (adjust as needed for test)
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=5)

    assert len(result) == 1
    # Compare relevant fields, context might have minor whitespace diffs depending on parser
    assert result[0]['image_url'] == expected_output[0]['image_url']
    assert result[0]['alt_text'] == expected_output[0]['alt_text']
    assert "text before" in result[0]['context_before']
    assert "text after" in result[0]['context_after']
    assert result[0]['found_alt_in_context'] == expected_output[0]['found_alt_in_context']


def test_no_images():
    """Tests HTML with no images."""
    html_content = b"""
    <!DOCTYPE html>
    <html><body><p>Just text.</p></body></html>
    """
    page_url = "http://example.com/noimg"
    result = parse_html_page(html_content, page_url)
    assert result == []

def test_image_no_alt():
    """Tests an image with no alt attribute."""
    html_content = b"""
    <!DOCTYPE html>
    <html><body>
    <p>Before</p><img src="/images/pic.png" /><p>After</p>
    </body></html>
    """
    page_url = "http://example.com/noalt"
    expected_output = [
        {
            'image_url': 'http://example.com/images/pic.png',
            'alt_text': '',
            'context_before': 'Before',
            'context_after': 'After',
            'found_alt_in_context': False
        }
    ]
    result = parse_html_page(html_content, page_url, min_context_words=1)
    assert len(result) == 1
    assert result[0]['image_url'] == expected_output[0]['image_url']
    assert result[0]['alt_text'] == expected_output[0]['alt_text']
    assert result[0]['found_alt_in_context'] == expected_output[0]['found_alt_in_context']

def test_image_short_alt():
    """Tests an image with alt text shorter than min_alt_len."""
    html_content = b"""
    <!DOCTYPE html>
    <html><body>
    <p>Before</p><img src="short.gif" alt="short" /><p>After</p>
    </body></html>
    """
    page_url = "http://example.com/shortalt"
    expected_output = [
        {
            'image_url': 'http://example.com/short.gif',
            'alt_text': 'short', # Alt text is extracted
            'context_before': 'Before',
            'context_after': 'After',
            'found_alt_in_context': False # But not marked as found due to length
        }
    ]
    # Assuming default min_alt_len is 10
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=1)
    assert len(result) == 1
    assert result[0]['image_url'] == expected_output[0]['image_url']
    assert result[0]['alt_text'] == expected_output[0]['alt_text']
    assert result[0]['found_alt_in_context'] == expected_output[0]['found_alt_in_context']

def test_multiple_images():
    """Tests multiple images on a page."""
    html_content = b"""
    <!DOCTYPE html>
    <html><body>
    <h1>Title</h1>
    <p>Text A</p>
    <img src="img1.png" alt="Alt text for first image long enough">
    <p>Text B is between images.</p>
    <img src="img2.jpg" alt="Alt text for second image also long enough">
    <p>Text C</p>
    </body></html>
    """
    page_url = "http://example.com/multi"
    # Expected order might depend on implementation details of context extraction
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=1)

    assert len(result) == 2

    # Check first image (img1.png)
    img1_result = next((r for r in result if "img1.png" in r['image_url']), None)
    assert img1_result is not None
    assert img1_result['alt_text'] == "Alt text for first image long enough"
    assert "Text A" in img1_result['context_before']
    assert "Text B is between images." in img1_result['context_after']
    assert img1_result['found_alt_in_context'] == True # Alt is in combined context

    # Check second image (img2.jpg)
    img2_result = next((r for r in result if "img2.jpg" in r['image_url']), None)
    assert img2_result is not None
    assert img2_result['alt_text'] == "Alt text for second image also long enough"
    assert "Text B is between images." in img2_result['context_before']
    assert "Text C" in img2_result['context_after']
    assert img2_result['found_alt_in_context'] == True # Alt is in combined context

def test_min_context_words_filter():
    """Tests filtering based on minimum context words."""
    html_content = b"""
    <!DOCTYPE html>
    <html><body>
    <p>Short</p>
    <img src="image.png" alt="Valid alt text here">
    <p>Context</p>
    </body></html>
    """
    page_url = "http://example.com/shortcontext"
    # Test with min_context_words=5, the context "Short Context" has 2 words
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=5)
    assert result == [] # Should be filtered out

    # Test with min_context_words=2
    result = parse_html_page(html_content, page_url, min_alt_len=10, min_context_words=2)
    assert len(result) == 1 # Should pass


# --- Add more test cases ---
# - Test max_images_per_page limit
# - Test relative vs absolute image URLs
# - Test different HTML structures (image inside link, figure/figcaption)
# - Test character encoding issues if possible
# - Test context boundary conditions (image at start/end of body)