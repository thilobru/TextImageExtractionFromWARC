# src/enrichment/image_downloader.py
import os
import requests
from PIL import Image, UnidentifiedImageError
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def download_and_validate_image(image_url, output_dir, min_size_bytes, min_resolution, timeout):
    """
    Downloads an image, validates its size and resolution, and saves it.

    Args:
        image_url (str): The URL of the image to download.
        output_dir (str): The directory to save the valid image.
        min_size_bytes (int): Minimum file size in bytes.
        min_resolution (tuple): Minimum resolution (width, height).
        timeout (int): Download timeout in seconds.

    Returns:
        str: The path to the saved image if successful, None otherwise.
    """
    if not image_url or not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
        logger.warning(f"Invalid image URL provided: {image_url}")
        return None

    try:
        # Generate a safe filename from the URL path
        parsed_url = urlparse(image_url)
        # Basic sanitization: take filename, replace non-alphanumeric
        base_name = os.path.basename(parsed_url.path)
        safe_base_name = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in base_name)
        if not safe_base_name or safe_base_name.startswith('.'): # Handle cases like '/' or '.xyz'
             # Fallback using hash or unique ID if needed, for now use simple fallback
             safe_base_name = f"image_{abs(hash(image_url))}.jpg" # Example fallback

        # Add extension if missing (guess jpg for now)
        if '.' not in safe_base_name:
             safe_base_name += ".jpg"

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, safe_base_name)

        # Avoid re-downloading if file exists (optional, depends on workflow)
        # if os.path.exists(output_path):
        #     logger.debug(f"Image already exists: {output_path}")
        #     # Optionally re-validate existing file here
        #     return output_path

        logger.debug(f"Attempting to download {image_url} to {output_path}")
        response = requests.get(image_url, stream=True, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'}) # Add User-Agent
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check size before saving fully
        content_length = int(response.headers.get('content-length', 0))
        if content_length < min_size_bytes:
            logger.debug(f"Image too small ({content_length} bytes) for {image_url}. Skipping.")
            return None

        # Save the image content
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.debug(f"Image content saved to {output_path}")

        # Validate the downloaded image resolution and integrity
        try:
            with Image.open(output_path) as img:
                width, height = img.size
                if width < min_resolution[0] or height < min_resolution[1]:
                    logger.debug(f"Image resolution too low ({width}x{height}) for {image_url}. Deleting.")
                    os.remove(output_path)
                    return None
                # Optional: Check format if needed (e.g., img.format in ['JPEG', 'PNG'])
                logger.info(f"Successfully downloaded and validated image: {output_path}")
                return output_path # Success!
        except UnidentifiedImageError:
            logger.warning(f"Cannot identify image file (possibly corrupted download): {output_path}. Deleting.")
            if os.path.exists(output_path): os.remove(output_path)
            return None
        except Exception as img_err: # Catch other PIL errors
             logger.warning(f"Error processing image file {output_path}: {img_err}. Deleting.")
             if os.path.exists(output_path): os.remove(output_path)
             return None

    except requests.exceptions.Timeout:
        logger.warning(f"Request for {image_url} timed out after {timeout} seconds.")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.warning(f"Failed to download {image_url}: {req_err}")
        # Clean up partial download if it exists
        if 'output_path' in locals() and os.path.exists(output_path):
             try:
                 os.remove(output_path)
             except OSError:
                 pass
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading {image_url}: {e}", exc_info=True)
        if 'output_path' in locals() and os.path.exists(output_path):
             try:
                 os.remove(output_path)
             except OSError:
                 pass
        return None

