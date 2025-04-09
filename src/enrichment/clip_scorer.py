# src/enrichment/clip_scorer.py
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
import logging
import os

logger = logging.getLogger(__name__)

# Global cache for CLIP model and processor to avoid reloading frequently
# Be mindful of memory usage if running many processes
_clip_model_cache = {}
_clip_processor_cache = {}

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    """Loads CLIP model and processor, caching them."""
    global _clip_model_cache, _clip_processor_cache
    if model_name not in _clip_model_cache:
        logger.info(f"Loading CLIP model: {model_name}")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
            _clip_model_cache[model_name] = model
            _clip_processor_cache[model_name] = processor
            logger.info(f"CLIP model loaded on device: {device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model '{model_name}': {e}")
            raise # Re-raise to signal failure
    return _clip_model_cache[model_name], _clip_processor_cache[model_name]

def get_clip_score(image_path, text, model_name="openai/clip-vit-base-patch32"):
    """
    Calculates the CLIP score (cosine similarity) between an image and text.

    Args:
        image_path (str): Path to the local image file.
        text (str): The text description.
        model_name (str): The CLIP model identifier.

    Returns:
        float: The calculated CLIP score (cosine similarity * 100), or None if an error occurs.
               Note: Some implementations return similarity [0,1], others * 100.
               This returns similarity [0, 1] range approximately.
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided for CLIP scoring.")
        return None
    if not image_path or not os.path.exists(image_path):
        logger.warning(f"Image path invalid or not found: {image_path}")
        return None

    try:
        model, processor = load_clip_model(model_name)
        device = model.device

        # Open and process the image
        image = Image.open(image_path)
        # Ensure image is RGB (CLIP expects 3 channels)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image and text using the CLIPProcessor
        # Use processor's padding and truncation for text
        inputs = processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length", # Pad to max length (e.g., 77 tokens)
            truncation=True      # Truncate if longer
        ).to(device)

        # Get embeddings using torch.no_grad() for efficiency during inference
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds # Shape: (1, embedding_dim)
            text_embeds = outputs.text_embeds    # Shape: (1, embedding_dim)

        # Normalize embeddings (important for cosine similarity)
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)

        # Calculate cosine similarity
        # similarity = image_embeds @ text_embeds.T # Matrix multiplication for batch, but here it's 1x1
        similarity = torch.sum(image_embeds * text_embeds, dim=-1) # Element-wise product and sum

        # Return the similarity score (typically between ~0 and 1 after normalization)
        # Convert from tensor to float
        score = similarity.cpu().item()
        logger.debug(f"Calculated CLIP score for {os.path.basename(image_path)}: {score:.4f}")
        return score

    except (OSError, UnidentifiedImageError) as img_err:
        logger.warning(f"Error opening or processing image {image_path}: {img_err}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calculating CLIP score for {image_path}: {e}", exc_info=True)
        return None

