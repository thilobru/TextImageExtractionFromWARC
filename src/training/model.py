# src/training/model.py
import tensorflow as tf
from tensorflow import keras # Keep for type hints if needed elsewhere
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer # Use AutoModel and AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def create_span_prediction_model(model_checkpoint="distilbert-base-uncased", tokenizer_vocab_size=None):
    """
    Loads a pre-trained Transformer model for Question Answering.

    Args:
        model_checkpoint (str): The Hugging Face model identifier.
        tokenizer_vocab_size (int, optional): Size of the tokenizer vocabulary (incl. special tokens).
                                              If None, model's original vocab size is assumed.
                                              Required if tokenizer was extended with special tokens.

    Returns:
        TFPreTrainedModel: Loaded Hugging Face model for Question Answering.
                           Model is NOT compiled here.
    """
    logger.info(f"Loading pre-trained QA model: {model_checkpoint}")
    try:
        # Load the specified QA model using TFAutoModelForQuestionAnswering
        model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

        # --- Handle Tokenizer Resizing ---
        # This is crucial if new tokens (like '[IMG]') were added to the tokenizer.
        # The model's token embedding layer needs to be resized to match.
        if tokenizer_vocab_size is not None:
            # Get the underlying base model (e.g., model.distilbert, model.bert) to resize embeddings
            # The attribute name for the base model can vary (e.g., 'distilbert', 'bert', 'roberta')
            # or it might be the model itself if it's not a "ForQuestionAnswering" head on top of a base.
            # TFAutoModelForQuestionAnswering might return a model that directly has resize_token_embeddings.
            
            current_vocab_size = model.config.vocab_size
            
            # Try to get the base model prefix (e.g. 'distilbert', 'bert')
            base_model_prefix = model.base_model_prefix 
            base_model = getattr(model, base_model_prefix, None) # e.g., model.distilbert or model.bert

            if base_model and hasattr(base_model, 'resize_token_embeddings'):
                if base_model.config.vocab_size != tokenizer_vocab_size:
                    logger.info(f"Base model original vocab size: {base_model.config.vocab_size}, Target tokenizer vocab size: {tokenizer_vocab_size}")
                    base_model.resize_token_embeddings(tokenizer_vocab_size)
                    # Also update the main model's config if necessary
                    model.config.vocab_size = tokenizer_vocab_size # Ensure the head model also knows the new size
                    logger.info(f"Resized base model token embeddings to {tokenizer_vocab_size}")
                else:
                    logger.info(f"Base model vocab size ({base_model.config.vocab_size}) already matches tokenizer vocab size ({tokenizer_vocab_size}). No resize needed.")
            elif hasattr(model, 'resize_token_embeddings'): # If the main model object has the method (e.g. T5ForConditionalGeneration)
                if model.config.vocab_size != tokenizer_vocab_size:
                    logger.info(f"Model original vocab size: {model.config.vocab_size}, Target tokenizer vocab size: {tokenizer_vocab_size}")
                    model.resize_token_embeddings(tokenizer_vocab_size)
                    logger.info(f"Resized main model token embeddings to {tokenizer_vocab_size}")
                else:
                    logger.info(f"Main model vocab size ({model.config.vocab_size}) already matches tokenizer vocab size ({tokenizer_vocab_size}). No resize needed.")
            elif current_vocab_size != tokenizer_vocab_size:
                 # Fallback if direct resizing isn't obvious
                 logger.warning(
                    f"Could not directly find 'resize_token_embeddings' on a base model or main model. "
                    f"Model vocab size is {current_vocab_size}, tokenizer has {tokenizer_vocab_size}. "
                    f"If special tokens were added, this might lead to errors or incorrect behavior. "
                    f"Ensure the model architecture supports dynamic resizing or that vocab sizes match."
                )
            else:
                logger.info(f"Model vocab size ({current_vocab_size}) matches tokenizer vocab size ({tokenizer_vocab_size}). No resize needed.")


    except Exception as e:
        logger.error(f"Failed to load/resize pre-trained QA model '{model_checkpoint}': {e}", exc_info=True)
        raise

    # Note: Compilation (optimizer, loss) is handled in the training script (train.py)
    logger.info(f"Successfully loaded QA model: {model_checkpoint}")
    model.summary(print_fn=logger.info) # Print summary using logger

    return model
