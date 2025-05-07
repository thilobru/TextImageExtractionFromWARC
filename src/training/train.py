# src/training/train.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .model import create_span_prediction_model
from .data_loader import prepare_training_data
from ..common.database import db_connect, get_training_contexts
from transformers import AutoTokenizer, TFPreTrainedModel # Import AutoTokenizer
import logging
import os
import wandb
from wandb.integration.keras import WandbCallback

logger = logging.getLogger(__name__)

# --- Callbacks (ExactMatch, IoUCallback) ---
class ExactMatch(keras.callbacks.Callback):
    def __init__(self, validation_data, history_log):
        super().__init__()
        # validation_data[0] is expected to be a dictionary of inputs for HF models
        # e.g., {'input_ids': ..., 'attention_mask': ...}
        self.x_eval_dict = validation_data[0]
        self.y_eval_start = validation_data[1][0] # True start indices
        self.y_eval_end = validation_data[1][1]   # True end indices
        self.history_log = history_log

    def on_epoch_end(self, epoch, logs=None):
        if not self.x_eval_dict or self.y_eval_start is None or self.y_eval_end is None:
             logger.warning("ExactMatch callback: No validation data provided or data is incomplete.")
             if logs is not None: logs['val_exact_match'] = 0.0 # Default to 0 if no data
             self.history_log.setdefault('val_exact_match', []).append(0.0)
             return

        # model.predict expects a dictionary for HF models if x_eval_dict is a dict
        predictions = self.model.predict(self.x_eval_dict)
        pred_start_logits = predictions.start_logits
        pred_end_logits = predictions.end_logits
        
        pred_start_idx = np.argmax(pred_start_logits, axis=-1)
        pred_end_idx = np.argmax(pred_end_logits, axis=-1)
        
        exact_matches = (pred_start_idx == self.y_eval_start) & (pred_end_idx == self.y_eval_end)
        acc = np.mean(exact_matches) if exact_matches.size > 0 else 0.0 # Handle empty case
        
        if logs is not None: logs['val_exact_match'] = acc
        self.history_log.setdefault('val_exact_match', []).append(acc)
        logger.info(f"Epoch {epoch+1}: Validation Exact Match = {acc:.4f}")

class IoUCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, history_log):
        super().__init__()
        self.x_eval_dict = validation_data[0] # Expects dict {'input_ids': ..., 'attention_mask': ...}
        self.y_eval_start = validation_data[1][0]
        self.y_eval_end = validation_data[1][1]
        self.history_log = history_log

    def on_epoch_end(self, epoch, logs=None):
        if not self.x_eval_dict or self.y_eval_start is None or self.y_eval_end is None:
             logger.warning("IoU callback: No validation data provided or data is incomplete.")
             if logs is not None: logs['val_iou'] = 0.0
             self.history_log.setdefault('val_iou', []).append(0.0)
             return

        predictions = self.model.predict(self.x_eval_dict)
        pred_start_logits = predictions.start_logits
        pred_end_logits = predictions.end_logits
        
        pred_start_idx = np.argmax(pred_start_logits, axis=-1)
        pred_end_idx = np.argmax(pred_end_logits, axis=-1)
        
        iou_list = []
        for i in range(len(pred_start_idx)):
            pred_s, pred_e = pred_start_idx[i], pred_end_idx[i]
            true_s, true_e = self.y_eval_start[i], self.y_eval_end[i]
            
            # Ensure valid spans (start <= end)
            if pred_s > pred_e or true_s > true_e: 
                iou_list.append(0.0)
                continue
            
            pred_range = set(range(pred_s, pred_e + 1))
            true_range = set(range(true_s, true_e + 1))
            
            intersection = len(pred_range.intersection(true_range))
            union = len(pred_range.union(true_range))
            
            iou = intersection / union if union > 0 else 0.0
            iou_list.append(iou)
            
        avg_iou = np.mean(iou_list) if iou_list else 0.0
        
        if logs is not None: logs['val_iou'] = avg_iou
        self.history_log.setdefault('val_iou', []).append(avg_iou)
        logger.info(f"Epoch {epoch+1}: Validation IoU = {avg_iou:.4f}")


# --- Main Training Function (Updated) ---
def train_model(db_path, model_save_dir, training_params, run_wandb=True):
    """
    Loads data, loads pre-trained QA model, compiles, and runs training.
    Saves final model using model.save_pretrained().
    """
    logger.info("Starting model training process...")

    # --- Setup ---
    if run_wandb:
        try:
             wandb.init(project="TextImageExtraction-QA", config=training_params) # New project name
             logger.info("Weights & Biases initialized.")
        except Exception as e:
             logger.error(f"Failed to initialize W&B: {e}. Continuing without W&B.")
             run_wandb = False
    else:
        logger.info("Skipping Weights & Biases initialization.")

    # Get model_checkpoint from training_params
    model_checkpoint = training_params.get('model_checkpoint', "distilbert-base-uncased")
    logger.info(f"Using model checkpoint: {model_checkpoint}")

    # Load Tokenizer using AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        special_token = "[IMG]"
        if special_token not in tokenizer.additional_special_tokens:
             num_added = tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
             if num_added > 0: 
                 logger.info(f"Added special token '{special_token}' to tokenizer. New vocab size: {len(tokenizer)}")
             else:
                 logger.info(f"Special token '{special_token}' already in tokenizer or not added.")
        else:
            logger.info(f"Special token '{special_token}' already part of the tokenizer's vocabulary.")
            
    except Exception as e:
        logger.error(f"Failed to load tokenizer for '{model_checkpoint}': {e}", exc_info=True)
        return

    # --- Load Data ---
    max_examples = training_params.get('max_examples', None)
    limit_log = f" (limit: {max_examples})" if max_examples is not None else ""
    logger.info(f"Loading training contexts from database...{limit_log}")
    try:
        with db_connect(db_path) as cursor:
            training_contexts = get_training_contexts(cursor, limit=max_examples)
    except Exception as e: 
        logger.error(f"Failed to load data from database: {e}", exc_info=True)
        return
    
    if not training_contexts: 
        logger.error("No training data found in the database matching criteria.")
        return

    logger.info(f"Preparing {len(training_contexts)} training examples...")
    # Pass the loaded tokenizer to prepare_training_data
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(
        training_contexts, tokenizer, training_params['max_len']
    )
    if x_tokens is None or len(x_tokens) == 0: 
        logger.error("Failed to prepare any valid training examples.")
        return

    # --- Prepare Validation Split ---
    val_split = training_params.get('validation_split', 0.2)
    validation_data_keras = None # For Keras fit method (model's internal validation)
    validation_data_callbacks = None # For custom callbacks (ExactMatch, IoU)

    if val_split > 0 and len(x_tokens) > 1:
         num_samples = len(x_tokens)
         indices = np.arange(num_samples)
         np.random.shuffle(indices) # Shuffle before splitting
         
         split_idx = int(num_samples * (1 - val_split))
         # Ensure at least one sample in each split if possible
         split_idx = max(1, split_idx) 
         split_idx = min(num_samples - 1, split_idx) if num_samples > 1 else 1

         train_indices = indices[:split_idx]
         val_indices = indices[split_idx:]

         # Data for model.fit (inputs and labels as dictionaries)
         x_train_fit = {'input_ids': x_tokens[train_indices], 'attention_mask': x_masks[train_indices]}
         y_train_fit = {'start_positions': y_starts[train_indices], 'end_positions': y_ends[train_indices]}

         x_val_fit = {'input_ids': x_tokens[val_indices], 'attention_mask': x_masks[val_indices]}
         y_val_fit = {'start_positions': y_starts[val_indices], 'end_positions': y_ends[val_indices]}
         validation_data_keras = (x_val_fit, y_val_fit)

         # Data for custom callbacks (inputs as dict, labels as list of arrays for easier access)
         validation_data_callbacks = (
             {'input_ids': x_tokens[val_indices], 'attention_mask': x_masks[val_indices]}, # x_eval_dict
             [y_starts[val_indices], y_ends[val_indices]] # y_eval_start, y_eval_end
         )
         logger.info(f"Using {len(train_indices)} samples for training, {len(val_indices)} for validation.")
    else:
         x_train_fit = {'input_ids': x_tokens, 'attention_mask': x_masks}
         y_train_fit = {'start_positions': y_starts, 'end_positions': y_ends}
         validation_data_keras = None
         validation_data_callbacks = None
         logger.info(f"Using {len(x_tokens)} samples for training, no validation split.")

    # --- Create & Compile Model ---
    # Pass tokenizer_vocab_size for potential embedding resize
    model = create_span_prediction_model(
        model_checkpoint=model_checkpoint,
        tokenizer_vocab_size=len(tokenizer) 
    )

    # Compile the Hugging Face model for use with Keras fit
    try:
        # Using legacy optimizer can sometimes avoid issues with HF models in TF
        from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
        optimizer = LegacyAdam(learning_rate=training_params['learning_rate'])
        logger.info(f"Using Legacy Adam optimizer with learning rate: {training_params['learning_rate']}.")
    except ImportError:
        logger.warning("Legacy Adam not found, using default tf.keras.optimizers.Adam. This might cause issues with some Hugging Face models.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=training_params['learning_rate'])

    # Compile *without* specifying loss, as the HF model calculates it internally
    # when labels ('start_positions', 'end_positions') are passed in y_true during fit.
    # We can add metrics here if Keras supports them directly with this setup, or rely on callbacks.
    model.compile(optimizer=optimizer) 
    logger.info(f"Compiled QA model '{model_checkpoint}'.")

    # --- Setup Callbacks ---
    callbacks = []
    history_log_for_custom_cb = {} # Separate log for custom callbacks if needed, Keras history is primary
    
    # Determine monitor metric for EarlyStopping and ModelCheckpoint
    # If validation data is present, monitor 'val_loss' (calculated by the model).
    # Otherwise, monitor 'loss'.
    monitor_metric = 'val_loss' if validation_data_keras else 'loss'
    mode = 'min' # We want to minimize loss

    if validation_data_callbacks: # If data for custom callbacks exists
        iou_callback = IoUCallback(validation_data_callbacks, history_log_for_custom_cb)
        exact_match_callback = ExactMatch(validation_data_callbacks, history_log_for_custom_cb)
        callbacks.extend([iou_callback, exact_match_callback])
        # If you want to use IoU for ModelCheckpoint/EarlyStopping, you'd need to ensure
        # it's part of `logs` dict in on_epoch_end and Keras picks it up.
        # For now, sticking to 'val_loss' or 'loss' is safer with HF internal loss.
        # monitor_metric = 'val_iou' # This would require IoU to be a Keras metric or logged correctly
        # mode = 'max'

    # ModelCheckpoint: Save the best model based on 'monitor_metric'
    # Note: HF models are best saved using save_pretrained, so we save the entire model directory.
    # Keras ModelCheckpoint with save_weights_only=False might not always work perfectly for complex HF models.
    # Instead, we'll use EarlyStopping with restore_best_weights=True and then save the model once after training.
    
    early_stopping_patience = training_params.get('early_stopping_patience', 3)
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=early_stopping_patience,
        mode=mode,
        restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
    )
    callbacks.append(early_stopping_callback)
    logger.info(f"Added EarlyStopping callback monitoring '{monitor_metric}' (mode: {mode}, patience: {early_stopping_patience}).")

    if run_wandb:
         wandb_callback_params = { "save_model": False } # We handle saving manually
         # Pass Keras-compatible validation data to WandbCallback if available
         if validation_data_keras: # This is the (x_val_fit, y_val_fit) tuple
             wandb_callback_params["validation_data"] = validation_data_keras
             # W&B callback can log gradients, weights, etc.
             # wandb_callback_params["log_gradients"] = True # Example
         callbacks.append(WandbCallback(**wandb_callback_params))
         logger.info("Added WandbCallback.")

    # --- Train ---
    logger.info("Starting model fitting...")
    history = model.fit(
        x_train_fit, # Dictionary input: {'input_ids': ..., 'attention_mask': ...}
        y_train_fit, # Dictionary labels: {'start_positions': ..., 'end_positions': ...}
        epochs=training_params['epochs'],
        batch_size=training_params['batch_size'],
        validation_data=validation_data_keras, # Keras validation data format
        callbacks=callbacks,
        verbose=1 # Or 2 for more detailed per-batch logging
    )
    logger.info("Model training finished.")

    # --- Save final model using save_pretrained ---
    # The model state restored by EarlyStopping (if restore_best_weights=True) contains the best weights.
    # So, we save this final state.
    os.makedirs(model_save_dir, exist_ok=True) # Ensure save directory exists
    final_model_save_path = os.path.join(model_save_dir, f"{model_checkpoint.replace('/', '_')}-final_qa_model")
    try:
        model.save_pretrained(final_model_save_path)
        tokenizer.save_pretrained(final_model_save_path) # Save tokenizer alongside the model
        logger.info(f"Final model and tokenizer saved to {final_model_save_path} using save_pretrained.")
        
        # Optionally, save training history and custom callback history
        if history:
            history_save_path = os.path.join(final_model_save_path, "training_history.json")
            import json
            with open(history_save_path, 'w') as f:
                json.dump(history.history, f, indent=4)
            logger.info(f"Keras training history saved to {history_save_path}")
        if history_log_for_custom_cb:
            custom_history_save_path = os.path.join(final_model_save_path, "custom_callback_history.json")
            with open(custom_history_save_path, 'w') as f:
                json.dump(history_log_for_custom_cb, f, indent=4)
            logger.info(f"Custom callback history saved to {custom_history_save_path}")

    except Exception as e:
        logger.error(f"Failed to save final model using save_pretrained: {e}", exc_info=True)

    if run_wandb:
        wandb.finish()

    return history
