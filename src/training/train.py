# src/training/train.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .model import create_span_prediction_model
from .data_loader import prepare_training_data
from ..common.database import db_connect, get_training_contexts
from transformers import DistilBertTokenizer, TFPreTrainedModel # Import base class
import logging
import os
import wandb
from wandb.integration.keras import WandbCallback

logger = logging.getLogger(__name__)

# --- Callbacks (ExactMatch, IoUCallback remain the same) ---
class ExactMatch(keras.callbacks.Callback):
    # ... (Keep implementation as before) ...
    def __init__(self, validation_data, history_log):
        super().__init__()
        self.x_eval = validation_data[0]
        self.y_eval_start = validation_data[1][0]
        self.y_eval_end = validation_data[1][1]
        self.history_log = history_log

    def on_epoch_end(self, epoch, logs=None):
        if not self.x_eval or self.y_eval_start is None or self.y_eval_end is None:
             logger.warning("ExactMatch callback: No validation data provided.")
             return
        # --- Change: Handle model output ---
        # model.predict might return the QA output object
        predictions = self.model.predict(self.x_eval)
        pred_start_logits = predictions.start_logits # Access logits from output object
        pred_end_logits = predictions.end_logits   # Access logits from output object
        # --- End Change ---
        pred_start_idx = np.argmax(pred_start_logits, axis=-1)
        pred_end_idx = np.argmax(pred_end_logits, axis=-1)
        exact_matches = (pred_start_idx == self.y_eval_start) & (pred_end_idx == self.y_eval_end)
        acc = np.mean(exact_matches)
        if logs is not None: logs['val_exact_match'] = acc
        self.history_log.setdefault('val_exact_match', []).append(acc)
        logger.info(f"Epoch {epoch+1}: Validation Exact Match = {acc:.4f}")

class IoUCallback(keras.callbacks.Callback):
    # ... (Keep implementation as before) ...
    def __init__(self, validation_data, history_log):
        super().__init__()
        self.x_eval = validation_data[0]
        self.y_eval_start = validation_data[1][0]
        self.y_eval_end = validation_data[1][1]
        self.history_log = history_log

    def on_epoch_end(self, epoch, logs=None):
        if not self.x_eval or self.y_eval_start is None or self.y_eval_end is None:
             logger.warning("IoU callback: No validation data provided.")
             return
        # --- Change: Handle model output ---
        predictions = self.model.predict(self.x_eval)
        pred_start_logits = predictions.start_logits
        pred_end_logits = predictions.end_logits
        # --- End Change ---
        pred_start_idx = np.argmax(pred_start_logits, axis=-1)
        pred_end_idx = np.argmax(pred_end_logits, axis=-1)
        iou_list = []
        for i in range(len(pred_start_idx)):
            pred_s, pred_e = pred_start_idx[i], pred_end_idx[i]
            true_s, true_e = self.y_eval_start[i], self.y_eval_end[i]
            if pred_s > pred_e or true_s > true_e: iou_list.append(0.0); continue
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
             wandb.init(project="SpanPredictionQA", config=training_params) # New project name?
             logger.info("Weights & Biases initialized.")
        except Exception as e:
             logger.error(f"Failed to initialize W&B: {e}. Continuing without W&B.")
             run_wandb = False
    else:
        logger.info("Skipping Weights & Biases initialization.")

    # Load Tokenizer
    try:
        # Use the same checkpoint name as the intended model for consistency
        model_checkpoint = training_params.get('model_checkpoint', "distilbert-base-uncased")
        tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)
        special_token = "[IMG]"
        if special_token not in tokenizer.additional_special_tokens:
             num_added = tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
             if num_added > 0: logger.info(f"Added special token '{special_token}' to tokenizer.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # --- Load Data (remains the same, uses limit) ---
    max_examples = training_params.get('max_examples', None)
    limit_log = f" (limit: {max_examples})" if max_examples is not None else ""
    logger.info(f"Loading training contexts from database...{limit_log}")
    try:
        with db_connect(db_path) as cursor:
            training_contexts = get_training_contexts(cursor, limit=max_examples)
    except Exception as e: logger.error(f"Failed to load data: {e}"); return
    if not training_contexts: logger.error("No training data found."); return

    logger.info(f"Preparing {len(training_contexts)} training examples...")
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(
        training_contexts, tokenizer, training_params['max_len']
    )
    if x_tokens is None or len(x_tokens) == 0: logger.error("Failed to prepare training examples."); return

    # Combine start/end labels for Keras fit with QA models
    # Keras expects y_true to match model output structure. HF QA models output logits.
    # The default loss calculation inside the HF model handles start/end positions.
    # We need to provide labels in a format Keras understands or use TFTrainer.
    # Let's try passing labels as a dictionary matching the expected input names for the loss calculation
    # (check TFDistilBertForQuestionAnswering documentation/source)
    # It expects 'start_positions' and 'end_positions'.
    y_train_dict = {'start_positions': y_starts, 'end_positions': y_ends}


    # --- Prepare Validation Split ---
    val_split = training_params.get('validation_split', 0.2)
    validation_data_keras = None # For Keras fit
    validation_data_cb = None # For custom callbacks

    if val_split > 0 and len(x_tokens) > 1:
         num_samples = len(x_tokens)
         indices = np.arange(num_samples)
         np.random.shuffle(indices)
         split_idx = int(num_samples * (1 - val_split)); split_idx = max(1, split_idx); split_idx = min(num_samples - 1, split_idx)
         train_indices = indices[:split_idx]; val_indices = indices[split_idx:]

         x_train_fit = {'input_ids': x_tokens[train_indices], 'attention_mask': x_masks[train_indices]}
         y_train_fit = {'start_positions': y_starts[train_indices], 'end_positions': y_ends[train_indices]}

         x_val_fit = {'input_ids': x_tokens[val_indices], 'attention_mask': x_masks[val_indices]}
         y_val_fit = {'start_positions': y_starts[val_indices], 'end_positions': y_ends[val_indices]}

         validation_data_keras = (x_val_fit, y_val_fit) # Data for Keras validation steps

         # Data for our custom callbacks (still need original format)
         validation_data_cb = ([x_tokens[val_indices], x_masks[val_indices]], # Inputs as list
                               [y_starts[val_indices], y_ends[val_indices]]) # Outputs as list

         logger.info(f"Using {len(train_indices)} samples for training, {len(val_indices)} for validation.")
    else:
         x_train_fit = {'input_ids': x_tokens, 'attention_mask': x_masks}
         y_train_fit = y_train_dict # Use the full dataset dict
         validation_data_keras = None
         validation_data_cb = None
         logger.info(f"Using {len(x_tokens)} samples for training, no validation split.")


    # --- Create & Compile Model ---
    model = create_span_prediction_model(
        max_len=training_params['max_len'],
        learning_rate=training_params['learning_rate'], # Not used by create_... now
        tokenizer_vocab_size=len(tokenizer),
        model_checkpoint=model_checkpoint # Pass checkpoint name
    )

    # Compile the Hugging Face model for use with Keras fit
    # The HF model computes the loss internally if labels are provided
    try:
        # Use legacy optimizer if needed (based on previous findings)
        from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
        optimizer = LegacyAdam(learning_rate=training_params['learning_rate'])
        logger.info("Using Legacy Adam optimizer.")
    except ImportError:
        logger.warning("Legacy Adam not found, using default Adam. Might cause issues.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=training_params['learning_rate'])

    # Compile *without* specifying loss, as the HF model calculates it internally
    # when labels ('start_positions', 'end_positions') are passed in y_true.
    model.compile(optimizer=optimizer)
    logger.info("Compiled QA model.")


    # --- Setup Callbacks ---
    callbacks = []
    history_log = {} # Use Keras history object instead?
    monitor_metric = 'val_loss' if validation_data_keras else 'loss' # Monitor loss calculated by the model
    mode = 'min' # Minimize loss

    # Custom callbacks need the specific validation_data_cb format
    if validation_data_cb:
        iou_callback = IoUCallback(validation_data_cb, history_log)
        exact_match_callback = ExactMatch(validation_data_cb, history_log)
        callbacks.extend([iou_callback, exact_match_callback])
        # monitor_metric = 'val_iou' # Cannot monitor custom metric directly for EarlyStopping/Checkpoint with internal loss
        # mode = 'max'

    # --- Change: Remove ModelCheckpoint ---
    # Checkpointing handled by saving the best model after training based on history
    # os.makedirs(model_save_dir, exist_ok=True)
    # model_checkpoint_path = ...
    # checkpoint_callback = keras.callbacks.ModelCheckpoint(...)
    # callbacks.append(checkpoint_callback)
    logger.info("Keras ModelCheckpoint callback removed. Saving best model after training.")

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=monitor_metric, # Monitor val_loss or loss
        patience=training_params.get('early_stopping_patience', 3),
        mode=mode,
        restore_best_weights=True # Restore best weights based on monitored metric
    )
    callbacks.append(early_stopping_callback)

    if run_wandb:
         wandb_callback_params = { "save_model": False }
         # Pass Keras-compatible validation data to WandbCallback if available
         if validation_data_keras:
             wandb_callback_params["validation_data"] = validation_data_keras
         callbacks.append(WandbCallback(**wandb_callback_params))
         logger.info("Added WandbCallback.")


    # --- Train ---
    logger.info("Starting model fitting...")
    history = model.fit(
        x_train_fit, # Use dictionary input
        y_train_fit, # Use dictionary labels
        epochs=training_params['epochs'],
        batch_size=training_params['batch_size'],
        validation_data=validation_data_keras, # Use Keras validation data format
        callbacks=callbacks,
        verbose=1
    )
    logger.info("Model training finished.")

    # --- Change: Save final model using save_pretrained ---
    # The model state restored by EarlyStopping contains the best weights.
    final_model_save_path = os.path.join(model_save_dir, "distilbert-qa-final") # Save to a directory
    try:
        model.save_pretrained(final_model_save_path)
        tokenizer.save_pretrained(final_model_save_path) # Save tokenizer too
        logger.info(f"Final model saved to {final_model_save_path} using save_pretrained.")
    except Exception as e:
        logger.error(f"Failed to save final model using save_pretrained: {e}")


    if run_wandb:
        wandb.finish()

    return history
