import tensorflow as tf
from tensorflow import keras
import numpy as np
from .modeldistilBERT import create_span_prediction_model
from .data_loader import prepare_training_data
from ..common.database import db_connect, get_training_contexts
from transformers import DistilBertTokenizer
import logging
import os
import wandb
from wandb.integration.keras import WandbCallback

logger = logging.getLogger(__name__)

# --- Callbacks (remain the same) ---
class ExactMatch(keras.callbacks.Callback):
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
        pred_start_logits, pred_end_logits = self.model.predict(self.x_eval)
        pred_start_idx = np.argmax(pred_start_logits, axis=-1)
        pred_end_idx = np.argmax(pred_end_logits, axis=-1)
        exact_matches = (pred_start_idx == self.y_eval_start) & (pred_end_idx == self.y_eval_end)
        acc = np.mean(exact_matches)
        if logs is not None: logs['val_exact_match'] = acc
        self.history_log.setdefault('val_exact_match', []).append(acc)
        logger.info(f"Epoch {epoch+1}: Validation Exact Match = {acc:.4f}")

class IoUCallback(keras.callbacks.Callback):
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
        pred_start_logits, pred_end_logits = self.model.predict(self.x_eval)
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
    Loads data, creates the model, and runs the training process.

    Args:
        db_path (str): Path to the SQLite database.
        model_save_dir (str): Directory to save the trained model.
        training_params (dict): Dictionary of training hyperparameters.
                                Should include 'max_examples' (int or None).
        run_wandb (bool): Whether to initialize and use wandb.
    """
    logger.info("Starting model training process...")

    # --- Setup ---
    if run_wandb:
        try:
             wandb.init(project="SpanPrediction", config=training_params)
             logger.info("Weights & Biases initialized.")
        except Exception as e:
             logger.error(f"Failed to initialize W&B: {e}. Continuing without W&B.")
             run_wandb = False
    else:
        logger.info("Skipping Weights & Biases initialization.")

    # Load Tokenizer
    try:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        special_token = "[IMG]"
        if special_token not in tokenizer.additional_special_tokens:
             tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
             logger.info(f"Added special token '{special_token}' to tokenizer.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # --- Load Data ---
    # --- Start Modification ---
    # Get max_examples from params, default to None (no limit)
    max_examples = training_params.get('max_examples', None)
    limit_log = f" (limit: {max_examples})" if max_examples is not None else ""
    logger.info(f"Loading training contexts from database...{limit_log}")
    # --- End Modification ---
    try:
        with db_connect(db_path) as cursor:
            # --- Start Modification ---
            # Pass max_examples as the limit to the DB function
            training_contexts = get_training_contexts(cursor, limit=max_examples)
            # --- End Modification ---
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        return

    if not training_contexts:
        logger.error("No training data found in the database matching criteria. Ensure Stage 1 ran in 'training' mode.")
        return

    logger.info(f"Preparing {len(training_contexts)} training examples...")
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(
        training_contexts, tokenizer, training_params['max_len']
    )

    # Check if prepare_training_data returned empty arrays (e.g., all examples rejected)
    if x_tokens is None or len(x_tokens) == 0:
        logger.error("Failed to prepare any valid training examples from the loaded contexts.")
        return

    # --- Prepare Validation Split ---
    val_split = training_params.get('validation_split', 0.2)
    if val_split > 0 and len(x_tokens) > 1: # Need at least 2 samples to split
         num_samples = len(x_tokens)
         indices = np.arange(num_samples)
         np.random.shuffle(indices)
         split_idx = int(num_samples * (1 - val_split))
         if split_idx == 0: # Ensure at least one training sample
             split_idx = 1
         if split_idx == num_samples: # Ensure at least one validation sample
              split_idx = num_samples - 1

         train_indices = indices[:split_idx]
         val_indices = indices[split_idx:]

         x_train = [x_tokens[train_indices], x_masks[train_indices]]
         y_train = [y_starts[train_indices], y_ends[train_indices]]
         x_val = [x_tokens[val_indices], x_masks[val_indices]]
         y_val = [y_starts[val_indices], y_ends[val_indices]]

         validation_data = (x_val, y_val)
         logger.info(f"Using {len(train_indices)} samples for training, {len(val_indices)} for validation.")
    else:
         x_train = [x_tokens, x_masks]
         y_train = [y_starts, y_ends]
         validation_data = None
         logger.info(f"Using {len(x_tokens)} samples for training, no validation split.")


    # --- Create Model ---
    model = create_span_prediction_model(
        max_len=training_params['max_len'],
        learning_rate=training_params['learning_rate'],
        tokenizer_vocab_size=len(tokenizer)
    )

    # --- Setup Callbacks ---
    callbacks = []
    history_log = {}
    monitor_metric = 'val_loss' if validation_data else 'loss'
    mode = 'min' if monitor_metric == 'val_loss' else 'max'

    if validation_data:
        iou_callback = IoUCallback(validation_data, history_log)
        exact_match_callback = ExactMatch(validation_data, history_log)
        callbacks.extend([iou_callback, exact_match_callback])
        monitor_metric = 'val_iou'
        mode = 'max'

    os.makedirs(model_save_dir, exist_ok=True)
    model_checkpoint_path = os.path.join(model_save_dir, "model-epoch{epoch:02d}-" + monitor_metric + "{"+monitor_metric+":.4f}.weights.h5") # Save weights only
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_weights_only=True, # Changed to True - more reliable for subclassed models
        monitor=monitor_metric,
        mode=mode,
        save_best_only=True
    )
    callbacks.append(checkpoint_callback)

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=training_params.get('early_stopping_patience', 3), # Allow configuring patience
        mode=mode,
        restore_best_weights=True
    )
    callbacks.append(early_stopping_callback)

    if run_wandb:
         wandb_callback_params = { "save_model": False } # Don't let W&B save model
         if validation_data:
             wandb_callback_params["validation_data"] = validation_data
             # wandb_callback_params["log_predictions"] = 10 # Optional: Log some predictions
         callbacks.append(WandbCallback(**wandb_callback_params))
         logger.info("Added WandbCallback.")


    # --- Train ---
    logger.info("Starting model fitting...")
    history = model.fit(
        x_train,
        y_train,
        epochs=training_params['epochs'],
        batch_size=training_params['batch_size'],
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    logger.info("Model training finished.")

    # Save final weights (optional, best are saved by checkpoint)
    final_weights_path = os.path.join(model_save_dir, "model-final.weights.h5")
    model.save_weights(final_weights_path)
    logger.info(f"Final model weights saved to {final_weights_path}")


    if run_wandb:
        wandb.finish()

    return history
