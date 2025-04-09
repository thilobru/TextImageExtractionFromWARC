# src/training/train.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .model import create_span_prediction_model
from .data_loader import prepare_training_data
from ..common.database import db_connect, get_training_contexts
from transformers import DistilBertTokenizer
import logging
import os
import wandb
# from wandb.keras import WandbMetricsLogger # Prefer WandbCallback
from wandb.integration.keras import WandbCallback

logger = logging.getLogger(__name__)

# --- Callbacks (Adapted from user code) ---

# Exact Match Callback
class ExactMatch(keras.callbacks.Callback):
    def __init__(self, validation_data, history_log):
        super().__init__()
        self.x_eval = validation_data[0] # Should be tuple (inputs, outputs) -> inputs = [ids, mask]
        self.y_eval_start = validation_data[1][0] # outputs = [start_ids, end_ids]
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

        logs = logs or {}
        logs['val_exact_match'] = acc
        self.history_log.setdefault('val_exact_match', []).append(acc)
        # W&B logging is handled by WandbCallback if configured
        # wandb.log({"val_exact_match": acc}, step=epoch)
        logger.info(f"Epoch {epoch+1}: Validation Exact Match = {acc:.4f}")


# IoU Callback
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

            # Ensure valid ranges
            if pred_s > pred_e or true_s > true_e:
                iou_list.append(0.0) # Invalid span prediction or label
                continue

            pred_range = set(range(pred_s, pred_e + 1))
            true_range = set(range(true_s, true_e + 1))

            intersection = len(pred_range.intersection(true_range))
            union = len(pred_range.union(true_range))

            iou = intersection / union if union > 0 else 0.0
            iou_list.append(iou)

        avg_iou = np.mean(iou_list) if iou_list else 0.0

        logs = logs or {}
        logs['val_iou'] = avg_iou
        self.history_log.setdefault('val_iou', []).append(avg_iou)
        # wandb.log({"val_iou": avg_iou}, step=epoch)
        logger.info(f"Epoch {epoch+1}: Validation IoU = {avg_iou:.4f}")

# --- Main Training Function ---

def train_model(db_path, model_save_dir, training_params, run_wandb=True):
    """
    Loads data, creates the model, and runs the training process.

    Args:
        db_path (str): Path to the SQLite database.
        model_save_dir (str): Directory to save the trained model.
        training_params (dict): Dictionary of training hyperparameters.
        run_wandb (bool): Whether to initialize and use wandb.
    """
    logger.info("Starting model training process...")

    # --- Setup ---
    if run_wandb:
        try:
             # Ensure you are logged in to wandb (`wandb login`)
             wandb.init(project="SpanPrediction", config=training_params)
             logger.info("Weights & Biases initialized.")
        except Exception as e:
             logger.error(f"Failed to initialize W&B: {e}. Continuing without W&B.")
             run_wandb = False # Disable if init fails
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
    logger.info("Loading training contexts from database...")
    try:
        with db_connect(db_path) as cursor:
            # Load all training candidates for now
            # Add limit/offset later if dataset is too large for memory
            training_contexts = get_training_contexts(cursor)
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        return

    if not training_contexts:
        logger.error("No training data found in the database. Ensure Stage 1 ran in 'training' mode.")
        return

    logger.info(f"Preparing {len(training_contexts)} training examples...")
    x_tokens, x_masks, y_starts, y_ends = prepare_training_data(
        training_contexts, tokenizer, training_params['max_len']
    )

    if x_tokens is None:
        logger.error("Failed to prepare training data.")
        return

    # --- Prepare Validation Split ---
    val_split = training_params.get('validation_split', 0.2)
    if val_split > 0:
         num_samples = len(x_tokens)
         indices = np.arange(num_samples)
         np.random.shuffle(indices) # Shuffle before splitting

         split_idx = int(num_samples * (1 - val_split))

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
        tokenizer_vocab_size=len(tokenizer) # Pass the actual vocab size
    )

    # --- Setup Callbacks ---
    callbacks = []
    history_log = {} # Simple dict to store history if not using built-in history

    # Model Checkpoint (save best model based on validation loss or IoU)
    os.makedirs(model_save_dir, exist_ok=True)
    # Use val_loss as default monitor, switch to val_iou if validation data exists
    monitor_metric = 'val_loss' if validation_data else 'loss'
    mode = 'min' if monitor_metric == 'val_loss' else 'max'

    # Monitor custom IoU metric if validation data is available
    if validation_data:
        iou_callback = IoUCallback(validation_data, history_log)
        exact_match_callback = ExactMatch(validation_data, history_log)
        callbacks.extend([iou_callback, exact_match_callback])
        monitor_metric = 'val_iou' # Change monitor if using IoU callback
        mode = 'max' # Maximize IoU

    model_checkpoint_path = os.path.join(model_save_dir, "model-epoch{epoch:02d}-" + monitor_metric + "{"+monitor_metric+":.4f}.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_weights_only=False, # Save entire model
        monitor=monitor_metric,
        mode=mode,
        save_best_only=True
    )
    callbacks.append(checkpoint_callback)

    # Early Stopping
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=3, # Stop after 3 epochs with no improvement
        mode=mode,
        restore_best_weights=True # Restore best weights found
    )
    callbacks.append(early_stopping_callback)

    # W&B Callback
    if run_wandb and validation_data:
         # Log gradients, weights, and validation data predictions
         callbacks.append(WandbCallback(
            # log_weights=True, # Can be very noisy
            # log_gradients=True, # Can be very noisy
            save_model=False, # Let Keras ModelCheckpoint handle saving
            validation_data=validation_data, # Log validation metrics
            # input_type='auto', # Try auto-detecting input/output types for logging
            # log_predictions=10 # Log predictions for 10 validation samples
         ))
         logger.info("Added WandbCallback.")
    elif run_wandb:
         callbacks.append(WandbCallback(save_model=False)) # Log metrics without validation data specifics


    # --- Train ---
    logger.info("Starting model fitting...")
    history = model.fit(
        x_train,
        y_train,
        epochs=training_params['epochs'],
        batch_size=training_params['batch_size'],
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1 # Or 2 for less output per epoch
    )
    logger.info("Model training finished.")

    # --- Save Final Model (optional, best is saved by checkpoint) ---
    # final_model_path = os.path.join(model_save_dir, "model-final.h5")
    # model.save(final_model_path)
    # logger.info(f"Final model saved to {final_model_path}")

    if run_wandb:
        wandb.finish()

    return history # Return history object