## Multimodal Dataset Creation from WARC Files

**1. Project Goal**

The primary objective of this project is to automatically extract high-quality image-description pairs from large web archive (WARC) files, specifically from the Common Crawl dataset. The extracted pairs are intended for downstream tasks, potentially including the fine-tuning of multimodal models like text-to-image generators (e.g., Stable Diffusion). The pipeline aims to identify images and find relevant descriptive text associated with them in the surrounding HTML context.

**2. Architecture & Technology Stack**

The project follows a multi-stage pipeline architecture implemented primarily in Python. Key technologies include:

* **WARC Processing:** `fastwarc` for efficiently reading WARC files, `resiliparse` for robust HTML parsing and text extraction.
* **Machine Learning:** `tensorflow` (with Keras) for training and running a neural network model, `transformers` for leveraging pre-trained models (DistilBERT for span prediction, CLIP for image-text similarity).
* **Data Storage:** `sqlite3` for storing intermediate and final results in a structured relational database.
* **Image Handling:** `requests` for downloading images, `Pillow` (PIL) for basic image validation.
* **Workflow & Utilities:** Standard Python libraries (`os`, `re`, `json`, `logging`, `argparse`), `tqdm` for progress bars, `pytest` for testing, potentially `wandb` for experiment tracking.
* **Project Structure:** A modular structure separating source code (`src/`) by functionality (common, data_extraction, training, inference, enrichment, analysis) from executable scripts (`scripts/`), data (`data/`), saved models (`models/`), and tests (`tests/`).

**3. Pipeline Stages**

The process is broken down into distinct, sequential stages, managed via scripts in the `scripts/` directory:

**`Stage 1/3a: Context Extraction from WARC (scripts/01_extract_context_from_warc.py)`**

* **Purpose:** Process raw WARC files to identify images and extract their associated context (surrounding text, alt text). Prepare data for both training the span predictor (Stage 2) and running inference (Stage 3b).
* **Input:** WARC files (`.warc`, `.warc.gz`).
* **Processing:**
    * Iterates through WARC records using `fastwarc.ArchiveIterator`.
    * Filters for valid HTML responses (`text/html`).
    * Parses HTML using `resiliparse.HTMLTree` (`src/data_extraction/html_parser.py`).
    * Identifies `<img>` tags, extracts `src` and `alt` attributes.
    * Uses a placeholder technique (`###IMG{i}###`) to replace `alt` attributes before calling `resiliparse.extract_plain_text(..., alt_texts=True)`.
    * Reconstructs `context_before` and `context_after` text relative to the placeholder's position in the extracted text stream.
    * Applies filters (e.g., `min_context_words`).
    * Performs a check (`found_alt_in_context`) by normalizing the `original_alt` and the `combined_context` and seeing if the alt text is present within the extracted text window (acting as a sanity check).
    * Uses `src/common/database.py` to save extracted information.
* **Output:** Populates the `ImageContext` table in the SQLite database (`image_data.db`) with columns like `image_url`, `page_url`, `warc_path`, `context_before`, `context_after`, `alt_text`.
* **Modes:**
    * `--mode training`: Additionally updates the `is_training_candidate` flag in `ImageContext` to `True` for entries where `found_alt_in_context` is true and the alt text meets length criteria.
    * `--mode inference`: Extracts all contexts without specifically marking training candidates.

**`Stage 2: Train Span Prediction Model (scripts/02_train_span_model.py)`**

* **Purpose:** Train a DistilBERT-based model to predict the start and end token indices of the image description (using `alt_text` as the ground truth label during training) within the surrounding context.
* **Input:** Training data fetched from the `ImageContext` table (where `is_training_candidate` is `True`) via `src/common/database.py`.
* **Processing:**
    * Loads data using `src/training/data_loader.py`, which prepares tokenized input sequences (`[CLS] context_before [IMG] context_after [SEP]`) and identifies the target start/end token indices corresponding to the `alt_text`.
    * Defines the model architecture using `src/training/model.py` (DistilBERT base + dense layers for start/end logits).
    * Uses `src/training/train.py` to manage the training loop (`model.fit`).
    * Includes callbacks for evaluation (`ExactMatch`, `IoUCallback`), model checkpointing (saving the best model based on validation IoU/loss), early stopping, and optional `wandb` logging.
* **Output:** A trained Keras model saved to the `models/span_predictor/` directory (e.g., `model-best.h5` or a SavedModel directory).

**`Stage 3b: Predict Description Spans (scripts/03_predict_descriptions.py)`**

* **Purpose:** Use the trained span prediction model to identify potential description spans for *all* relevant image contexts extracted in Stage 1/3a (not just the training candidates). This step focuses only on text prediction.
* **Input:**
    * The trained model saved from Stage 2.
    * Image contexts fetched from the `ImageContext` table via `src/common/database.py`.
* **Processing:**
    * Loads the trained model and tokenizer (`src/inference/predict_spans.py`).
    * Iterates through contexts fetched from the database in batches.
    * Prepares model input sequences (`[CLS] context_before [IMG] context_after [SEP]`, *without* inserting the original `alt_tag`).
    * Runs `model.predict` to get start/end logits.
    * Determines the most likely start/end token indices.
    * Calculates confidence scores based on probabilities derived from logits.
    * Decodes the predicted token span back into a text string (`predicted_text`).
    * Uses `src/common/database.py` to save the results.
* **Output:** Populates the `PredictedDescriptions` table in the database with `context_id`, `model_path`, `predicted_text`, token indices, confidence scores, and sets `status_enrichment` to 'pending'.

**`Stage 4: Enrich with CLIP Scores (scripts/04_enrich_clip_scores.py)`**

* **Purpose:** Download the images corresponding to the predicted descriptions and calculate the CLIP score (image-text similarity) for each pair.
* **Input:** Predictions from the `PredictedDescriptions` table where `status_enrichment` is 'pending', fetched via `src/common/database.py`. Requires image URLs from the linked `ImageContext` table.
* **Processing:**
    * Iterates through pending predictions.
    * Downloads the image using `requests` and validates its size/resolution using `Pillow` (`src/enrichment/image_downloader.py`). Handles download errors.
    * If download is successful, calculates the CLIP score between the downloaded image and the `predicted_text` using `transformers` CLIPModel and CLIPProcessor (`src/enrichment/clip_scorer.py`). Handles CLIP calculation errors.
    * Uses `src/common/database.py` to save the results.
* **Output:** Populates the `ClipResults` table with `prediction_id`, `image_local_path`, `clip_score`, `clip_model_name`, and any `error_message`. Updates the `status_enrichment` field in the corresponding `PredictedDescriptions` row (e.g., to 'clip_scored', 'failed_download', 'failed_clip').

**`Stage 5: Analyze Results (scripts/05_analyze_results.py)`**

* **Purpose:** Analyze the final enriched dataset (predictions with CLIP scores) to understand model performance, correlations between confidence and CLIP scores, and distribution of scores.
* **Input:** Data joined from `PredictedDescriptions`, `ImageContext`, and `ClipResults` tables, fetched via `src/common/database.py`. Can filter by CLIP score or other criteria.
* **Processing:**
    * Loads the final data (potentially into a pandas DataFrame).
    * Generates descriptive statistics (using functions potentially defined in `src/analysis/report.py` or directly in the script/notebook).
    * Creates visualizations (box plots, histograms, scatter plots, heatmaps) using `matplotlib` and `seaborn`.
* **Output:** Saved plots and statistical summary files (e.g., in `./analysis_output/`).

**4. Data Management (SQLite)**

* A single SQLite database (`data/database/image_data.db`) acts as the central hub.
* **`ImageContext`**: Stores raw extracted info linking images to their source page and surrounding text. `is_training_candidate` flags data for Stage 2.
* **`PredictedDescriptions`**: Stores the output of the text model (Stage 3b), including the predicted text and confidence scores. `status_enrichment` tracks Stage 4 processing.
* **`ClipResults`**: Stores the outcome of Stage 4, primarily the CLIP score and path to the downloaded image (if successful). Linked one-to-one with `PredictedDescriptions`.
* **Schema & Indices:** Defined in `src/common/database.py`, includes indices for efficient querying. Uses `ON CONFLICT DO NOTHING` for `ImageContext` to handle reruns. Uses WAL mode for better read concurrency.

**5. Configuration**

* Default paths and parameters are defined in `src/common/config.py`.
* Wrapper scripts (`scripts/`) use `argparse` to allow overriding defaults via command-line arguments (e.g., `--db-path`, `--model-path`, `--mode`).
* Using a `config.yaml` file is a potential alternative for managing configuration.

**6. Testing**

* Unit tests using `pytest` are placed in the `tests/` directory.
* Current focus is on testing `src/data_extraction/html_parser.py`.
* Further tests for database interactions, data loading, and other modules are recommended.

**7. Running the Pipeline**

* Each stage is executed via its corresponding script in the `scripts/` directory (e.g., `python scripts/01_...py <args>`).
* Stages are typically run sequentially, as the output of one stage (stored in the database) is the input for the next.

**8. Current Status & Next Steps**

* The project structure is defined.
* Core code modules for Stages 1, 2, 3b, and 4 are implemented.
* Stage 1 (`01_extract_context_from_warc.py`) has been run successfully on a sample file after fixing initial bugs.
* Unit tests for `html_parser.py` have been added and mostly pass, confirming the "Intention A" logic is working.
* **Next Steps:**
    * Run Stage 1 on all desired WARC files (in `training` mode).
    * Run Stage 2 (`02_train_span_model.py`) to train the model.
    * Run Stage 3b (`03_predict_descriptions.py`) using the trained model.
    * Run Stage 4 (`04_enrich_clip_scores.py`) to download images and get CLIP scores.
    * Run Stage 5 (`05_analyze_results.py`) or use notebooks to analyze the final dataset.
    * Add more tests for other modules (`database`, `data_loader`, etc.).
    * Refine error handling and logging.
    * Consider performance and potential parallelization improvements if needed (e.g., for Stage 4 downloads/CLIP scoring, possibly using `multiprocessing` or `asyncio` within the script, or exploring Dask/Ray/Spark if scaling significantly).

This document provides a snapshot of the project's design and current state based on our interactions.
