# Multimodal Dataset Creation from WARC Files

**1. Description**

This project implements a pipeline to automatically extract image-description pairs from large web archive (WARC) files (e.g., from Common Crawl). It identifies images within HTML pages, extracts surrounding text context, uses a trained transformer model (DistilBERT) to predict descriptive text spans, downloads images, and scores the image-text pair similarity using CLIP. Results are stored in an SQLite database.

**2. Features**

* Processes standard WARC files.
* Extracts image URLs, alt text, and surrounding text context.
* Trains a span prediction model to identify potential descriptions.
* Predicts descriptions for images using the trained model.
* Downloads images and calculates CLIP similarity scores.
* Stores all intermediate and final data in an SQLite database.
* Modular pipeline structure with separate scripts for each stage.
* Includes basic testing infrastructure using `pytest`.

**3. Technology Stack**

* Python 3.x
* TensorFlow / Keras
* Transformers (Hugging Face)
* fastwarc
* resiliparse
* SQLite (via `sqlite3`)
* Requests
* Pillow (PIL)
* NumPy
* pytest (for testing)
* tqdm (for progress bars)
* (Optional) wandb (for experiment tracking)

**4. Project Structure**

```
image-description-extractor/
│
├── data/             # SQLite DB, WARC file links/storage
├── models/           # Saved trained models
├── notebooks/        # Jupyter notebooks for analysis/experimentation
├── src/              # Core Python source code modules
│   ├── common/
│   ├── data_extraction/
│   ├── training/
│   ├── inference/
│   ├── enrichment/
│   └── analysis/
├── scripts/          # Executable scripts to run pipeline stages
├── tests/            # Unit and integration tests
│
├── Dockerfile        # (Optional) For containerization
├── requirements.txt  # Python dependencies
├── config.yaml       # (Optional) Configuration file
├── README.md         # This file
└── .gitignore        # Git ignore rules
```

**5. Setup & Installation**

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd image-description-extractor
    ```
2.  **Create Virtual Environment:** (Recommended)
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Obtain WARC Files:** Download or link WARC files into the `data/warc/` directory (or configure the path).
5.  **Initialize Database:** The database schema is typically initialized automatically the first time you run `scripts/01_extract_context_from_warc.py`. Ensure the `data/database/` directory exists or is created by the script.

**6. Configuration**

* Default paths (database, models, WARC directory) and parameters (model settings, thresholds) are set in `src/common/config.py`.
* You can modify this file directly or override settings using command-line arguments provided by the scripts in the `scripts/` directory.
* (Optional) A `config.yaml` file can be used for configuration if implemented in `src/common/config.py`.

**7. Usage: Running the Pipeline**

Execute the scripts in the `scripts/` directory sequentially. Use `--help` with any script to see available command-line arguments.

1.  **Extract Context (Stage 1/3a):**
    * Process WARC files and populate the `ImageContext` table.
    * Use `--mode training` to flag candidates for model training based on alt text presence.
    * Use `--mode inference` to extract all contexts for later prediction.
    ```bash
    # Example: Process a directory for training
    python scripts/01_extract_context_from_warc.py /path/to/your/warc/directory --mode training --db-path data/database/image_data.db

    # Example: Process a single file for inference
    python scripts/01_extract_context_from_warc.py /path/to/your/warc/file.warc.gz --mode inference
    ```

2.  **Train Span Model (Stage 2):**
    * Trains the DistilBERT model using data flagged with `is_training_candidate=1`.
    * Saves the best model to the path specified by `--model-save-dir` (defaults to `models/span_predictor/`).
    ```bash
    python scripts/02_trainSpanModel.py --db-path data/database/image_data.db --model-save-dir models/span_predictor/ --epochs 5
    ```

3.  **Predict Descriptions (Stage 3b):**
    * Uses the trained model to predict description spans for contexts in the database.
    * Saves predictions to the `PredictedDescriptions` table.
    ```bash
    python scripts/03_predict_descriptions.py --model-path models/span_predictor/model-best.h5 --db-path data/database/image_data.db
    ```
    *(Adjust `--model-path` to your saved model file/directory)*

4.  **Enrich with CLIP Scores (Stage 4):**
    * Downloads images for predictions marked 'pending'.
    * Calculates CLIP scores.
    * Updates `PredictedDescriptions` status and saves results to `ClipResults`.
    ```bash
    python scripts/04_enrich_clip_scores.py --db-path data/database/image_data.db --images-dir ./images_output
    ```
    *(Ensure `--images-dir` has sufficient storage space)*

5.  **Analyze Results (Stage 5):**
    * Fetches final data from the database.
    * Generates analysis plots and statistics (implementation details may vary).
    ```bash
    python scripts/05_analyze_results.py --db-path data/database/image_data.db --output-dir ./analysis_output --min-clip 0.2
    ```

**8. Testing**

* Install testing requirements: `pip install pytest`
* Run tests from the project root directory:
    ```bash
    pytest tests/
    ```

**9. License**

* 

**10. Contributing**

* 
