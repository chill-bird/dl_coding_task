"""
constants.py
---

Provides constants for project.
"""

### Change for each run ###

# DATASET = "euro_sat_ms"
DATASET = "euro_sat_rgb"


### Fixed constants ###

SEED = 3780947

DATASETS = {
    # Multispectral TIF
    "euro_sat_ms": {
        "zip_filename": "EuroSAT_MS.zip",
        "unzip_dirname": "EuroSAT_MS",
        "format": ".tif",
    },
    # JPG (RGB)
    "euro_sat_rgb": {
        "zip_filename": "EuroSAT_RGB.zip",
        "unzip_dirname": "EuroSAT_RGB",
        "format": ".jpg",
    },
    # EXAMPLE
    "flowers": {
        "zip_filename": "102flowersn.zip",
        "unzip_dirname": "flowers_data",
        "format": ".jpg",
    },
}

DATASET_DIR_NAME = DATASETS[DATASET]["unzip_dirname"]
IMG_FORMAT = DATASETS[DATASET]["format"]

OUTPUT_DIR_NAME = "results"
BEST_MODEL_FILENAME = "test_logits_best_model.npy"
PREDICTIONS_DIR_NAME = "predictions"

TEST_FILE = "testfile.csv"
TRAIN_FILE = "trainfile.csv"
VAL_FILE = "valfile.csv"
CLASS_INDEX_FILE = "class_index.csv"
SPLIT_FILES = {"test": TEST_FILE, "train": TRAIN_FILE, "val": VAL_FILE}

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
