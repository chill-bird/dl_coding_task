"""
constants.py
---

Provides constants for project.
"""

### Change for each run ###

TIF_DATASET = "euro_sat_ms"
TIF_CHANNELS = [3,2,1,7,10,11] # Bands: Red, Green, Blue, NIR, SWIR1, SWIR2
RGB_DATASET = "euro_sat_rgb"


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
}

RGB_DATASET_DIR_NAME = DATASETS[RGB_DATASET]["unzip_dirname"]
TIF_DATASET_DIR_NAME = DATASETS[TIF_DATASET]["unzip_dirname"]

OUTPUT_DIR_NAME = "results"
BEST_MODEL_FILENAME = "best_model.pt"
REPRODUCE_OUTPUT_DIR_NAME = "test"
LOGITS_TEST_SET_FILE = "test_logits_best_model.npy"
REPRODUCED_LOGITS_TEST_SET_FILE = "reproduced_test_logits_best_model.npy"

OUTPUT_DIR_FINETUNED_TASK2 = "task2_finetuned"
OUTPUT_DIR_FINETUNED_TASK3 = "task3_finetuned"

TEST_FILE = "testfile.csv"
TRAIN_FILE = "trainfile.csv"
VAL_FILE = "valfile.csv"
CLASS_INDEX_FILE = "class_index.csv"
SPLIT_FILES = {"test": TEST_FILE, "train": TRAIN_FILE, "val": VAL_FILE}

LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
