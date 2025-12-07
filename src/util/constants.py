"""
constants.py
---

Provides constants for project.
"""

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

TEST_FILE = "testfile.csv"
TRAIN_FILE = "trainfile.csv"
VAL_FILE = "valfile.csv"
CLASS_INDEX_FILE = "class_index.csv"
SPLIT_FILES = TEST_FILE, TRAIN_FILE, VAL_FILE
