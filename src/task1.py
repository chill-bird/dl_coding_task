"""
Task 1 - Data splitting

Perform a train-val-test split of the data which depends on a seed parameter, and use a manual
seed so that you and us can reproduce the same split your experiments. Use at least 2500
images for training, 1000 for validation and 2000 for testing.
"""

from pathlib import Path
from zipfile import extractall

FLOWERS_ZIP_FILENAME = "102flowersn.zip"
EURO_SAT_MS_ZIP_FILENAME = "EuroSAT_MS.zip"
EURO_SAT_RGB_ZIP_FILENAME = "EuroSAT_RGB.zip"


def unzip_data(dat_dir: Path):
    """Unzips data located at dat_dir"""

    for zip_file in [FLOWERS_ZIP_FILENAME, EURO_SAT_MS_ZIP_FILENAME, EURO_SAT_RGB_ZIP_FILENAME]:
        path = Path(dat_dir / zip_file).resolve()
        assert path.is_file(f"Could not find data file at {path}")
        extractall(path)


def split_data(dat_dir: Path):
    """Creates train-val-test split of the data located at dat_dir."""
    unzip_data()
