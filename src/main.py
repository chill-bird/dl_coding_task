"""
main.py
---

Runs project
"""

import argparse
from src.constants import (
    BATCH_SIZE,
    CLASS_INDEX_FILE,
    DATASETS,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    SPLIT_FILES,
)
from src.util.paths import parse_dat_dir, root_path
from src.task1.split_data import split_data
from src.task2.fine_tune import fine_tune

DATASET = "euro_sat_ms"
# DATASET = "euro_sat_rgb"


def main():
    """Runs tasks"""

    parser = argparse.ArgumentParser(description="DL for Master Students Coding task")
    parser.add_argument(
        "--dat_dir",
        "-D",
        type=str,
        default=None,
        help="Path to data directory containing zip files",
    )
    parser.add_argument(
        "--root_dir",
        "-R",
        type=str,
        default=None,
        help="Path to root directory of the project code",
    )
    args = parser.parse_args()

    # Dataset parent directory (dat_dir) containing zip files
    dat_dir = parse_dat_dir(args.dat_dir)

    # Dataset directory (dataset_dir) containing image data
    dataset_dir = dat_dir / DATASETS[DATASET]["unzip_dirname"]
    # File extension of image data
    img_format = DATASETS[DATASET]["format"]

    print(f"Settings:\nROOT DIR:{root_path()}\nDAT_DIR:  {dat_dir}\nIMG_EXT:  {img_format}\n")

    # Task 1
    split_data(dat_dir, DATASET, seed=SEED)

    # Task 2
    # fine_tune(
    #     dataset_dir=dataset_dir,
    #     img_format=img_format,
    #     class_index_file_name=CLASS_INDEX_FILE,
    #     split_files=SPLIT_FILES,
    #     learning_rate=LEARNING_RATE,
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     seed=SEED,
    # )


if __name__ == "__main__":
    main()
