"""
main.py
---

Runs project
"""

import argparse
from src.util.constants import DATASETS, CLASS_INDEX_FILE, TEST_FILE, TRAIN_FILE, VAL_FILE
from src.util.paths import parse_dat_dir, root_path
from src.task1.split_data import split_data
from src.task2.eurosat_dataset import EuroSatDataset

SEED = 3780947
DATASET = "euro_sat_ms"
# DATASET = "euro_sat_rgb"


def main():
    """Runs tasks"""

    parser = argparse.ArgumentParser(description="DL for Master Students Coding task")
    parser.add_argument("--dat_dir", "-d", type=str, default=None, help="Path to data directory")
    args = parser.parse_args()

    # Dataset parent directory (dat_dir) containing zip files
    dat_dir = parse_dat_dir(args.dat_dir)
    # Dataset directory (dataset_dir) containing image data
    dataset_dir = dat_dir / DATASETS[DATASET]["unzip_dirname"]
    img_format = DATASETS[DATASET]["format"]

    print(f"Settings:\nROOT DIR:{root_path()}\nDAT_DIR:  {dat_dir}\nIMG_EXT:  {img_format}\n")

    # Task 1
    split_data(dat_dir, DATASET, seed=SEED)

    # Task 2


if __name__ == "__main__":
    main()
