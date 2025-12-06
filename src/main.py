"""
main.py
---

Runs project
"""

import argparse
import torch
from util.paths import parse_dat_dir, root_path
from util.seed import set_seed
from task1 import split_data

SEED = 3780947
DATASET = "euro_sat_ms"
# DATASET = "euro_sat_rgb"


def main():
    """Runs tasks"""

    # (
    #     print("CUDA available.\n")
    #     if torch.cuda.is_available()
    #     else print("CUDA not available, using CPU...\n")
    # )

    parser = argparse.ArgumentParser(description="DL for Master Students Coding task")
    parser.add_argument("--dat_dir", "-d", type=str, default=None, help="Path to data directory")
    args = parser.parse_args()

    dat_dir = parse_dat_dir(args.dat_dir)

    print(f"Settings:\nROOT DIR:{root_path()}\nDAT_DIR:  {dat_dir}\n")
    
    # Task 1
    split_data(dat_dir, DATASET, seed=SEED)

    # Task 2


if __name__ == "__main__":
    main()
