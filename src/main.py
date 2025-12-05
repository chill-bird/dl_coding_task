"""
main.py
---

Runs project
"""

import argparse
from util.paths import parse_dat_dir
from task1 import split_data


def main():
    """Runs tasks"""
    parser = argparse.ArgumentParser(description="DL for Master Students Coding task")
    parser.add_argument("--dat_dir", "-d", type=str, default=None, help="Path to data directory")
    args = parser.parse_args()

    dat_dir = parse_dat_dir(args.dat_dir)
    print(dat_dir)
    train_val_test_split = split_data(dat_dir)


if __name__ == "__main__":
    main()
