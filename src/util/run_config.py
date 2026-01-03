import argparse
from pathlib import Path
from src.util.paths import parse_dat_dir


def get_args():
    """Parses console arguments for running scripts."""

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

    return args


def get_dat_dir_args() -> Path:
    """Parses console arguments for running scripts and returns dat_dir"""

    args = get_args()
    return parse_dat_dir(args.dat_dir)