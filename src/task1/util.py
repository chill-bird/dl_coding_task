"""
task1_util.py
---

Provides utility functions for task1.
"""

import numpy as np
from pathlib import Path
import zipfile


def unzip_data(data_dict, dat_dir: Path, check_if_exist: bool = False) -> None:
    """Unzips data specified in data_dict to at dat_dir"""

    # Iterate over datasets defined in data_dict
    for _, info_dict in data_dict.items():

        # Find zip files
        zip_filename = info_dict["zip_filename"]
        path = Path(dat_dir / zip_filename).resolve()
        assert path.is_file(), f"Could not find data file at {path}"

        # Skip unzipping if unzipped files already exist
        if check_if_exist:
            unzipped_dir = Path(dat_dir / info_dict["unzip_dirname"]).resolve()
            if unzipped_dir.is_dir():
                print(f"Skipping unzipping, because {info_dict["unzip_dirname"]} already exists")
                continue

        # Unzip files
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(dat_dir)


def print_split_stats(train_files: list[Path], val_files: list[Path], test_files: list[Path]):
    """Prints statistics about the splits."""
    total = len(train_files) + len(val_files) + len(test_files)
    print(f"Training set:   {len(train_files):5d} images ({len(train_files)/total*100:5.1f}%)")
    print(f"Validation set: {len(val_files):5d} images ({len(val_files)/total*100:5.1f}%)")
    print(f"Test set:       {len(test_files):5d} images ({len(test_files)/total*100:5.1f}%)")
    print(f"Total:          {total:5d} images")


def verify_size_requirements(
    all_files: np.ndarray, train_files: np.ndarray, val_files: np.ndarray, test_files: np.ndarray
) -> None:
    """Verifies size requirements for all data subsets."""

    assert (
        len(all_files) >= 5500
    ), f"ERROR: Not enough images: Expect min 5500, got {len(all_files)}"
    assert len(train_files) >= 2500, f"ERROR: Training set too small: {len(train_files)} < 2500"
    assert len(val_files) >= 1000, f"ERROR: Validation set too small: {len(val_files)} < 1000"
    assert len(test_files) >= 2000, f"ERROR: Test set too small: {len(test_files)} < 2000"


def verify_disjoint(train_files: list[Path], val_files: list[Path], test_files: list[Path]):
    """
    Verifies that train, val, and test splits are disjoint, meaning that they do not have any element in common.
    """

    # Assert no duplicates in training samples
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)

    assert len(train_set) == len(train_files), "Duplicates detected in train set"
    assert len(val_set) == len(val_files), "Duplicates detected in val set"
    assert len(test_set) == len(test_files), "Duplicates detected in test set"

    # Assert empty intersections between two sets each, resulting in overall disjoint sets
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    assert (
        train_test_overlap == set()
    ), f"ERROR: {len(train_val_overlap)} files overlap between train and val"
    assert (
        train_test_overlap == set()
    ), f"ERROR: {len(train_test_overlap)} files overlap between train and test"
    assert (
        val_test_overlap == set()
    ), f"ERROR: {len(val_test_overlap)} files overlap between val and test"
