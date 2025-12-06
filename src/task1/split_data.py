"""
Task 1 - Data splitting

Perform a train-val-test split of the data which depends on a seed parameter, and use a manual
seed so that you and us can reproduce the same split your experiments. Use at least 2500
images for training, 1000 for validation and 2000 for testing.
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from src.util.paths import dataset_path
from src.util.seed import set_seed
from src.task1.util import (
    print_split_stats,
    unzip_data,
    verify_disjoint,
    verify_size_requirements,
)

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


def get_filenames_and_classes(dataset_dir: Path, img_format: str) -> tuple[list[Path], list[str]]:
    """
    Collects all image file names from a dataset directory and its subdirectories.

    Args:
        dataset_dir: Directory containing subdirectories of images where subdirectory name specifies ground truth class
        img_format: Image file extension of dataset
    Returns:
        tuple: (list of image file paths, list of corresponding class labels)
    """

    dataset_dir = dataset_dir.resolve()
    assert dataset_dir.is_dir(), f"Dataset directory does not exist at {dataset_dir}"

    image_files = []
    labels = []

    # Collect images from subdirectories
    for class_dir in sorted(dataset_dir.iterdir()):

        if class_dir.is_dir():

            class_name = class_dir.name

            # Collect all image files in this class
            for image_file in sorted(class_dir.iterdir()):
                if image_file.is_file() and image_file.suffix.lower() == img_format:
                    image_files.append(f"{class_name}/{image_file.stem}")
                    labels.append(class_name)

    return image_files, labels


def create_split_files(
    target_dir: str | Path,
    train_files: np.ndarray,
    train_labels: np.ndarray,
    val_files: np.ndarray,
    val_labels: np.ndarray,
    test_files: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    """Creates 3 split files for train, val and test set as CSV files

    Args:
        target_dir: Directory to save split files to
        train_files: Array containing filenames of train set
        train_labels: Array containing labels of train set
        val_files: Array containing filenames of val set
        val_labels: Array containing labels of val set
        test_files: Array containing filenames of test set
        test_labels: Array containing labels of test set
    """

    assert (
        target_dir.is_dir()
    ), f"Target directory to save split files does not exist at {target_dir}"

    # Train file
    with open(target_dir / "trainfile.csv", "w", encoding="UTF-8") as f:
        f.write("file,label\n")
        for file, label in zip(train_files, train_labels):
            f.write(f"{file},{label}\n")

    # Val file
    with open(target_dir / "valfile.csv", "w", encoding="UTF-8") as f:
        f.write("file,label\n")
        for file, label in zip(val_files, val_labels):
            f.write(f"{file},{label}\n")

    # Test file
    with open(target_dir / "testfile.csv", "w", encoding="UTF-8") as f:
        f.write("file,label\n")
        for file, label in zip(test_files, test_labels):
            f.write(f"{file},{label}\n")


def split_data(dat_dir: Path, dataset_name: str, seed: int):
    """
    Creates stratified train-val-test split of the data located at dat_dir.
    Data in both datasets are equivalent except their image format.

    Args:
        dat_dir: Path to the data directory
        dataset_name: Name of the dataset
        seed: Random seed for reproducibility

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' keys with lists of file paths
    """

    print("=" * 60 + "\nTASK 1 - Data Splitting\n" + "=" * 60 + "\n")

    set_seed(seed)

    print("[1] Unzipping all available data")
    unzip_data(DATASETS, dat_dir, check_if_exist=True)

    print(f"\n[2] Collecting labelled images from {dataset_name} dataset")
    dataset_dir = dataset_path(DATASETS, dataset_name, dat_dir)
    img_format = DATASETS[dataset_name]["format"]

    all_imgs, labels = get_filenames_and_classes(dataset_dir, img_format)
    all_imgs = np.array(all_imgs)
    labels = np.array(labels)
    print(f"Found {len(all_imgs)} images across {len(np.unique(labels))} classes")

    print("\n[3] Creating Splits")
    # First split: 60% train, 40% temp
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_imgs, labels, test_size=0.40, random_state=seed, stratify=labels
    )

    # Second split: Split the remaining 40% into 50% val and 50% test
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    print_split_stats(train_imgs, val_imgs, test_imgs)

    print("\n[4] Creating Split Files")
    create_split_files(
        dat_dir, train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels
    )

    print("\n[5] Checking split requirements")
    verify_size_requirements(all_imgs, train_imgs, val_imgs, test_imgs)
    verify_disjoint(train_imgs, val_imgs, test_imgs)

    print("\n[✓] TASK 1")
