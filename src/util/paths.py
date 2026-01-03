"""
paths.py
---

Provides utility functions for locating paths in the project.
"""

from src.constants import OUTPUT_DIR_NAME
from pathlib import Path


def root_path() -> Path:
    """Returns root path of the project"""
    return Path(__file__).parent.parent.parent.resolve()


def parse_dat_dir(path: str | Path | None) -> Path:
    """Returns data directory of the project"""
    if path is None:
        root = Path(root_path())
        dat_dir = Path(root / "dat").resolve()
        print(f"Dat dir not set, using default path at {dat_dir}.\n")
    else:
        dat_dir = Path(path).resolve()

    assert dat_dir.is_dir(), f"Data directory not found at {dat_dir}."
    check_if_datasets_exist(dat_dir)
    return dat_dir


def results_parent_dir() -> Path:
    """
    Returns path to results parent directory.
    Files for each run are saved in subdirectories.
    """
    path = Path(root_path() / OUTPUT_DIR_NAME).resolve()
    return path


def dataset_path(datasets: dict[str, dict], dataset_name: str, dat_dir: Path) -> Path:
    """Returns dataset directory (subdirectory of dat_dir)"""
    dataset_dir_name = datasets[dataset_name]["unzip_dirname"]
    dataset_dir = Path(dat_dir / dataset_dir_name).resolve()
    return dataset_dir


def check_if_datasets_exist(dat_dir: Path):
    assert dat_dir.is_dir(), f"Data directory not found at {dat_dir}."

    eurosat_ms_file = "EuroSAT_MS.zip"
    eurosat_rgb_file = "EuroSAT_RGB.zip"
    eurosat_ms_file_location = Path(dat_dir / eurosat_ms_file).resolve()
    eurosat_rgb_file_location = Path(dat_dir / eurosat_rgb_file).resolve()

    assert (
        eurosat_ms_file_location.is_file()
    ), f"Euro Sat MS data set does not exist at {eurosat_ms_file_location}"
    assert (
        eurosat_rgb_file_location.is_file()
    ), f"Euro Sat RGB data set does not exist at {eurosat_rgb_file_location}"


def find_most_recent_train_results_dir() -> Path:
    """Returns path to output directory of most recent run"""

    results_parent_path = results_parent_dir()
    target_dir = sorted(results_parent_path.iterdir(), reverse=True)[0]
    return target_dir.resolve()
