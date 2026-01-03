"""
util.py
---

Provides utility functions for task3.
"""

import pandas as pd
from pathlib import Path
from src.constants import CLASS_INDEX_FILE


def parse_class_index(dataset_dir: Path) -> pd.DataFrame:
    """Parses class to index file as pd.DataFrame"""

    path = dataset_dir / CLASS_INDEX_FILE
    assert path.is_file(), f"Class index file not found at {path}"
    df = pd.read_csv(path)
    return df


def class_to_index_map(dataset_dir: Path) -> dict[str, int]:
    """Returns dict of class (key), index (value)"""
    df = parse_class_index(dataset_dir)
    return dict(zip(df["class"], df["index"]))


def index_to_class_map(dataset_dir: Path) -> dict[str, int]:
    """Returns dict of index (key), class (value)"""
    df = parse_class_index(dataset_dir)
    return dict(zip(df["index"], df["class"]))
