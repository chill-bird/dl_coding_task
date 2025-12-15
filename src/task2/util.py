"""
util.py
---

Provides utility functions for task2.
"""

import pandas as pd
from pathlib import Path


def parse_class_index(class_index_file: Path) -> pd.DataFrame:
    """Parses class to index file as pd.DataFrame"""

    assert class_index_file.is_file(), f"Class index file not found at {class_index_file}"
    df = pd.read_csv(class_index_file)
    return df


def class_to_index_map(class_index_file: Path) -> dict[str, int]:
    """Returns dict of class (key), index (value)"""
    df = parse_class_index(class_index_file)
    return dict(zip(df["class"], df["index"]))


def index_to_class_map(class_index_file: Path) -> dict[str, int]:
    """Returns dict of index (key), class (value)"""
    df = parse_class_index(class_index_file)
    return dict(zip(df["index"], df["class"]))
