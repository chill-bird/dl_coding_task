"""
multispec_dataset.py
---

Provides custom class for EuroSAT dataset.
"""

import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from src.task2._classname_index_mapping import class_to_index_map
import numpy as np
import tifffile as tiff
from src.constants import TIF_CHANNELS


class MultiSpecDataset(Dataset):
    """Represents Dataset class for EuroSAT images"""

    def __init__(
        self,
        root_dir: str | Path,
        split_file: str | Path,
        img_format=str,
        transform=None,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform
        self.split_file = Path(self.root_dir / split_file).resolve()
        self.class_to_index_map = class_to_index_map(self.root_dir)
        self.samples = self.read_imgs_from_split_file()
        self.img_format = img_format
        assert (
            self.img_format == ".tif"
        ), "Image format must be set to '.tif' for task 3."

    def __len__(self):
        """Sample size"""
        return len(self.samples)

    def __getitem__(self, index):
        """Get a single sample"""
        sample = self.samples[index]
        file_path = sample["file"]

        img = self._load_multichannel_tiff(file_path, TIF_CHANNELS)
        img = self.transform(img)
        label = sample["label"]

        return {"image": img, "label": label}

    def _load_multichannel_tiff(self, file_path, selected_channels):
        """Loads a multi-channel TIFF image and selects specified channels."""

        img = tiff.imread(file_path)
        img_subset = img[:, :, selected_channels]
        tensor = img_subset.astype(np.float32)
        return tensor

    def read_imgs_from_split_file(self):
        """Saves entries from split files as samples."""

        assert self.split_file.is_file(), f"Split file not found at {self.split_file}"

        # Read image files and labels from CSV
        df = pd.read_csv(self.split_file)
        # Convert relative paths to absolute paths using pathlib
        df["file"] = df["file"].apply(lambda f: self.root_dir / f)
        df["label"] = df["label"].apply(lambda c: self.class_to_index_map[c])

        result = df.to_dict("records")
        return result
