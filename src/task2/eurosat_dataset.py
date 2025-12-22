"""
eurosat_dataset.py
---

Provides custom class for EuroSAT dataset.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset
from src.task2._classname_index_mapping import class_to_index_map


class EuroSatDataset(Dataset):
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
        assert self.img_format in [
            ".jpg",
            ".tif",
        ], f"Image file extension must be one of '.jpg' or '.tif'. Got {img_format} instead"

    def __len__(self):
        """Sample size"""
        return len(self.samples)

    def __getitem__(self, index):
        """Get a single sample"""
        sample = self.samples[index]
        file_path = sample["file"]
        img = (
            Image.open(file_path).convert("RGB") if self.img_format == ".jpg" else imread(file_path)
        )
        img = self.transform(img)

        label = sample["label"]
        return {"image": img, "label": label}

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
