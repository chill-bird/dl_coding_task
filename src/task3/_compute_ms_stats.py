import csv
import json
from pathlib import Path

import numpy as np
import tifffile as tiff


def read_split_csv(csv_path: Path):
    """
    Expects rows like:
      River/River_597.tif,River
      Residential/Residential_314.tif,Residential
    Returns: list of (rel_path, label_str)
    """
    pairs = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # allow accidental spaces
            rel = row[0].strip()
            label = row[1].strip() if len(row) > 1 else ""
            if rel:  # ignore empty lines
                pairs.append((rel, label))
    return pairs

def compute_channel_meanstd(train_files, dataset_root: Path, num_channels: int):
    """
    Compute global per-channel mean/std on TRAIN SPLIT ONLY.
    Assumes EuroSAT MS TIFFs with shape (H, W, C) and uint16 values.
    """

    sum_ = np.zeros(num_channels, dtype=np.float64)
    sum2 = np.zeros(num_channels, dtype=np.float64)
    n = 0  # number of pixels accumulated

    for rel_path in train_files:
        if rel_path[0] == "file":
            continue

        fn = dataset_root / rel_path[0]

        img = tiff.imread(fn)  # (H, W, C), uint16

        if img.ndim != 3:
            raise ValueError(f"{fn} has invalid shape {img.shape}, expected (H,W,C)")
        if img.shape[2] != num_channels:
            raise ValueError(
                f"{fn} has {img.shape[2]} channels, expected {num_channels}"
            )

        # Pflicht: uint16 -> float32 in [0,1]
        x = img.astype(np.float32) / 65535.0

        flat = x.reshape(-1, num_channels)  # (pixels, C)

        sum_ += flat.sum(axis=0)
        sum2 += (flat * flat).sum(axis=0)
        n += flat.shape[0]

    mean = sum_ / n
    var = sum2 / n - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))

    return mean.astype(np.float32), std.astype(np.float32)
