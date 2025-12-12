"""
data_loader.py
---

Provides data loaders for EuroSatDatasets.
"""

from torch.utils.data import DataLoader
from pathlib import Path

from src.task2.eurosat_dataset import EuroSatDataset
from src.util.constants import CLASS_INDEX_FILE


def dataloaders(
    dataset_dir: Path,
    split_files: dict,
    img_format: str,
    aug_dict: dict,
    aug_name: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders."""

    train_file = split_files["train"]
    test_file = split_files["test"]
    val_file = split_files["val"]

    train_dataset = EuroSatDataset(
        root_dir=dataset_dir,
        split_file=train_file,
        class_index_file=CLASS_INDEX_FILE,
        img_format=img_format,
        transform=aug_dict[aug_name],
    )

    val_dataset = EuroSatDataset(
        root_dir=dataset_dir,
        split_file=val_file,
        class_index_file=CLASS_INDEX_FILE,
        img_format=img_format,
        transform=aug_dict["val"],
    )

    test_dataset = EuroSatDataset(
        root_dir=dataset_dir,
        split_file=test_file,
        class_index_file=CLASS_INDEX_FILE,
        img_format=img_format,
        transform=aug_dict["val"],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, test_dataset
