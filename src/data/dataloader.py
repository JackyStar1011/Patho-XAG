from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CAMELDataset
from transforms import (
    get_train_roi_transform,
    get_eval_roi_transform,
    get_image_transform,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "splits"


# ============================================================
# DataLoader
# ============================================================

def create_dataloader(
    split,
    mode,
    batch_size=2,
    num_workers=0,
):
    """
    Create a DataLoader for train, val, or test.

    split:
        train
        val
        test

    mode:
        whole
        patch
    """

    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Invalid split: {split}"
        )

    if mode not in {"whole", "patch"}:
        raise ValueError(
            f"Invalid mode: {mode}"
        )

    csv_path = SPLIT_DIR / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {csv_path}"
        )

    # Training uses augmentation.
    if split == "train":
        roi_transform = get_train_roi_transform()
        shuffle = True

    else:
        roi_transform = get_eval_roi_transform()
        shuffle = False

    image_transform = get_image_transform()

    dataset = CAMELDataset(
        csv_path=csv_path,
        mode=mode,
        roi_transform=roi_transform,
        transform=image_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Whole-image
    # --------------------------------------------------------

    whole_loader = create_dataloader(
        split="train",
        mode="whole",
        batch_size=2,
        num_workers=0,
    )

    whole_batch = next(iter(whole_loader))

    print("=" * 60)
    print("Whole-image DataLoader")
    print("=" * 60)

    print(
        f"Image batch : "
        f"{whole_batch['image'].shape}"
    )

    print(
        f"Label batch : "
        f"{whole_batch['label'].shape}"
    )

    # --------------------------------------------------------
    # Patch
    # --------------------------------------------------------

    patch_loader = create_dataloader(
        split="train",
        mode="patch",
        batch_size=2,
        num_workers=0,
    )

    patch_batch = next(iter(patch_loader))

    print("\n" + "=" * 60)
    print("Patch DataLoader")
    print("=" * 60)

    print(
        f"Image batch : "
        f"{patch_batch['image'].shape}"
    )

    print(
        f"Label batch : "
        f"{patch_batch['label'].shape}"
    )