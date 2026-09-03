from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# CAMEL Dataset
# ============================================================

class CAMELDataset(Dataset):
    """
    Load CAMEL ROI images from a split CSV.

    Supported modes:
        whole : return the full 1280 x 1280 ROI
        patch : split the ROI into 25 patches of 256 x 256
    """

    def __init__(
        self,
        csv_path,
        mode="whole",
        roi_transform=None,
        transform=None,
        roi_size=1280,
        patch_size=256,
    ):
        self.csv_path = Path(csv_path)
        self.mode = mode

        self.roi_transform = roi_transform
        self.transform = transform

        self.roi_size = roi_size
        self.patch_size = patch_size

        # Check mode
        if self.mode not in {"whole", "patch"}:
            raise ValueError(
                f"Invalid mode: {self.mode}. "
                "Use 'whole' or 'patch'."
            )

        # Check if ROI can be split evenly
        if self.roi_size % self.patch_size != 0:
            raise ValueError(
                f"ROI size {self.roi_size} is not divisible "
                f"by patch size {self.patch_size}."
            )

        self.grid_size = self.roi_size // self.patch_size

        # Load split metadata
        self.df = pd.read_csv(
            self.csv_path,
            dtype={"wsi_id": str},
        )

        self._validate_dataframe()

    # --------------------------------------------------------
    # Basic dataset methods
    # --------------------------------------------------------

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self._resolve_image_path(
            row["image_path"]
        )

        image = self._load_image(image_path)

        if self.roi_transform is not None:
            image = self.roi_transform(image)
            
        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        if self.mode == "whole":
            image = self._prepare_whole_image(image)

            return {
                "image": image,
                "label": label,
                "filename": row["filename"],
                "wsi_id": row["wsi_id"],
                "x": int(row["x"]),
                "y": int(row["y"]),
            }

        patches = self._create_patches(image)

        return {
            "image": patches,
            "label": label,
            "filename": row["filename"],
            "wsi_id": row["wsi_id"],
            "x": int(row["x"]),
            "y": int(row["y"]),
        }

    # --------------------------------------------------------
    # Image loading
    # --------------------------------------------------------

    def _resolve_image_path(self, image_path):
        """
        Convert a saved relative path into a full local path.
        """

        path = Path(image_path)

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    def _load_image(self, image_path):
        """
        Load one ROI as an RGB image.
        """

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        expected_size = (
            self.roi_size,
            self.roi_size,
        )

        if image.size != expected_size:
            raise ValueError(
                f"Unexpected image size for {image_path.name}: "
                f"{image.size}. Expected {expected_size}."
            )

        return image

    # --------------------------------------------------------
    # Whole-image mode
    # --------------------------------------------------------

    def _prepare_whole_image(self, image):
        """
        Prepare the full ROI.
        """

        if self.transform is not None:
            image = self.transform(image)

        return image

    # --------------------------------------------------------
    # Patch mode
    # --------------------------------------------------------

    def _create_patches(self, image):
        """
        Split a 1280 x 1280 ROI into a 5 x 5 grid.

        Each patch is 256 x 256.

        Output after tensor transform:
            [25, 3, 256, 256]
        """

        patches = []

        for row in range(self.grid_size):
            for col in range(self.grid_size):

                left = col * self.patch_size
                top = row * self.patch_size

                right = left + self.patch_size
                bottom = top + self.patch_size

                patch = image.crop(
                    (left, top, right, bottom)
                )

                if self.transform is not None:
                    patch = self.transform(patch)

                patches.append(patch)

        # If transform converts PIL images to tensors,
        # combine all patches into one tensor.
        if patches and torch.is_tensor(patches[0]):
            patches = torch.stack(
                patches,
                dim=0,
            )

        return patches

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_dataframe(self):
        """
        Check required columns in the split CSV.
        """

        required_columns = {
            "filename",
            "label",
            "wsi_id",
            "x",
            "y",
            "image_path",
        }

        missing_columns = (
            required_columns
            - set(self.df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns in {self.csv_path}: "
                f"{sorted(missing_columns)}"
            )

        # CAMEL currently has binary labels
        labels = set(self.df["label"].unique())

        if not labels.issubset({0, 1}):
            raise ValueError(
                f"Unexpected labels: {labels}"
            )


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":

    transform = transforms.ToTensor()

    train_csv = PROJECT_ROOT / "splits" / "train.csv"

    # --------------------------------------------------------
    # Whole-image test
    # --------------------------------------------------------

    whole_dataset = CAMELDataset(
        csv_path=train_csv,
        mode="whole",
        transform=transform,
    )

    whole_sample = whole_dataset[0]

    print("=" * 60)
    print("Whole-image mode")
    print("=" * 60)

    print(f"Dataset size : {len(whole_dataset):,}")
    print(f"Image shape  : {whole_sample['image'].shape}")
    print(f"Label        : {whole_sample['label'].item()}")
    print(f"WSI ID       : {whole_sample['wsi_id']}")
    print(f"Position     : ({whole_sample['x']}, {whole_sample['y']})")

    # --------------------------------------------------------
    # Patch test
    # --------------------------------------------------------

    patch_dataset = CAMELDataset(
        csv_path=train_csv,
        mode="patch",
        transform=transform,
    )

    patch_sample = patch_dataset[0]

    print("\n" + "=" * 60)
    print("Patch mode")
    print("=" * 60)

    print(f"Dataset size : {len(patch_dataset):,}")
    print(f"Patch shape  : {patch_sample['image'].shape}")
    print(f"Label        : {patch_sample['label'].item()}")
    print(f"WSI ID       : {patch_sample['wsi_id']}")
    print(f"Position     : ({patch_sample['x']}, {patch_sample['y']})")
