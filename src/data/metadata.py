from pathlib import Path
import re

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAMEL_ROOT = PROJECT_ROOT / "src" / "data" / "CAMEL"
LABEL_PATH = CAMEL_ROOT / "label.csv"


# ============================================================
# CAMEL filename format
#
# image_<WSI_ID>_<X>_<Y>.png
#
# Example:
# image_000001_10240_25600.png
# ============================================================

FILENAME_PATTERN = re.compile(
    r"image_(\d{6})_(\d+)_(\d+)\.png$"
)


def parse_filename(filename: str) -> tuple[str, int, int]:
    """
    Parse metadata encoded in a CAMEL filename.

    Parameters
    ----------
    filename : str
        Example:
        image_000001_10240_25600.png

    Returns
    -------
    tuple[str, int, int]
        wsi_id, x, y
    """

    match = FILENAME_PATTERN.fullmatch(filename)

    if match is None:
        raise ValueError(
            f"Invalid CAMEL filename format: {filename}"
        )

    wsi_id = match.group(1)
    x = int(match.group(2))
    y = int(match.group(3))

    return wsi_id, x, y


def build_image_index(camel_root: Path = CAMEL_ROOT) -> dict[str, Path]:
    """
    Scan all patches-* directories and create a mapping:

        filename -> absolute image path

    Raises an error if the same filename appears more than once.
    """

    image_index = {}

    patch_dirs = sorted(camel_root.glob("patches-*"))

    if not patch_dirs:
        raise FileNotFoundError(
            f"No patches-* directories found in: {camel_root}"
        )

    for patch_dir in patch_dirs:

        if not patch_dir.is_dir():
            continue

        for image_path in patch_dir.glob("*.png"):

            filename = image_path.name

            if filename in image_index:
                raise ValueError(
                    "Duplicate image filename found:\n"
                    f"{filename}\n"
                    f"Existing: {image_index[filename]}\n"
                    f"Duplicate: {image_path}"
                )

            image_index[filename] = image_path.resolve()

    return image_index


def load_metadata(
    label_path: Path = LABEL_PATH,
    camel_root: Path = CAMEL_ROOT,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load and construct CAMEL metadata.

    Output columns
    --------------
    filename
    label
    wsi_id
    x
    y
    image_path
    """

    # --------------------------------------------------------
    # 1. Check paths
    # --------------------------------------------------------

    if not label_path.exists():
        raise FileNotFoundError(
            f"label.csv not found: {label_path}"
        )

    if not camel_root.exists():
        raise FileNotFoundError(
            f"CAMEL root not found: {camel_root}"
        )

    # --------------------------------------------------------
    # 2. Read label.csv
    #
    # CAMEL label.csv has NO HEADER
    # --------------------------------------------------------

    df = pd.read_csv(
        label_path,
        header=None,
        names=["filename", "label"],
    )

    # --------------------------------------------------------
    # 3. Parse filename
    # --------------------------------------------------------

    parsed = df["filename"].apply(parse_filename)

    df[["wsi_id", "x", "y"]] = pd.DataFrame(
        parsed.tolist(),
        index=df.index,
    )

    # --------------------------------------------------------
    # 4. Build image index
    # --------------------------------------------------------

    image_index = build_image_index(camel_root)

    # --------------------------------------------------------
    # 5. Resolve each filename to its actual file path
    # --------------------------------------------------------

    df["image_path"] = df["filename"].map(image_index)

    # Convert Path / NaN into strings for easier CSV export later.
    df["image_path"] = df["image_path"].apply(
        lambda path: str(path) if isinstance(path, Path) else None
    )

    # --------------------------------------------------------
    # 6. Validation
    # --------------------------------------------------------

    if validate:
        validate_metadata(
            df=df,
            image_index=image_index,
        )

    # --------------------------------------------------------
    # 7. Stable ordering
    # --------------------------------------------------------

    df = df.sort_values(
        by=["wsi_id", "x", "y"]
    ).reset_index(drop=True)

    return df


def validate_metadata(
    df: pd.DataFrame,
    image_index: dict[str, Path],
) -> None:
    """
    Validate consistency between label.csv and image files.
    """

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    if df[["filename", "label", "wsi_id", "x", "y"]].isnull().any().any():
        raise ValueError(
            "Missing values detected in CAMEL metadata."
        )

    # --------------------------------------------------------
    # Duplicate filename in label.csv
    # --------------------------------------------------------

    duplicated_filenames = df["filename"].duplicated()

    if duplicated_filenames.any():
        duplicates = df.loc[
            duplicated_filenames,
            "filename",
        ].tolist()

        raise ValueError(
            f"Duplicated filenames in label.csv: {duplicates[:10]}"
        )

    # --------------------------------------------------------
    # Binary labels only
    # --------------------------------------------------------

    valid_labels = {0, 1}
    observed_labels = set(df["label"].unique())

    if not observed_labels.issubset(valid_labels):
        raise ValueError(
            f"Unexpected labels found: {observed_labels}"
        )

    # --------------------------------------------------------
    # Check missing image files
    # --------------------------------------------------------

    missing_images = df[df["image_path"].isnull()]

    if not missing_images.empty:
        example = missing_images["filename"].head(10).tolist()

        raise FileNotFoundError(
            f"{len(missing_images)} images listed in label.csv "
            f"were not found in patches-* directories.\n"
            f"Examples: {example}"
        )

    # --------------------------------------------------------
    # Check images that exist but are not in label.csv
    # --------------------------------------------------------

    csv_filenames = set(df["filename"])
    disk_filenames = set(image_index.keys())

    unlabelled_images = disk_filenames - csv_filenames

    if unlabelled_images:
        examples = sorted(unlabelled_images)[:10]

        raise ValueError(
            f"{len(unlabelled_images)} image files exist on disk "
            f"but are missing from label.csv.\n"
            f"Examples: {examples}"
        )

    # --------------------------------------------------------
    # Duplicate spatial position
    # --------------------------------------------------------

    duplicated_positions = df.duplicated(
        subset=["wsi_id", "x", "y"]
    )

    if duplicated_positions.any():
        duplicates = df.loc[
            duplicated_positions,
            ["filename", "wsi_id", "x", "y"],
        ].head(10)

        raise ValueError(
            "Duplicate spatial positions detected:\n"
            f"{duplicates}"
        )


def get_wsi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ROI-level metadata into one row per WSI.

    Output columns
    --------------
    wsi_id
    roi_count
    normal_count
    adenoma_count
    adenoma_ratio
    """

    summary = (
        df.groupby("wsi_id")
        .agg(
            roi_count=("filename", "count"),
            normal_count=("label", lambda x: (x == 0).sum()),
            adenoma_count=("label", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )

    summary["adenoma_ratio"] = (
        summary["adenoma_count"]
        / summary["roi_count"]
    )

    return summary


if __name__ == "__main__":

    metadata = load_metadata()

    print("=" * 70)
    print("CAMEL Metadata")
    print("=" * 70)

    print("\n[ROI metadata]")
    print(metadata.head())

    print("\nShape:")
    print(metadata.shape)

    print("\n[WSI summary]")
    wsi_summary = get_wsi_summary(metadata)

    print(wsi_summary.head())

    print("\nSummary")
    print(f"ROI images : {len(metadata):,}")
    print(f"WSIs       : {metadata['wsi_id'].nunique()}")
    print(f"Normal     : {(metadata['label'] == 0).sum():,}")
    print(f"Adenoma    : {(metadata['label'] == 1).sum():,}")
