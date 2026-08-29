from pathlib import Path
import re

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAMEL_ROOT = PROJECT_ROOT / "src" / "data" / "CAMEL"
LABEL_PATH = CAMEL_ROOT / "label.csv"


# ============================================================
# Filename parser
# ============================================================

FILENAME_PATTERN = re.compile(
    r"image_(\d{6})_(\d+)_(\d+)\.png$"
)


def parse_filename(filename: str) -> tuple[str, int, int]:
    """
    Parse CAMEL filename.

    Example:
        image_000001_10240_25600.png

    Returns:
        wsi_id = "000001"
        x = 10240
        y = 25600
    """
    match = FILENAME_PATTERN.match(filename)

    if match is None:
        raise ValueError(
            f"Cannot parse CAMEL filename: {filename}"
        )

    wsi_id = match.group(1)
    x = int(match.group(2))
    y = int(match.group(3))

    return wsi_id, x, y


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("CAMEL Dataset Inspection")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check paths
    # --------------------------------------------------------

    print(f"\nCAMEL root : {CAMEL_ROOT}")
    print(f"Label file : {LABEL_PATH}")

    if not CAMEL_ROOT.exists():
        raise FileNotFoundError(
            f"CAMEL directory not found: {CAMEL_ROOT}"
        )

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"label.csv not found: {LABEL_PATH}"
        )

    # --------------------------------------------------------
    # 2. Read label.csv
    # CAMEL label.csv has NO HEADER
    # --------------------------------------------------------

    df = pd.read_csv(
        LABEL_PATH,
        header=None,
        names=["filename", "label"],
    )

    print("\n[Dataset]")
    print(f"Total metadata rows : {len(df):,}")

    # --------------------------------------------------------
    # 3. Basic validation
    # --------------------------------------------------------

    print("\n[Missing values]")
    print(df.isnull().sum())

    print("\n[Duplicated filenames]")
    print(df["filename"].duplicated().sum())

    print("\n[Unique labels]")
    print(sorted(df["label"].unique()))

    print("\n[Label distribution]")
    print(df["label"].value_counts().sort_index())

    # --------------------------------------------------------
    # 4. Parse filename metadata
    # --------------------------------------------------------

    parsed = df["filename"].apply(parse_filename)

    df[["wsi_id", "x", "y"]] = pd.DataFrame(
        parsed.tolist(),
        index=df.index,
    )

    print("\n[Parsed metadata example]")
    print(df.head())

    # --------------------------------------------------------
    # 5. WSI statistics
    # --------------------------------------------------------

    num_wsi = df["wsi_id"].nunique()

    print("\n[WSI]")
    print(f"Unique WSI : {num_wsi}")

    roi_per_wsi = df.groupby("wsi_id").size()

    print(f"Min ROI / WSI    : {roi_per_wsi.min()}")
    print(f"Max ROI / WSI    : {roi_per_wsi.max()}")
    print(f"Mean ROI / WSI   : {roi_per_wsi.mean():.2f}")
    print(f"Median ROI / WSI : {roi_per_wsi.median():.2f}")

    # --------------------------------------------------------
    # 6. Label composition per WSI
    # --------------------------------------------------------

    label_sets = df.groupby("wsi_id")["label"].apply(set)

    only_normal = (label_sets == {0}).sum()
    only_adenoma = (label_sets == {1}).sum()
    mixed = label_sets.apply(lambda labels: labels == {0, 1}).sum()

    print("\n[WSI label composition]")
    print(f"Only Normal (0) : {only_normal}")
    print(f"Only Adenoma (1): {only_adenoma}")
    print(f"Mixed labels     : {mixed}")

    # --------------------------------------------------------
    # 7. Coordinate statistics
    # --------------------------------------------------------

    print("\n[Coordinate ranges]")
    print(f"x min : {df['x'].min()}")
    print(f"x max : {df['x'].max()}")
    print(f"y min : {df['y'].min()}")
    print(f"y max : {df['y'].max()}")

    unique_x = sorted(df["x"].unique())
    unique_y = sorted(df["y"].unique())

    print(f"\nUnique x coordinates : {len(unique_x)}")
    print(f"Unique y coordinates : {len(unique_y)}")

    # --------------------------------------------------------
    # 8. Check expected grid alignment
    # --------------------------------------------------------

    ROI_SIZE = 1280

    invalid_x = df[df["x"] % ROI_SIZE != 0]
    invalid_y = df[df["y"] % ROI_SIZE != 0]

    print("\n[Coordinate alignment check]")
    print(f"x not divisible by {ROI_SIZE}: {len(invalid_x)}")
    print(f"y not divisible by {ROI_SIZE}: {len(invalid_y)}")

    # --------------------------------------------------------
    # 9. Check duplicate spatial positions
    # --------------------------------------------------------

    duplicate_positions = df.duplicated(
        subset=["wsi_id", "x", "y"],
        keep=False,
    )

    duplicate_position_count = duplicate_positions.sum()

    print("\n[Spatial duplicate check]")
    print(
        f"Duplicate (WSI, x, y) positions: "
        f"{duplicate_position_count}"
    )

    if duplicate_position_count > 0:
        print("\nDuplicated positions:")
        print(
            df.loc[
                duplicate_positions,
                ["filename", "wsi_id", "x", "y", "label"],
            ]
            .sort_values(["wsi_id", "x", "y"])
            .head(20)
        )

    # --------------------------------------------------------
    # 10. Approximate spatial extent per WSI
    # --------------------------------------------------------

    spatial_extent = (
        df.groupby("wsi_id")
        .agg(
            x_min=("x", "min"),
            x_max=("x", "max"),
            y_min=("y", "min"),
            y_max=("y", "max"),
            roi_count=("filename", "count"),
        )
        .reset_index()
    )

    spatial_extent["approx_width"] = (
        spatial_extent["x_max"]
        - spatial_extent["x_min"]
        + ROI_SIZE
    )

    spatial_extent["approx_height"] = (
        spatial_extent["y_max"]
        - spatial_extent["y_min"]
        + ROI_SIZE
    )

    print("\n[Approximate WSI spatial extent]")
    print(spatial_extent.head())

    # --------------------------------------------------------
    # 11. Check whether image files actually exist
    # --------------------------------------------------------

    def find_image(filename: str) -> bool:
        return (CAMEL_ROOT / filename).exists()

    # Only valid if images are directly inside CAMEL_ROOT.
    # If CAMEL has image subfolders, adjust this later.
    existing_count = df["filename"].apply(find_image).sum()

    print("\n[Image file check]")
    print(f"Images found directly in CAMEL root: {existing_count:,}")
    print(f"Images listed in CSV              : {len(df):,}")

    # --------------------------------------------------------
    # 12. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"ROI images : {len(df):,}")
    print(f"WSIs       : {num_wsi}")
    print(f"Normal     : {(df['label'] == 0).sum():,}")
    print(f"Adenoma    : {(df['label'] == 1).sum():,}")

    print("\nMetadata fields:")
    print("filename, label, wsi_id, x, y")


if __name__ == "__main__":
    main()
