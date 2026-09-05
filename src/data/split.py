from pathlib import Path

import numpy as np
import pandas as pd

from src.data.metadata import load_metadata, get_wsi_summary


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "splits"


# ============================================================
# Split configuration
# ============================================================

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

RANDOM_SEED = 42

# Number of candidate splits to try.
# More trials = better chance of finding a balanced split.
NUM_TRIALS = 5000


# ============================================================
# Helper functions
# ============================================================

def compute_target_wsi_counts(
    num_wsi: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    """
    Convert split ratios into exact integer WSI counts.

    Example for 177 WSI:
        train = 124
        val   = 27
        test  = 26
    """

    split_names = list(ratios.keys())

    raw_counts = {
        name: num_wsi * ratios[name]
        for name in split_names
    }

    counts = {
        name: int(np.floor(raw_counts[name]))
        for name in split_names
    }

    remaining = num_wsi - sum(counts.values())

    # Give remaining WSIs to splits with largest fractional parts
    fractional_parts = sorted(
        split_names,
        key=lambda name: raw_counts[name] - counts[name],
        reverse=True,
    )

    for name in fractional_parts[:remaining]:
        counts[name] += 1

    return counts


def calculate_split_score(
    split_summary: dict[str, dict[str, int]],
    global_summary: dict[str, int],
    ratios: dict[str, float],
) -> float:
    """
    Calculate how far a candidate split is from the desired
    ROI and class distributions.

    Lower score = better split.
    """

    score = 0.0

    for split_name, ratio in ratios.items():

        current = split_summary[split_name]

        target_roi = global_summary["roi_count"] * ratio
        target_normal = global_summary["normal_count"] * ratio
        target_adenoma = global_summary["adenoma_count"] * ratio

        roi_error = (
            abs(current["roi_count"] - target_roi)
            / max(target_roi, 1)
        )

        normal_error = (
            abs(current["normal_count"] - target_normal)
            / max(target_normal, 1)
        )

        adenoma_error = (
            abs(current["adenoma_count"] - target_adenoma)
            / max(target_adenoma, 1)
        )

        score += (
            roi_error
            + normal_error
            + adenoma_error
        )

    return score


def generate_candidate_split(
    wsi_summary: pd.DataFrame,
    target_counts: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """
    Generate one candidate WSI-level split.

    WSIs are randomly shuffled and assigned while preserving
    the exact number of WSIs required for each split.
    """

    wsi_ids = wsi_summary["wsi_id"].to_numpy().copy()

    rng.shuffle(wsi_ids)

    train_end = target_counts["train"]
    val_end = train_end + target_counts["val"]

    assignments = {
        "train": wsi_ids[:train_end].tolist(),
        "val": wsi_ids[train_end:val_end].tolist(),
        "test": wsi_ids[val_end:].tolist(),
    }

    return assignments


def summarize_assignment(
    assignments: dict[str, list[str]],
    wsi_summary: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """
    Calculate ROI and label counts for each split.
    """

    result = {}

    for split_name, wsi_ids in assignments.items():

        subset = wsi_summary[
            wsi_summary["wsi_id"].isin(wsi_ids)
        ]

        result[split_name] = {
            "wsi_count": len(subset),
            "roi_count": int(subset["roi_count"].sum()),
            "normal_count": int(subset["normal_count"].sum()),
            "adenoma_count": int(subset["adenoma_count"].sum()),
        }

    return result


# ============================================================
# Main split search
# ============================================================

def find_balanced_split(
    metadata: pd.DataFrame,
    ratios: dict[str, float] = SPLIT_RATIOS,
    random_seed: int = RANDOM_SEED,
    num_trials: int = NUM_TRIALS,
):
    """
    Search for a WSI-level split that approximately balances:

    1. Number of WSIs
    2. Number of ROIs
    3. Number of Normal ROIs
    4. Number of Adenoma ROIs

    WSI counts are fixed exactly according to the desired ratio.
    """

    wsi_summary = get_wsi_summary(metadata)

    num_wsi = len(wsi_summary)

    target_counts = compute_target_wsi_counts(
        num_wsi=num_wsi,
        ratios=ratios,
    )

    global_summary = {
        "roi_count": int(wsi_summary["roi_count"].sum()),
        "normal_count": int(wsi_summary["normal_count"].sum()),
        "adenoma_count": int(wsi_summary["adenoma_count"].sum()),
    }

    print("=" * 70)
    print("Searching for balanced WSI-level split")
    print("=" * 70)

    print("\nTarget WSI counts:")
    for split_name, count in target_counts.items():
        print(
            f"{split_name:>5}: "
            f"{count} WSI "
            f"({count / num_wsi:.2%})"
        )

    rng = np.random.default_rng(random_seed)

    best_score = float("inf")
    best_assignment = None
    best_summary = None

    for trial in range(num_trials):

        assignment = generate_candidate_split(
            wsi_summary=wsi_summary,
            target_counts=target_counts,
            rng=rng,
        )

        summary = summarize_assignment(
            assignments=assignment,
            wsi_summary=wsi_summary,
        )

        score = calculate_split_score(
            split_summary=summary,
            global_summary=global_summary,
            ratios=ratios,
        )

        if score < best_score:
            best_score = score
            best_assignment = assignment
            best_summary = summary

    print(f"\nBest score after {num_trials:,} trials: {best_score:.6f}")

    return best_assignment, best_summary


# ============================================================
# Validation
# ============================================================

def validate_split(
    assignments: dict[str, list[str]],
    metadata: pd.DataFrame,
):
    """
    Verify that no WSI appears in more than one split and
    every WSI is assigned exactly once.
    """

    train = set(assignments["train"])
    val = set(assignments["val"])
    test = set(assignments["test"])

    if train & val:
        raise ValueError("WSI leakage detected between train and val.")

    if train & test:
        raise ValueError("WSI leakage detected between train and test.")

    if val & test:
        raise ValueError("WSI leakage detected between val and test.")

    all_assigned = train | val | test
    all_dataset = set(metadata["wsi_id"].unique())

    if all_assigned != all_dataset:

        missing = all_dataset - all_assigned
        extra = all_assigned - all_dataset

        raise ValueError(
            "Split does not match dataset WSIs.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )


# ============================================================
# Save splits
# ============================================================

def save_splits(
    metadata: pd.DataFrame,
    assignments: dict[str, list[str]],
    output_dir: Path = SPLIT_DIR,
):
    """
    Save:

        splits/
        ├── train.csv
        ├── val.csv
        ├── test.csv
        └── wsi_split.csv
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save ROI-level split files
    # --------------------------------------------------------

    for split_name, wsi_ids in assignments.items():

        split_df = metadata[
            metadata["wsi_id"].isin(wsi_ids)
        ].copy()

        split_df = split_df.sort_values(
            ["wsi_id", "x", "y"]
        )

        output_path = output_dir / f"{split_name}.csv"

        split_df.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Saved {split_name:>5}: "
            f"{len(split_df):,} ROI -> {output_path}"
        )

    # --------------------------------------------------------
    # Save WSI-level mapping
    # --------------------------------------------------------

    rows = []

    for split_name, wsi_ids in assignments.items():

        for wsi_id in sorted(wsi_ids):

            rows.append({
                "wsi_id": wsi_id,
                "split": split_name,
            })

    wsi_split_df = pd.DataFrame(rows)

    wsi_split_df = wsi_split_df.sort_values(
        ["split", "wsi_id"]
    )

    wsi_split_path = output_dir / "wsi_split.csv"

    wsi_split_df.to_csv(
        wsi_split_path,
        index=False,
    )

    print(
        f"Saved WSI mapping -> {wsi_split_path}"
    )


# ============================================================
# Reporting
# ============================================================

def print_split_report(
    summary: dict[str, dict[str, int]],
):
    """
    Print statistics of final split.
    """

    total_wsi = sum(
        item["wsi_count"]
        for item in summary.values()
    )

    total_roi = sum(
        item["roi_count"]
        for item in summary.values()
    )

    total_normal = sum(
        item["normal_count"]
        for item in summary.values()
    )

    total_adenoma = sum(
        item["adenoma_count"]
        for item in summary.values()
    )

    print("\n" + "=" * 70)
    print("Final Split")
    print("=" * 70)

    for split_name in ["train", "val", "test"]:

        s = summary[split_name]

        adenoma_ratio = (
            s["adenoma_count"]
            / s["roi_count"]
        )

        print(f"\n[{split_name.upper()}]")

        print(
            f"WSI     : {s['wsi_count']:4d} "
            f"({s['wsi_count'] / total_wsi:.2%})"
        )

        print(
            f"ROI     : {s['roi_count']:5d} "
            f"({s['roi_count'] / total_roi:.2%})"
        )

        print(
            f"Normal  : {s['normal_count']:5d} "
            f"({s['normal_count'] / total_normal:.2%})"
        )

        print(
            f"Adenoma : {s['adenoma_count']:5d} "
            f"({s['adenoma_count'] / total_adenoma:.2%})"
        )

        print(
            f"Adenoma ratio within split: "
            f"{adenoma_ratio:.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("Loading CAMEL metadata...")

    metadata = load_metadata()

    print(
        f"Loaded {len(metadata):,} ROI "
        f"from {metadata['wsi_id'].nunique()} WSIs."
    )

    assignments, summary = find_balanced_split(
        metadata=metadata,
    )

    validate_split(
        assignments=assignments,
        metadata=metadata,
    )

    print_split_report(summary)

    print("\nSaving split files...")

    save_splits(
        metadata=metadata,
        assignments=assignments,
    )

    print("\nSplit completed successfully.")


if __name__ == "__main__":
    main()
