"""
BDD100K Dataset Reduction Script

Randomly removes excess training/validation/test images and labels
to optimize training time while maintaining 70:10:20 ratio.

Original: 100,000 images (70K train, 10K val, 20K test)
Target:   85,714 images (60K train, ~8,571 val, ~17,143 test)
"""

import os
import random
from pathlib import Path
from collections import Counter

# Configuration
BASE_DIR = Path(r"c:\My Projects\road-object-detection")
DATASET_DIR = BASE_DIR / "dataset"

# Target sizes (maintaining 70:10:20 ratio)
TARGET_SIZES = {
    'train': 60000,      # 70,000 → 60,000
    'val': 8571,        # 10,000 → 8,571
    'test': 17143,       # 20,000 → 17,143
}

# Random seed for reproducibility
RANDOM_SEED = 42


def get_current_counts():
    """Count current files in each split."""
    counts = {}
    for split in ['train', 'val', 'test']:
        img_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split
        img_count = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        lbl_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        counts[split] = {'images': img_count, 'labels': lbl_count}
    return counts


def reduce_split(split_name, target_size):
    """
    Reduce a split to target size by randomly removing files.
    Removes both images and labels in sync.
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")

    img_dir = DATASET_DIR / "images" / split_name
    lbl_dir = DATASET_DIR / "labels" / split_name

    # Get current files
    img_files = list(img_dir.glob("*.jpg"))
    current_count = len(img_files)

    print(f"Current images: {current_count:,}")
    print(f"Target: {target_size:,}")
    print(f"Need to remove: {current_count - target_size:,}")

    if current_count <= target_size:
        print(f"✓ Already at or below target size. Skipping.")
        return 0

    # Randomly select files to keep
    random.seed(RANDOM_SEED)
    keep_files = set(random.sample(img_files, target_size))
    remove_files = set(img_files) - keep_files

    # Remove files
    removed_count = 0
    for img_path in remove_files:
        # Remove image
        img_path.unlink()
        removed_count += 1

        # Remove corresponding label (both .txt and .json)
        label_txt = lbl_dir / f"{img_path.stem}.txt"
        label_json = lbl_dir / f"{img_path.stem}.json"
        if label_txt.exists():
            label_txt.unlink()
        if label_json.exists():
            label_json.unlink()

    print(f"✓ Removed {removed_count:,} image/label pairs")

    # Verify
    final_img_count = len(list(img_dir.glob("*.jpg")))
    final_lbl_count = len(list(lbl_dir.glob("*.txt")))
    print(f"Final images: {final_img_count:,}")
    print(f"Final labels: {final_lbl_count:,}")

    if final_img_count != target_size or final_lbl_count != target_size:
        print(f"⚠ Warning: Final count ({final_img_count:,}) doesn't match target ({target_size:,})")

    return removed_count


def create_removed_files_backup(removed_files_dict):
    """Create a backup file listing all removed filenames."""
    backup_path = BASE_DIR / "removed_files.txt"

    with open(backup_path, 'w') as f:
        for split, files in removed_files_dict.items():
            if files:
                f.write(f"# {split.upper()}\n")
                for file_path in sorted(files):
                    f.write(f"{file_path.stem}\n")
                f.write("\n")

    print(f"\n✓ Created backup: {backup_path}")


def verify_integrity():
    """Verify that all images have corresponding labels and vice versa."""
    print(f"\n{'='*60}")
    print("INTEGRITY CHECK")
    print(f"{'='*60}")

    all_good = True
    for split in ['train', 'val', 'test']:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split

        # Get all filenames (without extension)
        if img_dir.exists() and lbl_dir.exists():
            img_names = {f.stem for f in img_dir.glob("*.jpg")}
            lbl_names = {f.stem for f in lbl_dir.glob("*.txt")}

            # Check for missing labels
            missing_labels = img_names - lbl_names
            missing_images = lbl_names - img_names

            if missing_labels:
                print(f"⚠️  {split.upper()}: {len(missing_labels):,} images without labels")
                all_good = False

            if missing_images:
                print(f"⚠️  {split.upper()}: {len(missing_images):,} labels without images")
                all_good = False

    if all_good:
        print("✓ All images have corresponding labels and vice versa.")


def main():
    """Main execution function."""
    print("="*60)
    print("BDD100K Dataset Reduction Script")
    print("="*60)
    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"\nTarget sizes (70:10:20 ratio):")
    for split, size in TARGET_SIZES.items():
        print(f"  {split:6s}: {size:>8,} images")

    # Show current state
    print(f"\n{'='*60}")
    print("CURRENT STATE")
    print(f"{'='*60}")
    current_counts = get_current_counts()
    for split, counts in current_counts.items():
        print(f"  {split:6s}: {counts['images']:>8,} images, {counts['labels']:>8,} labels")

    # Get removed files for backup (before we start deleting)
    removed_files = {}
    for split in ['train', 'val', 'test']:
        target = TARGET_SIZES[split]
        img_dir = DATASET_DIR / "images" / split
        img_files = list(img_dir.glob("*.jpg"))

        if len(img_files) > target:
            random.seed(RANDOM_SEED)
            keep = set(random.sample(img_files, target))
            remove = set(img_files) - keep
            removed_files[split] = remove

    # Ask for confirmation
    total_remove = sum(len(files) for files in removed_files.values())
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total files to remove: {total_remove:,}")
    for split, files in removed_files.items():
        print(f"  {split:6s}: {len(files):>8,}")

    # Check for user to proceed
    # For now, we'll proceed automatically since this is part of the approved plan

    # Execute reduction
    total_removed = 0
    for split in ['train', 'val', 'test']:
        removed = reduce_split(split, TARGET_SIZES[split])
        total_removed += removed

    # Create backup of removed files
    create_removed_files_backup(removed_files)

    # Integrity check
    verify_integrity()

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total files removed: {total_removed:,}")

    print(f"\n{'='*60}")
    print("NEW DATASET SIZES")
    print(f"{'='*60}")
    final_counts = get_current_counts()
    for split, counts in final_counts.items():
        print(f"  {split:6s}: {counts['images']:>8,} images, {counts['labels']:>8,} labels")

    total_images = sum(c['images'] for c in final_counts.values())
    print(f"\nTotal dataset: {total_images:,} images")

    # Check how close we are to targets
    print(f"\n{'='*60}")
    print("TARGET vs ACTUAL COMPARISON")
    print(f"{'='*60}")
    for split, counts in final_counts.items():
        target = TARGET_SIZES[split]
        actual = counts['images']
        diff = actual - target
        status = "✓" if diff == 0 else ("+" if diff > 0 else "") + f"{diff:,}"
        print(f"  {split:6s}: Target={target:,}, Actual={actual:,}, Diff={status}")


if __name__ == "__main__":
    main()
