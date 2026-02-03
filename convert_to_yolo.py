"""
BDD100K to YOLO Label Format Converter

Converts BDD100K JSON labels to YOLO TXT format.
YOLO format: <class_id> <x_center> <y_center> <width> <height>
All values normalized to [0, 1].
"""

import json
import os
from pathlib import Path

# Configuration
BASE_DIR = Path(r"c:\My Projects\road-object-detection")
# Labels are already in dataset/labels folder (JSON format)
BDD_LABELS_DIR = BASE_DIR / "dataset" / "labels"
# Output to the same location (converting JSON to TXT)
YOLO_LABELS_DIR = BASE_DIR / "dataset" / "labels"

# Image dimensions (all BDD100K images are 1280x720)
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# Detection categories (12 classes for YOLO11)
# Traffic lights are split by color
# We exclude lane/area categories as they're not for object detection
DETECTION_CATEGORIES = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'person': 3,
    'rider': 4,
    'bike': 5,
    'motor': 6,
    'traffic light green': 7,
    'traffic light red': 8,
    'traffic light yellow': 9,
    'traffic sign': 10,
    'train': 11,
}

# Note: 'trailer' is in detection_categories list but has 0 samples
# Could be added if needed: 'trailer': 10


def convert_bbox_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """
    Convert BDD100K bbox format (x1, y1, x2, y2) to YOLO format.

    Args:
        x1, y1, x2, y2: Bounding box coordinates
        img_width, img_height: Image dimensions

    Returns:
        x_center, y_center, width, height (normalized to [0, 1])
    """
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    # Clip to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def convert_single_label(json_path, output_path):
    """
    Convert a single BDD100K JSON label file to YOLO TXT format.

    Args:
        json_path: Path to input JSON file
        output_path: Path to output TXT file

    Returns:
        Number of objects written, or -1 if error
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Get image ID from filename
        img_id = json_path.stem

        # Extract objects from frame
        objects = []
        if 'frames' in data and len(data['frames']) > 0:
            for obj in data['frames'][0].get('objects', []):
                category = obj.get('category')

                # Handle traffic light sub-categories by color
                if category == 'traffic light':
                    attrs = obj.get('attributes', {})
                    color = attrs.get('trafficLightColor', 'none')
                    # Skip traffic lights with no visible color
                    if color == 'none':
                        continue
                    category = f'traffic light {color}'

                # Only process detection categories
                if category not in DETECTION_CATEGORIES:
                    continue

                # Skip objects without box2d
                if 'box2d' not in obj:
                    continue

                # Get bounding box
                box = obj['box2d']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Convert to YOLO format
                class_id = DETECTION_CATEGORIES[category]
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    x1, y1, x2, y2, IMG_WIDTH, IMG_HEIGHT
                )

                objects.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Write to file
        if objects:
            with open(output_path, 'w') as f:
                f.write('\n'.join(objects))
        else:
            # Create empty file for images with no detection objects
            with open(output_path, 'w') as f:
                f.write('')

        return len(objects)

    except Exception as e:
        print(f"Error converting {json_path}: {e}")
        return -1


def convert_split(split_name):
    """
    Convert all labels for a given split (train, val, test).

    Args:
        split_name: One of 'train', 'val', 'test'
    """
    print(f"\n{'='*60}")
    print(f"Converting {split_name.upper()} split")
    print(f"{'='*60}")

    # Paths
    input_dir = BDD_LABELS_DIR / split_name
    output_dir = YOLO_LABELS_DIR / split_name

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    json_files = list(input_dir.glob("*.json"))
    print(f"Found {len(json_files):,} label files")

    # Convert each file
    stats = {
        'processed': 0,
        'objects': 0,
        'errors': 0,
        'empty': 0
    }

    for i, json_path in enumerate(json_files):
        # Progress indicator every 10000 files
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,} / {len(json_files):,} files...")

        # Output file has same name but .txt extension
        output_path = output_dir / f"{json_path.stem}.txt"

        num_objects = convert_single_label(json_path, output_path)

        if num_objects >= 0:
            stats['processed'] += 1
            stats['objects'] += num_objects
            if num_objects == 0:
                stats['empty'] += 1
        else:
            stats['errors'] += 1

    # Print statistics
    print(f"\n{split_name.upper()} Conversion Summary:")
    print(f"  Processed: {stats['processed']:,} files")
    print(f"  Total objects: {stats['objects']:,}")
    print(f"  Empty labels: {stats['empty']:,}")
    print(f"  Errors: {stats['errors']}")

    return stats


def main():
    """Convert all splits."""
    print("="*60)
    print("BDD100K to YOLO Label Format Converter")
    print("="*60)
    print(f"\nInput directory: {BDD_LABELS_DIR}")
    print(f"Output directory: {YOLO_LABELS_DIR}")
    print(f"\nDetection categories ({len(DETECTION_CATEGORIES)}):")
    for name, idx in sorted(DETECTION_CATEGORIES.items(), key=lambda x: x[1]):
        print(f"  {idx}: {name}")

    # Convert each split
    splits = ['train', 'val', 'test']
    total_stats = {
        'processed': 0,
        'objects': 0,
        'errors': 0
    }

    for split in splits:
        stats = convert_split(split)
        total_stats['processed'] += stats['processed']
        total_stats['objects'] += stats['objects']
        total_stats['errors'] += stats['errors']

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {total_stats['processed']:,}")
    print(f"Total objects written: {total_stats['objects']:,}")
    print(f"Total errors: {total_stats['errors']}")
    print(f"\nLabels saved to: {YOLO_LABELS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
