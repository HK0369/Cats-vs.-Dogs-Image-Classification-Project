import os
import shutil
import random
from pathlib import Path

# Paths
RAW_TRAIN_DIR = Path("data/raw/training_set/training_set")
RAW_TEST_DIR = Path("data/raw/test_set/test_set")
PROCESSED_DIR = Path("data/processed")

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

# Split ratio
VAL_SPLIT = 0.2
random.seed(42)


def prepare_directories():
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for category in ["cats", "dogs"]:
            os.makedirs(split / category, exist_ok=True)


def split_train_val():
    print("📂 Splitting training data into train/val sets...")

    for category in ["cats", "dogs"]:
        category_path = RAW_TRAIN_DIR / category
        images = os.listdir(category_path)
        random.shuffle(images)

        split_idx = int(len(images) * (1 - VAL_SPLIT))
        train_files = images[:split_idx]
        val_files = images[split_idx:]

        # Copy train files
        for img in train_files:
            shutil.copy(category_path / img, TRAIN_DIR / category)

        # Copy val files
        for img in val_files:
            shutil.copy(category_path / img, VAL_DIR / category)

    print("✅ Train/Val split completed.")


def prepare_test_set():
    print("📂 Copying test data...")

    for category in ["cats", "dogs"]:
        category_path = RAW_TEST_DIR / category
        images = os.listdir(category_path)

        for img in images:
            shutil.copy(category_path / img, TEST_DIR / category)

    print("✅ Test set copied.")


if __name__ == "__main__":
    prepare_directories()
    split_train_val()
    prepare_test_set()
    print("🎯 Data preparation completed.")
