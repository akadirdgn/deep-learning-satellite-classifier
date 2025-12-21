import os
import shutil
import random
from configs.settings import DATA_DIR

# ==========================================
# CONFIGURATION
# ==========================================

# Source directory for raw data (EuroSAT)
# This was hardcoded in the original script. 
# Ideal way is to pass it as argument, but for now we keep it here or from config if added.
SOURCE_DIR = r"C:\Users\kadir\Downloads\EuroSAT\2750"

# Target directory is defined in configs/settings.py as DATA_DIR
TARGET_DIR = DATA_DIR

# Split Ratios
# 70% Train, 15% Valid, 15% Test
SPLIT_RATIOS = (0.7, 0.15, 0.15)

def create_folders():
    """Create the directory structure for train, valid, and test sets."""
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return []

    # Get class names from source directory (e.g., Forest, River)
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    # Create directories: data/train/Forest, data/test/River etc.
    for split in ['train', 'valid', 'test']:
        for class_name in classes:
            os.makedirs(os.path.join(TARGET_DIR, split, class_name), exist_ok=True)
    return classes

def distribute_data(classes):
    """Distribute images from source to target folders based on split ratios."""
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {os.path.abspath(TARGET_DIR)}")
    print("-" * 50)

    total_images = 0

    for class_name in classes:
        current_source_path = os.path.join(SOURCE_DIR, class_name)
        images = os.listdir(current_source_path)

        # Shuffle to ensure random distribution
        random.shuffle(images)

        total = len(images)
        train_end = int(total * SPLIT_RATIOS[0])
        valid_end = train_end + int(total * SPLIT_RATIOS[1])

        train_files = images[:train_end]
        valid_files = images[train_end:valid_end]
        test_files = images[valid_end:]

        # Helper function to copy files
        def copy_files(file_list, split_name):
            for file_name in file_list:
                src = os.path.join(current_source_path, file_name)
                dst = os.path.join(TARGET_DIR, split_name, class_name, file_name)
                shutil.copy(src, dst)

        copy_files(train_files, 'train')
        copy_files(valid_files, 'valid')
        copy_files(test_files, 'test')

        print(f">> {class_name:<20} -> Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
        total_images += total

    print("-" * 50)
    print(f"TOTAL PROCESSED IMAGES: {total_images}")

def prepare():
    """Main execution function for data preparation."""
    classes = create_folders()
    if classes:
        distribute_data(classes)
        print(f"\n>> DONE! Data prepared in '{TARGET_DIR}' folder.")

if __name__ == "__main__":
    prepare()
