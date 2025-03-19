import os
import shutil
import random

def split_dataset(image_dir, label_dir, dest_dir, split_ratio=0.9, seed=42):
    # Ensure directories exist
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print("Image or label directory does not exist!")
        return

    # Create destination directories
    train_image_dest = os.path.join(dest_dir, "train", "images")
    train_label_dest = os.path.join(dest_dir, "train", "labels")
    val_image_dest = os.path.join(dest_dir, "val", "images")
    val_label_dest = os.path.join(dest_dir, "val", "labels")

    for dir_path in [train_image_dest, train_label_dest, val_image_dest, val_label_dest]:
        os.makedirs(dir_path, exist_ok=True)

    # List all the image and label files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(label_dir) if f.replace('.txt', '.jpg') in image_files]

    # Ensure they match 1-to-1
    assert len(image_files) == len(label_files)

    # Set random seed
    random.seed(seed)
    random.shuffle(image_files)  # Shuffle images and use the same order for labels

    # Split the dataset
    train_size = int(len(image_files) * split_ratio)
    train_image_files = image_files[:train_size]
    val_image_files = image_files[train_size:]

    # Copy train files
    for image_file in train_image_files:
        label_file = image_file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(train_image_dest, image_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_dest, label_file))

    # Copy val files
    for image_file in val_image_files:
        label_file = image_file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(val_image_dest, image_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(val_label_dest, label_file))


if __name__ == "__main__":
    IMAGE_DIR = r'...'
    LABEL_DIR = r'...'
    DEST_DIR = r'...'
    SPLIT_RATIO = 0.9  # 90% train, 10% validation
    SEED = 42

    split_dataset(IMAGE_DIR, LABEL_DIR, DEST_DIR, SPLIT_RATIO, SEED)
