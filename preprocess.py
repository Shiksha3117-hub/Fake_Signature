import os
import cv2
import random
import shutil

BASE_PATH = "dataset"
PROCESSED_PATH = "processed"
TRAIN_PATH = "train"
TEST_PATH = "test"

IMG_SIZE = 128

def preprocess_image(img_path):
    img = cv2.imread(img_path)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return img


# Create required folders
for folder in [
    "processed/genuine", "processed/forged",
    "train/genuine", "train/forged",
    "test/genuine", "test/forged"
]:
    os.makedirs(folder, exist_ok=True)


# Preprocess and save
for category in ["genuine", "forged"]:
    category_path = os.path.join(BASE_PATH, category)
    files = os.listdir(category_path)

    for file in files:
        img_path = os.path.join(category_path, file)
        img = preprocess_image(img_path)

        save_path = os.path.join(PROCESSED_PATH, category, file)
        cv2.imwrite(save_path, img)

print("✅ Preprocessing complete.")


# Train/Test Split
for category in ["genuine", "forged"]:
    category_path = os.path.join(PROCESSED_PATH, category)
    files = os.listdir(category_path)

    random.shuffle(files)
    split = int(0.8 * len(files))

    train_files = files[:split]
    test_files = files[split:]

    for file in train_files:
        shutil.copy(
            os.path.join(category_path, file),
            os.path.join(TRAIN_PATH, category, file)
        )

    for file in test_files:
        shutil.copy(
            os.path.join(category_path, file),
            os.path.join(TEST_PATH, category, file)
        )

print("✅ Train/Test split done.")
