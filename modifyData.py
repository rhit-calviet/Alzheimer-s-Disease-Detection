"""
Dataset Restructuring Script for Alzheimer’s Disease MRI Classification

This script prepares an Alzheimer’s MRI dataset by combining original and augmented images,
splitting them into training, validation, and testing sets, and organizing them into structured directories.

Steps:
1. **Define Dataset Paths**:
   - `OriginalDataset`: Contains the original images for each category.
   - `AugmentedAlzheimerDataset`: Contains augmented images for each category.
   - `TRAIN`, `VAL`, `TEST`: Destination folders for the split dataset.

2. **Categories**:
   - Four categories are supported: `MildDemented`, `ModerateDemented`, `NonDemented`, `VeryMildDemented`.

3. **Directory Creation**:
   - Creates subdirectories for each category under `TRAIN`, `VAL`, and `TEST`.

4. **Data Combination and Splitting**:
   - Combines original and augmented images for each category.
   - Splits the dataset into:
     - **TRAIN**: 75%
     - **VAL**: 5%
     - **TEST**: 20%
   - Random splits are controlled using a fixed random seed for reproducibility.

5. **Image Organization**:
   - Copies images into their respective directories based on the split.

6. **Cleanup**:
   - Removes the `OriginalDataset` and `AugmentedAlzheimerDataset` directories after restructuring.

Usage:
- Ensure the `OriginalDataset` and `AugmentedAlzheimerDataset` directories exist under the specified `DATA` base path.
- Run the script to organize and split the dataset for deep learning experiments. Thedataset is sourced from Kaggle:
https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data.

Output:
- A structured dataset under `TRAIN`, `VAL`, and `TEST` directories, ready for model training and evaluation.
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# Base Path
data_path = r'..\DATA'
print('Found Path')

# Dataset paths
original_dataset_path = os.path.join(data_path, 'OriginalDataset')
augmented_dataset_path = os.path.join(data_path, 'AugmentedAlzheimerDataset')

# New folders for split
train_path = os.path.join(data_path, 'TRAIN')
val_path = os.path.join(data_path, 'VAL')
test_path = os.path.join(data_path, 'TEST')

# Categories
categories = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Create the TRAIN, VAL, and TEST folders
for split in [train_path, val_path, test_path]:
    for category in categories:
        os.makedirs(os.path.join(split, category), exist_ok=True)

# Function to combine and split data
def combine_and_split_data():
    for category in categories:
        print('loop')
        # Collect all images from original and augmented datasets
        original_images = [
            os.path.join(original_dataset_path, category, img)
            for img in os.listdir(os.path.join(original_dataset_path, category))
            if os.path.isfile(os.path.join(original_dataset_path, category, img))
        ]
        augmented_images = [
            os.path.join(augmented_dataset_path, category, img)
            for img in os.listdir(os.path.join(augmented_dataset_path, category))
            if os.path.isfile(os.path.join(augmented_dataset_path, category, img))
        ]
        all_images = original_images + augmented_images

        # Split the dataset into TRAIN (75%), VAL (5%), TEST (20%)
        train_images, temp_images = train_test_split(all_images, test_size=0.25, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=0.8, random_state=42)

        # Move images into their respective directories
        for img in train_images:
            shutil.copy(img, os.path.join(train_path, category))
        for img in val_images:
            shutil.copy(img, os.path.join(val_path, category))
        for img in test_images:
            shutil.copy(img, os.path.join(test_path, category))

# Run the function
combine_and_split_data()

# Remove the original folders
shutil.rmtree(original_dataset_path)
shutil.rmtree(augmented_dataset_path)

print("Dataset restructuring complete! Original folders removed.")
