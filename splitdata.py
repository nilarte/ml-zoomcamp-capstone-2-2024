import os
import shutil
import random

# Paths
train_dir = "data/train"
test_dir = "data/test"

# Create the test directory structure
os.makedirs(test_dir, exist_ok=True)
for category in ["old", "young"]:
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Function to split data
def split_data(source_dir, target_dir, test_ratio=0.25):
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        # Get all files in the category
        files = os.listdir(category_path)
        random.shuffle(files)  # Shuffle files for random selection
        
        # Split files
        test_size = int(len(files) * test_ratio)
        test_files = files[:test_size]
        
        # Move test files to the test directory
        for file_name in test_files:
            src_file = os.path.join(category_path, file_name)
            dest_file = os.path.join(target_dir, category, file_name)
            shutil.move(src_file, dest_file)

# Perform the split
split_data(train_dir, test_dir, test_ratio=0.25)

print("Data split completed!")
