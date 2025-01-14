import os
import shutil
import pandas as pd

# Paths
csv_file = "train.csv"
train_folder = "Train"
output_folder = "Organized_Train"

# Create output folder structure
classes = ["OLD", "YOUNG", "MIDDLE"]
for cls in classes:
    os.makedirs(os.path.join(output_folder, cls), exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Move files based on Class
for index, row in df.iterrows():
    img_id = row['ID']
    img_class = row['Class']
    
    # Source and destination paths
    source_path = os.path.join(train_folder, img_id)
    destination_path = os.path.join(output_folder, img_class, img_id)
    
    # Move the file
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
    else:
        print(f"File not found: {source_path}")

print("Images have been organized!")
