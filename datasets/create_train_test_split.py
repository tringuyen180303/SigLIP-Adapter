import os
import csv
import random
from sklearn.model_selection import train_test_split

def generate_caltech101_csv(dataset_root, output_dir, test_ratio=0.2, seed=42):
    image_dir = os.path.join(dataset_root, '101_ObjectCategories')
    data = []

    # Collect all images and class info
    for class_idx, class_name in enumerate(sorted(os.listdir(image_dir))):
        class_path = os.path.join(image_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_file in os.listdir(class_path):
            if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                impath = os.path.join(class_path, img_file)
                data.append((impath, class_idx, class_name))

    # Shuffle and split
    random.seed(seed)
    train_data, test_data = train_test_split(data, test_size=test_ratio, stratify=[x[2] for x in data], random_state=seed)

    # Write CSV files
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in [('train', train_data), ('test', test_data)]:
        with open(os.path.join(output_dir, f'{split_name}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['impath', 'label', 'classname'])
            for row in split_data:
                writer.writerow(row)

    print(f"CSV files created in: {output_dir}")
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

# Example usage
generate_caltech101_csv("../data/caltech-101", "../data/caltech-101/")
