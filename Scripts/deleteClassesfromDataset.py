import os
from tqdm import tqdm

def load_yolo_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    labels = [int(line.strip().split()[0]) for line in lines]
    return labels

def delete_files_with_class(image_folder, label_folder, class_ids_to_delete):
    image_files_deleted = 0
    label_files_deleted = 0
    
    # Get list of all label files
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    
    for label_file in tqdm(label_files, desc="Processing labels"):
        label_path = os.path.join(label_folder, label_file)
        labels = load_yolo_labels(label_path)
        
        # Check if any of the specified class IDs are in the labels
        if any(class_id in labels for class_id in class_ids_to_delete):
            # Delete the label file
            os.remove(label_path)
            label_files_deleted += 1
            
            # Corresponding image file
            image_file = os.path.join(image_folder, label_file.replace('.txt', '.jpg').replace('.txt', '.jpeg').replace('.txt', '.png'))
            if os.path.exists(image_file):
                os.remove(image_file)
                image_files_deleted += 1
    
    return image_files_deleted, label_files_deleted

# Example usage
image_folder = 'D:\git\PestVisionChallenge\synthdata\images'
label_folder = 'D:\git\PestVisionChallenge\synthdata\labels'
class_ids_to_delete = [61, 94, 95]

image_files_deleted, label_files_deleted = delete_files_with_class(image_folder, label_folder, class_ids_to_delete)

print(f"Number of image files deleted: {image_files_deleted}")
print(f"Number of label files deleted: {label_files_deleted}")
