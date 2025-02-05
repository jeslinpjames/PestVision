import os
import shutil

def organize_images_by_class(image_folder, text_file, output_folder, class_range=(0, 13)):
    """
    Organizes images into separate folders based on class labels.
    
    :param image_folder: Path to the folder containing images.
    :param text_file: Path to the text file containing image names and class labels.
    :param output_folder: Path to the destination folder where images will be sorted.
    :param class_range: Tuple indicating the range of class labels to include.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the text file and process images
    with open(text_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # Skip invalid lines
            
            img_name, class_label = parts
            class_label = int(class_label)
            
            # Only process classes within the specified range
            if class_range[0] <= class_label <= class_range[1]:
                class_folder = os.path.join(output_folder, f'class_{class_label}')
                os.makedirs(class_folder, exist_ok=True)
                
                src_path = os.path.join(image_folder, img_name)
                dest_path = os.path.join(class_folder, img_name)
                
                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                    print(f'Copied: {img_name} -> {class_folder}')
                else:
                    print(f'Warning: {src_path} not found.')

# Example usage
image_folder = r"C:/Users/jesli/Downloads/ip102_v1.1-001/ip102_v1.1/images"
# text_files = [r"C:/path/to/val.txt", r"C:/path/to/test.txt", r"C:/path/to/train.txt"]
output_folder = r"C:/Users/jesli/Downloads/sorted_images"
text_file="C:/Users/jesli/Downloads/ip102_v1.1-001/ip102_v1.1/val.txt"
organize_images_by_class(image_folder, text_file, output_folder)
# for text_file in text_files:
#     organize_images_by_class(image_folder, text_file, output_folder)
