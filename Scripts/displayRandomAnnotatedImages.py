import os
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_yolo_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append((class_id, x_center, y_center, width, height))
    return labels

def annotate_image(image_file, labels):
    image = cv2.imread(image_file)
    height, width, _ = image.shape
    
    for label in labels:
        class_id, x_center, y_center, bbox_width, bbox_height = label
        xmin = int((x_center - bbox_width / 2) * width)
        ymin = int((y_center - bbox_height / 2) * height)
        xmax = int((x_center + bbox_width / 2) * width)
        ymax = int((y_center + bbox_height / 2) * height)
        
        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2) # Red color in BGR
        
        # Put class ID text near the bounding box
        cv2.putText(image, str(class_id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return image

def display_images(images):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, (image_file, labels) in zip(axes.flatten(), images):
        image = annotate_image(image_file, labels)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(os.path.basename(image_file))
        ax.axis('off')
    plt.show()

def process_dataset(image_folder, label_folder, num_times):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    total_images = len(image_files)
    num_images = min(10, total_images)
    
    step_size = max(total_images // num_times, num_images)
    
    for i in range(num_times):
        images_to_display = []
        start_idx = (i * step_size) % total_images
        end_idx = (start_idx + num_images) % total_images
        if end_idx > start_idx:
            selected_files = image_files[start_idx:end_idx]
        else:
            selected_files = image_files[start_idx:] + image_files[:end_idx]

        for image_file in selected_files:
            image_path = os.path.join(image_folder, image_file)
            label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            if os.path.exists(label_path):
                labels = load_yolo_labels(label_path)
                images_to_display.append((image_path, labels))
        
        display_images(images_to_display)

# Example usage
image_folder = 'D:\git\PestVisionChallenge\synthdata\images'
label_folder = 'D:\git\PestVisionChallenge\synthdata\labels'
num_times = 10  # Number of times to display 10 images
process_dataset(image_folder, label_folder, num_times)
