import os
import random
import cv2
import numpy as np
from tqdm import tqdm

def load_yolo_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        labels.append((class_id, x_center, y_center, width, height))
    return labels

def annotate_image(image_file, labels, output_file):
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
        
    cv2.imwrite(output_file, image)

def process_images(image_folder, label_folder, output_folder, num_images):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get a list of all images in the folder
    all_images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Randomly select x images
    selected_images = random.sample(all_images, num_images)
    
    for image_file in tqdm(selected_images, desc="Annotating images"):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        output_path = os.path.join(output_folder, image_file)
        
        if os.path.exists(label_path):
            labels = load_yolo_labels(label_path)
            annotate_image(image_path, labels, output_path)

# Example usage
image_folder = "C:/Users/jesli/Downloads/PestVisionEvalImages/Images"
label_folder = 'C:/Users/jesli/Downloads/prediction'
output_folder = 'aaaa'
num_images = 1337

process_images(image_folder, label_folder, output_folder, num_images)
