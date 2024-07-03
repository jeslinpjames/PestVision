import os
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

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

def process_dataset(root_folder):
    image_folder = os.path.join(root_folder, 'images/train')
    label_folder = os.path.join(root_folder, 'labels/train')

    class_images = defaultdict(list)

    # Collect images for each class
    for label_file in tqdm(os.listdir(label_folder), desc="Processing labels"):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_folder, label_file)
            labels = load_yolo_labels(label_path)
            for label in labels:
                class_id = label[0]
                image_file = os.path.join(image_folder, label_file.replace('.txt', '.jpg').replace('.txt', '.jpeg').replace('.txt', '.png'))
                if os.path.exists(image_file):
                    class_images[class_id].append((image_file, labels))

    # Display 10 images for each class
    for class_id in range(102):
        if class_id in class_images and len(class_images[class_id]) > 0:
            print(f"Displaying 10 images for class {class_id}")
            images_to_display = random.sample(class_images[class_id], min(10, len(class_images[class_id])))
            display_images(images_to_display)

# Example usage
root_folder = 'D:\\git\\PestVisionChallenge\\Detection_IP102'
process_dataset(root_folder)
