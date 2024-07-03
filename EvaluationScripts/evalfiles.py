import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

def annotate_images(input_dir, output_dir, model_path):
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all images in the input directory
    images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

    # Iterate over all images in the input directory with progress tracking
    for filename in tqdm(images, desc="Processing images"):
        # Construct the full path to the image file
        img_path = os.path.join(input_dir, filename)

        # Read the image
        img = cv2.imread(img_path)

        # Make predictions using the YOLO model on the mirrored image
        results = model(img)

        # Draw bounding boxes on the image
        annotated_img = results[0].plot()

        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))

        # Save the annotated image to the output directory
        output_path = os.path.join(output_dir, filename)
        pil_img.save(output_path)

        print(f"Annotated and saved: {output_path}")

# Define input and output directories and model path
input_dir = r"D:/git/New folder/data/val/images"
output_dir = r"D:/git/PestVisionChallenge/results"
model_path = r"C:/Users/jesli/Downloads/train8/weights/best.pt"

# Annotate all images in the input directory and save to output directory
annotate_images(input_dir, output_dir, model_path)
