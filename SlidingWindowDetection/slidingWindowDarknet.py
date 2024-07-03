import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(draw, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    label = str(classes[class_id])
    color = (0, 0, 255)  # Red color in BGR
    draw.rectangle([x, y, x_plus_w, y_plus_h], outline=color, width=2)
    draw.text((x, y), f"{label} {confidence:.2f}", fill=color)

def process_segments(image, net, segment_size, classes):
    height, width, _ = image.shape
    results_list = []

    for y in range(0, height, segment_size):
        for x in range(0, width, segment_size):
            # Extract the segment
            segment = image[y:y+segment_size, x:x+segment_size]

            # Pad the segment if it's at the edge
            padded_segment = np.zeros((segment_size, segment_size, 3), dtype=np.uint8)
            padded_segment[:segment.shape[0], :segment.shape[1]] = segment

            # Create a blob from the segment
            blob = cv2.dnn.blobFromImage(padded_segment, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(get_output_layers(net))

            results_list.append((outs, x, y))

    return results_list

def annotate_image(image, results_list, segment_size, classes):
    draw = ImageDraw.Draw(image)

    for outs, x_offset, y_offset in results_list:
        for detection in outs:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(obj[0] * segment_size) + x_offset
                    center_y = int(obj[1] * segment_size) + y_offset
                    w = int(obj[2] * segment_size)
                    h = int(obj[3] * segment_size)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    draw_prediction(draw, class_id, confidence, x, y, x + w, y + h, classes)

    return image

def process_images_from_directory(directory_path, weights_path, config_path, names_path, output_dir, x, segment_size=640):
    # Load the Darknet model
    net = cv2.dnn.readNet(weights_path, config_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Load class names
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the first x image filenames
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:x]

    for filename in image_files:
        image_path = os.path.join(directory_path, filename)
        output_path = os.path.join(output_dir, filename)

        # Load the high-resolution image
        image = cv2.imread(image_path)

        # Process image segments
        results_list = process_segments(image, net, segment_size, classes)

        # Create a PIL image for annotation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Annotate the image
        annotated_image = annotate_image(pil_image, results_list, segment_size, classes)

        # Save the final annotated image
        annotated_image.save(output_path)
        print(f"Annotated image saved to {output_path}")

# Define paths and parameters
directory_path = "D:/git/New folder/data/val/images"
weights_path = "D:/git/Darknet/yolov4/training/yolov4-custom_last.weights"
config_path = "D:/git/Darknet/yolov4/darknet/cfg/yolov4-custom.cfg"
names_path = "D:/git/Darknet/yolov4/darknet/data/obj.names"
output_dir = "annotated_output_images_Darknet"
x = 50 # Number of images to process

# Run the main function
process_images_from_directory(directory_path, weights_path, config_path, names_path, output_dir, x)
