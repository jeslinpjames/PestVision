import os
import cv2
import numpy as np

# Paths
weights_path = "D:\\git\\Darknet\\yolov4\\trainingwith eval\\yolov4-custom_last.weights"
config_path = "D:\\git\\Darknet\\yolov4\\1 classyolov4-custom.cfg"
names_path = "D:\\git\\Darknet\\yolov4\\obj.names"
input_folder = "D:\\git\\New folder\\data\\val\\images"
output_folder = "D:\\git\\New folder\\data\\val\\detectionswitheval1class"

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)

# Get the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get list of all images in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Function to process each image
def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Save detections in YOLO format
    with open(output_path, 'w') as f:
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

# Iterate over all images and run detection
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    output_txt_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.txt')
    process_image(image_path, output_txt_path)

print("Object detection complete. Results saved in:", output_folder)
