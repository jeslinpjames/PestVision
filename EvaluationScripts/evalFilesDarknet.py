import cv2
import numpy as np
import os
from tqdm import tqdm

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    label = str(classes[class_id])
    color = (0, 0, 255)  # Red color in BGR
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, f"{label} {confidence:.2f}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def annotate_images(input_dir, output_dir, weights_path, config_path, names_path):
    # Load the Darknet model
    net = cv2.dnn.readNet(weights_path, config_path)
    
    # Set preferable backend and target to CUDA
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Load class names
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all images in the input directory
    images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    
    # Iterate over all images in the input directory with progress tracking
    for filename in tqdm(images, desc="Processing images"):
        # Construct the full path to the image file
        img_path = os.path.join(input_dir, filename)
        
        # Read the image
        image = cv2.imread(img_path)
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 1/255
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set the blob as input to the network
        net.setInput(blob)
        
        # Run inference
        outs = net.forward(get_output_layers(net))
        
        # Initialize lists for detected bounding boxes, confidences, and class IDs
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        
        # Process each output layer
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        # Draw bounding boxes on the image
        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes)
        
        # Save the annotated image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)
        print(f"Annotated and saved: {output_path}")

# Define input and output directories and model paths
input_dir = r"C:/Users/jesli/Downloads/Images"
output_dir = r"D:/git/PestVisionChallenge/results"
weights_path = r"D:/git/Darknet/yolov4/training/yolov4-custom_5000.weights"
config_path = r"D:/git/Darknet/yolov4/darknetfor1classwithouteval/cfg/yolov4-custom.cfg"
names_path = r"D:/git/Darknet/yolov4/darknetfor1classwitheval/data/obj.names"

# Annotate all images in the input directory and save to output directory
annotate_images(input_dir, output_dir, weights_path, config_path, names_path)
