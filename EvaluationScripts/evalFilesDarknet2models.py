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

def annotate_image(net, image, classes):
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
    
    return image

def annotate_images(input_dir, output_dir, weights_path1, config_path1, names_path1, weights_path2, config_path2, names_path2):
    # Load the Darknet models
    net1 = cv2.dnn.readNet(weights_path1, config_path1)
    net2 = cv2.dnn.readNet(weights_path2, config_path2)
    
    # Set the preferred backend and target to CUDA
    net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Load class names
    with open(names_path1, 'r') as f:
        classes1 = [line.strip() for line in f.readlines()]
    with open(names_path2, 'r') as f:
        classes2 = [line.strip() for line in f.readlines()]
    
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
        
        # Annotate images with both models
        annotated_img1 = annotate_image(net1, image.copy(), classes1)
        annotated_img2 = annotate_image(net2, image.copy(), classes2)
        
        # Concatenate images side by side for comparison
        combined_img = np.concatenate((annotated_img1, annotated_img2), axis=1)
        
        # Add model labels to the combined image
        cv2.putText(combined_img, "Model 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_img, "Model 2", (image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save the combined image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, combined_img)
        print(f"Annotated and saved: {output_path}")

# Define input and output directories and model paths
input_dir = r"C:/Users/jesli/Downloads/Images"
output_dir = r"D:/git/PestVisionChallenge/results2models"
weights_path1 = r"D:/git/Darknet/yolov4/yolov4-custom_best.weights"
config_path1 = r"D:/git/Darknet/yolov4/darknetfor1classwitheval/cfg/yolov4-custom.cfg"  # You need to provide the path to your first config file
names_path1 = r"D:/git/Darknet/yolov4/darknetfor1classwitheval/data/obj.names"  # You need to provide the path to your first class names file
weights_path2 = r"d:/git/Darknet/yolov4/training/yolov4-custom_last.weights"  # You need to provide the path to your second weights file
config_path2 = r"D:/git/Darknet/yolov4/darknetfor1classwitheval/cfg/yolov4-custom.cfg" # You need to provide the path to your second config file
names_path2 = r"D:/git/Darknet/yolov4/darknetfor1classwitheval/data/obj.names"  # You need to provide the path to your second class names file

# Annotate all images in the input directory and save to output directory
annotate_images(input_dir, output_dir, weights_path1, config_path1, names_path1, weights_path2, config_path2, names_path2)
