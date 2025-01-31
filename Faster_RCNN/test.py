import os
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import glob
import random

# ✅ Define Pest Class Labels
CLASS_MAP = {
    0: "Rice Leaf Roller",
    1: "Rice Leaf Caterpillar",
    2: "Paddy Stem Maggot",
    3: "Asiatic Rice Borer",
    4: "Yellow Rice Borer",
    5: "Rice Gall Midge",
    6: "Rice Stemfly",
    7: "Brown Plant Hopper",
    8: "White Backed Plant Hopper",
    9: "Small Brown Plant Hopper",
    10: "Rice Water Weevil",
    11: "Rice Leafhopper",
    12: "Grain Spreader Thrips",
    13: "Rice Shell Pest"
}

NUM_CLASSES = len(CLASS_MAP)  # 14 pest classes

# ✅ Step 1: Load the Pretrained Model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ✅ Step 2: Load Model Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(NUM_CLASSES).to(device)

weights_path = r"D:/git/PestVision/Faster_RCNN/fasterrcnn_resnet50_final.pth"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"✅ Model weights loaded from {weights_path}")
else:
    print(f"❌ ERROR: Model weights file not found: {weights_path}")
    exit()

model.eval()  # Set model to evaluation mode

# ✅ Step 3: Select Random Test Images
test_images_path = r"D:/split_2/test/images"
all_test_images = glob.glob(os.path.join(test_images_path, "*.*"))

if len(all_test_images) == 0:
    print("❌ ERROR: No test images found in directory!")
    exit()

selected_images = random.sample(all_test_images, min(20, len(all_test_images)))  # Select 5 random images

# ✅ Step 4: Function to Run Inference
def run_inference(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Transform image
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        prediction = model(image_tensor)

    return image, prediction[0]

# ✅ Step 5: Function to Draw Bounding Boxes
def draw_boxes(image, prediction, score_threshold=0.5):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    ax = plt.gca()

    for i in range(len(prediction["boxes"])):
        score = prediction["scores"][i].item()
        if score < score_threshold:
            continue  # Skip low-confidence predictions

        box = prediction["boxes"][i].cpu().numpy()
        label = prediction["labels"][i].item()
        x_min, y_min, x_max, y_max = box

        # Draw bounding box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add label
        class_name = CLASS_MAP.get(label, "Unknown")
        plt.text(x_min, y_min - 5, f"{class_name} ({score:.2f})", color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis("off")
    plt.show()

# ✅ Step 6: Run Inference on Selected Images
for img_path in selected_images:
    print(f"🔍 Evaluating: {os.path.basename(img_path)}")
    img, pred = run_inference(img_path)
    draw_boxes(img, pred)
