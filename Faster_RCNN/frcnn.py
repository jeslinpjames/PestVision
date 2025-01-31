import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt

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

# ✅ Define Dataset Class
class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))  # Supports jpg, png, jpeg
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

        # Remove images without annotations
        self.img_files = [img for img in self.img_files if os.path.exists(self.get_label_path(img))]

        print(f"✅ Found {len(self.img_files)} images with labels")

    def get_label_path(self, img_path):
        """Ensure correct label file format"""
        img_name = os.path.splitext(os.path.basename(img_path))[0]  # Get name without extension
        label_path = os.path.join(self.label_dir, img_name + ".txt")
        return label_path

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.get_label_path(img_path)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Read annotation
        boxes = []
        labels = []
        w, h = image.size  # Image width and height

        with open(label_path, "r") as file:
            for line in file.readlines():
                if not line.strip():
                    continue
                class_id, x_center, y_center, width, height = map(float, line.strip().split())

                # Convert YOLO format (normalized) to Faster R-CNN format (absolute)
                x_min = (x_center - width / 2) * w
                y_min = (y_center - height / 2) * h
                x_max = (x_center + width / 2) * w
                y_max = (y_center + height / 2) * h

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id))  # No background shift

        if not boxes:
            return self.__getitem__((idx + 1) % len(self.img_files))  # Skip empty annotations

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target


# ✅ Add this block to prevent multiprocessing errors on Windows
if __name__ == "__main__":
    # ✅ Define Dataset Paths
    train_dataset = YoloDataset(
        img_dir=r"D:/split_2/train/images",
        label_dir=r"D:/split_2/train/labels",
        transform=transforms.ToTensor()
    )

    val_dataset = YoloDataset(
        img_dir=r"D:/split_2/val/images",
        label_dir=r"D:/split_2/val/labels",
        transform=transforms.ToTensor()
    )

    print(f"Total images before filtering: {len(glob.glob('D:/split_2/train/images/*.*'))}")
    print(f"Total images after filtering: {len(train_dataset.img_files)}")

    # ✅ DataLoader (Reduce num_workers to avoid Windows multiprocessing issues)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    import torchvision

    # ✅ Load Faster R-CNN Model
    def get_model(num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(NUM_CLASSES).to(device)

    # ✅ Define Optimizer & Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    def train_one_epoch(model, optimizer, data_loader, device, epoch):
        model.train()
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch}] Loss: {loss.item():.4f}")

    # ✅ Training Loop
    num_epochs = 5
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        torch.save(model.state_dict(), f"fasterrcnn_resnet50_epoch_{epoch + 1}.pth")
        print(f"Model saved: fasterrcnn_resnet50_epoch_{epoch + 1}.pth")

    # ✅ Load Trained Model
    model.load_state_dict(torch.load("fasterrcnn_resnet50_epoch_5.pth"))
    model.to(device)
    model.eval()

    def prepare_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        return image_tensor, image

    # ✅ Run Model on a Sample Image
    image_path = "D:/split_2/test/images/sample.jpg"
    image_tensor, image = prepare_image(image_path)

    with torch.no_grad():
        prediction = model(image_tensor)

    # ✅ Draw Predictions
    def draw_boxes(image, prediction):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for box, label, score in zip(prediction[0]['boxes'].cpu().numpy(),
                                    prediction[0]['labels'].cpu().numpy(),
                                    prediction[0]['scores'].cpu().numpy()):
            if score > 0.5:
                x_min, y_min, x_max, y_max = box
                class_name = CLASS_MAP[label]
                plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none'))
                plt.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r')
        plt.axis('off')
        plt.show()

    draw_boxes(image, prediction)
