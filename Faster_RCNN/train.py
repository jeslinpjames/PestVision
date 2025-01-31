import os
import glob
import random
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

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
    def __init__(self, img_dir, label_dir, transform=None, num_samples=2000):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # ✅ Load all image paths
        all_images = sorted(glob.glob(os.path.join(img_dir, "*.*")))

        # ✅ Randomly select `num_samples` images
        self.img_files = random.sample(all_images, min(num_samples, len(all_images)))

        print(f"✅ Selected {len(self.img_files)} images for training")

    def get_label_path(self, img_path):
        """Ensure correct label file format"""
        img_name = os.path.splitext(os.path.basename(img_path))[0]  # Get name without extension
        return os.path.join(self.label_dir, img_name + ".txt")

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


# ✅ Training Setup (Windows Fix)
if __name__ == "__main__":
    # ✅ Define Dataset Paths
    train_dataset = YoloDataset(
        img_dir=r"D:/split_2/train/images",
        label_dir=r"D:/split_2/train/labels",
        transform=transforms.ToTensor(),
        num_samples=5000  # ✅ Use only 2000 images
    )

    val_dataset = YoloDataset(
        img_dir=r"D:/split_2/val/images",
        label_dir=r"D:/split_2/val/labels",
        transform=transforms.ToTensor(),
        num_samples=500  # ✅ Use only 500 images for validation
    )

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

    # ✅ Training Function with Progress Bar
    def train_one_epoch(model, optimizer, data_loader, device, epoch):
        model.train()
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")

        print(f"✅ Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    # ✅ Training Loop
    num_epochs = 5
    loss_history = []

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        loss_history.append(loss)
        
        # Save model every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"fasterrcnn_resnet50_epoch_{epoch + 1}.pth")
            print(f"✅ Model saved: fasterrcnn_resnet50_epoch_{epoch + 1}.pth")

    # ✅ Save final model
    torch.save(model.state_dict(), "fasterrcnn_resnet50_final.pth")

    # ✅ Save loss history for plotting
    with open("training_loss.json", "w") as f:
        json.dump(loss_history, f)

    print("✅ Training complete! Model & loss history saved.")

    # ✅ Plot Training Loss
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker="o", linestyle="-", color="b")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Progress")
    plt.grid()
    plt.show()
