import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image
import os
import numpy as np
from tqdm import tqdm
import logging
from PIL import Image
from torch.utils.data import Dataset

# Hyperparameters
latent_dim = 120
img_size = 320
channels = 3
batch_size = 32
lr = 0.0001
b1 = 0.5
b2 = 0.999
n_epochs = 500
sample_interval = 300

img_shape = (channels, img_size, img_size)

# Set up logging to save output to both terminal and text file
logging.basicConfig(level=logging.INFO, 
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler("training_log.txt"),
                        logging.StreamHandler()
                    ])

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), latent_dim, 1, 1)
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Conv2d(1024, 1, 1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        validity = validity.view(validity.size(0), -1)  # Flatten to (batch_size, 1)
        return validity

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers with different learning rates
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr * 2, betas=(b1, b2))  # Increased LR for generator
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# # Load best weights if available
# if os.path.exists("saved_models/best_generator.pth"):
#     generator.load_state_dict(torch.load("saved_models/best_generator.pth"))
#     logging.info("Resumed training from best generator model")

# if os.path.exists("saved_models/best_discriminator.pth"):
#     discriminator.load_state_dict(torch.load("saved_models/best_discriminator.pth"))
#     logging.info("Resumed training from best discriminator model")

# Variables to track the best losses
best_g_loss = float('inf')
best_d_loss = float('inf')

class SingleFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in the directory {folder_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0  # Returning 0 as a dummy label

# Image dataset path
data_path = 'gan_data'

# Configure data loader
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = SingleFolderDataset(data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Move models to GPU
generator = generator.to(device)
discriminator = discriminator.to(device)

# Ensure the directory exists before saving images
os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# Training loop
for epoch in range(n_epochs):
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    num_batches = len(dataloader)

    for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)):

        # Move tensors to the configured device
        real_imgs = imgs.to(device)
        valid = torch.ones((imgs.size(0), 1), requires_grad=False).to(device) * (0.9 + torch.rand((imgs.size(0), 1)).to(device) * 0.1)
        fake = torch.zeros((imgs.size(0), 1), requires_grad=False).to(device) * (0.1 + torch.rand((imgs.size(0), 1)).to(device) * 0.1)
        z = torch.randn((imgs.size(0), latent_dim)).to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z)

        # Feature matching loss
        real_features = discriminator.model[:-1](real_imgs)
        fake_features = discriminator.model[:-1](gen_imgs)
        fm_loss = torch.mean(torch.abs(real_features - fake_features))

        # Generator loss (Adversarial + Feature Matching)
        g_loss = fm_loss + adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Discriminator loss
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Accumulate epoch losses
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()

        # Log progress and save images
        if i % sample_interval == 0:
            logging.info(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            save_image(gen_imgs.data[:25].cpu(), f"images/{epoch}_{i}.png", nrow=5, normalize=True)

    # Calculate average losses for the epoch
    avg_g_loss = epoch_g_loss / num_batches
    avg_d_loss = epoch_d_loss / num_batches

    # Save the models with the least loss after the epoch
    if avg_g_loss < best_g_loss:
        best_g_loss = avg_g_loss
        torch.save(generator.state_dict(), "saved_models/best_generator.pth")
        logging.info(f"Saved new best generator model with average G loss: {best_g_loss}")

    if avg_d_loss < best_d_loss:
        best_d_loss = avg_d_loss
        torch.save(discriminator.state_dict(), "saved_models/best_discriminator.pth")
        logging.info(f"Saved new best discriminator model with average D loss: {best_d_loss}")

# Save final models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
logging.info("Final models saved: generator.pth and discriminator.pth")
