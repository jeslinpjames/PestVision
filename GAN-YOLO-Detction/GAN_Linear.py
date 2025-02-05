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
latent_dim = 100
img_size = 300
channels = 3
batch_size = 128
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 250
sample_interval = 500
 
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
 
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers
 
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
 
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
 
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
 
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
 
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
 
# Loss function
adversarial_loss = nn.BCELoss()
 
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
 
num_gen_params = sum(p.numel() for p in generator.parameters())
num_disc_params = sum(p.numel() for p in discriminator.parameters())
 
logging.info(f"Number of parameters in the generator: {num_gen_params}")
logging.info(f"Number of parameters in the discriminator: {num_disc_params}")
 
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
 
# Initialize variables to track the best losses
best_g_loss = float('inf')
best_d_loss = float('inf')
 
# Ensure the directory exists before saving images and models
os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)
 
# Wrap the outer loop with tqdm to show progress for epochs
for epoch in range(n_epochs):
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    num_batches = len(dataloader)
 
    # Wrap the dataloader with tqdm to show progress for batches
    for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)):
 
        # Move tensors to the configured device
        real_imgs = imgs.to(device)
        valid = torch.ones((imgs.size(0), 1), requires_grad=False).to(device)
        fake = torch.zeros((imgs.size(0), 1), requires_grad=False).to(device)
        z = torch.randn((imgs.size(0), latent_dim)).to(device)
 
        # -----------------
        #  Train Generator
        # -----------------
 
        optimizer_G.zero_grad()
 
        # Generate a batch of images
        gen_imgs = generator(z)
 
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
 
        g_loss.backward()
        optimizer_G.step()
 
        # ---------------------
        #  Train Discriminator
        # ---------------------
 
        optimizer_D.zero_grad()
 
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
 
        d_loss.backward()
        optimizer_D.step()
 
        # Accumulate the epoch losses
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
 
        # Log progress and update tqdm bar
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
 
    # Save models every 20 epochs
    if epoch % 20 == 0:
        torch.save(generator.state_dict(), f"saved_models/generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"saved_models/discriminator_epoch_{epoch}.pth")
        logging.info(f"Saved models at epoch {epoch}")
 
# Save final models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
logging.info("Final models saved: generator.pth and discriminator.pth")