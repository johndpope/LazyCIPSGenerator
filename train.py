import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from cips import LazyCIPSLoss,LazyCIPSGenerator


class CelebAHQMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        # Ensure mask is binary
        mask = (mask > 0.5).float()
        
        return img, mask




class InpaintingDataset(Dataset):
    def __init__(self, image_paths, mask_size=(64, 64), transform=None):
        self.image_paths = image_paths
        self.mask_size = mask_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Create a random mask
        mask = torch.zeros(1, img.shape[1], img.shape[2])
        x = np.random.randint(0, img.shape[1] - self.mask_size[0])
        y = np.random.randint(0, img.shape[2] - self.mask_size[1])
        mask[:, x:x+self.mask_size[0], y:y+self.mask_size[1]] = 1
        
        return img, mask

def train_lazy_cips(model, dataloader, num_epochs=10, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = LazyCIPSLoss().to(device)
    
    for epoch in range(num_epochs):
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            masked_images = images * (1 - masks)
            latents = torch.randn(images.shape[0], model.input_dim).to(device)
            
            generated_images = model(latents, masked_images, masks, images.shape[2])
            
            # Compute loss
            loss, loss_dict = criterion(generated_images, images, masks)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, L1: {loss_dict['l1']:.4f}, "
                      f"Perceptual: {loss_dict['perceptual']:.4f}")
        
        # Visualize results
        if (epoch + 1) % 5 == 0:
            visualize_results(model, images, masks, masked_images, generated_images)
            visualize_results(model, images, masks, masked_images, generated_images)

def visualize_results(model, images, masks, masked_images, generated_images):
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        axs[i, 0].imshow(images[i].cpu().permute(1, 2, 0).detach().numpy())
        axs[i, 0].set_title("Original")
        axs[i, 1].imshow(masked_images[i].cpu().permute(1, 2, 0).detach().numpy())
        axs[i, 1].set_title("Masked")
        axs[i, 2].imshow(generated_images[i].cpu().permute(1, 2, 0).detach().numpy())
        axs[i, 2].set_title("Generated")
        axs[i, 3].imshow(masks[i].cpu().squeeze().detach().numpy(), cmap='gray')
        axs[i, 3].set_title("Mask")
    plt.tight_layout()
    plt.show()

# Usage example
if __name__ == "__main__":
    # Assume we have a list of image paths
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    dataset = CelebAHQMaskDataset('path/to/celebA_images', 'path/to/celebA_masks', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # dataset = InpaintingDataset(image_paths, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Initialize model
    model = LazyCIPSGenerator(input_dim=256, max_resolution=256).to('cuda')
    
    # Train model
    train_lazy_cips(model, dataloader)
