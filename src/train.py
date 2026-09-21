import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import UNet
from loss import DiceBCELoss
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import BrainTumorDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_dataset = BrainTumorDataset(
        image_dir="../data/train/images", 
        mask_dir="../data/train/masks", 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 10
    os.makedirs("../checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            preds = model(images)
            loss = criterion(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(train_loader)
        print(f"Kết thúc Epoch {epoch+1} - Average Loss: {avg_loss:.4f}\n")
        
        torch.save(model.state_dict(), f"../checkpoints/unet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()