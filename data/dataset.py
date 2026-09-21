import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import apply_clahe, z_score_normalize

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_name = self.images[index].replace('.tif', '_mask.tif').replace('.png', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image_np = np.array(image)
        image_clahe = apply_clahe(image_np)
        image = Image.fromarray(image_clahe)

        if self.transform is not None:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
            
        image = z_score_normalize(image)

        return image, mask