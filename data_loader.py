import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(os.path.join(data_dir, "input")))
        self.target_images = sorted(os.listdir(os.path.join(data_dir, "output")))
        
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self.data_dir, "input", self.input_images[idx])
        target_path = os.path.join(self.data_dir, "output", self.target_images[idx])
        
        input_img = Image.open(input_path).convert('L')  # Convert to grayscale
        target_img = Image.open(target_path).convert('L')
        
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            
        return input_img, target_img