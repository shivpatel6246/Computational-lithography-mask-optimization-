# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from model import UNet
from data_loader import MaskDataset
import os
import platform
import time
from datetime import datetime, timedelta

def get_device():
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) available - using Apple Silicon GPU")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print("\nStarting training process...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    # Enable automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        # Training phase with progress tracking
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Use mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            optimizer.zero_grad(set_to_none=True)  # Slightly more efficient than zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                images_processed = (batch_idx + 1) * inputs.size(0)
                total_images = len(train_loader.dataset)
                progress = 100. * images_processed / total_images
                print(f"Epoch {epoch+1}: {images_processed}/{total_images} ({progress:.1f}%) "
                      f"Loss: {loss.item():.4f}")
        
        # Quick validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        # Calculate epoch metrics
        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s")
        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        # Save if best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    return train_losses, val_losses

def main():
    # Performance Optimizations
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Hyperparameters optimized for speed
    batch_size = 16  # Increased from 4
    num_epochs = 20   # Reduced from 100
    learning_rate = 0.01
    image_size = 128  # Reduced from 256
    
    # Transform with smaller image size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = MaskDataset(data_dir='dataset/train', transform=transform)
    val_dataset = MaskDataset(data_dir='dataset/val', transform=transform)
    
    # Create optimized dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,  # Increased from 0
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size*2,  # Larger batches for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model and optimizer
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # Save training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    main()