import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from model import UNet
from data_loader import MaskDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def evaluate_model(model, test_loader, device):
    model.eval()
    mse_losses = []
    mae_losses = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Convert to numpy for metric calculation
            outputs_np = outputs.cpu().numpy().squeeze()
            targets_np = targets.cpu().numpy().squeeze()
            
            mse = mean_squared_error(targets_np, outputs_np)
            mae = mean_absolute_error(targets_np, outputs_np)
            
            mse_losses.append(mse)
            mae_losses.append(mae)
            
            # Save sample predictions
            if i < 6:  # Save first 5 predictions
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.imshow(inputs.cpu().numpy().squeeze(), cmap='gray')
                ax1.set_title('Input Mask')
                ax2.imshow(outputs.cpu().numpy().squeeze(), cmap='gray')
                ax2.set_title('Predicted Pattern')
                ax3.imshow(targets.cpu().numpy().squeeze(), cmap='gray')
                ax3.set_title('Actual Pattern')
                plt.savefig(f'prediction_sample_{i}.png')
                plt.close()
    
    return np.mean(mse_losses), np.mean(mae_losses)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create test dataset and dataloader
    test_dataset = MaskDataset(data_dir='dataset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = UNet().to(device)
    checkpoint = torch.load('best_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])  # Extract model state_dict
    
    # Evaluate
    mse, mae = evaluate_model(model, test_loader, device)
    
    # Print results
    print(f'Test Mean Squared Error: {mse:.4f}')
    print(f'Test Mean Absolute Error: {mae:.4f}')


if __name__ == "__main__":
    main()