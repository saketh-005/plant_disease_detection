import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json
from pathlib import Path

# Configuration
class Config:
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001
    IMG_SIZE = 256
    
    # Update this path to your dataset location
    DATA_DIR = "/kaggle/input/plant-disease-dataset/New_Plant_Diseases_DatasetAugmented/New_Plant_Diseases_DatasetAugmented"  
    
    # Output paths
    MODEL_SAVE_PATH = "/kaggle/working/plant_disease_model.pth"
    CLASS_INDICES_PATH = "/kaggle/working/class_indices.json"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model Definition
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.flattened_size = 128 * 16 * 16
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_datasets():
    # Set up paths
    train_dir = os.path.join(Config.DATA_DIR, 'train')
    valid_dir = os.path.join(Config.DATA_DIR, 'valid')
    
    # Print dataset info
    print("\n=== Dataset Information ===")
    print(f"Training data: {train_dir}")
    print(f"Validation data: {valid_dir}")

    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Only normalization for validation
    valid_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    valid_dataset = ImageFolder(valid_dir, transform=valid_transform)
    
    # Save class indices
    class_to_idx = train_dataset.class_to_idx
    with open(Config.CLASS_INDICES_PATH, 'w') as f:
        json.dump(class_to_idx, f)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    print(f"\nDataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Number of classes: {len(class_to_idx)}")
    
    return train_loader, valid_loader, class_to_idx

def main():
    train_loader, valid_loader, class_to_idx = get_datasets()
    
    # Initialize model
    model = SimpleCNN(num_classes=len(class_to_idx)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch {epoch+1}/{Config.EPOCHS}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS} - Training', 
                   unit='batch', ncols=100, position=0, leave=True)
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar
            epoch_loss = running_loss / ((pbar.n + 1) * Config.BATCH_SIZE)
            epoch_acc = running_corrects.double() / ((pbar.n + 1) * Config.BATCH_SIZE)
            pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'acc': f'{epoch_acc:.2%}'})
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        val_epoch_loss = val_running_loss / len(valid_dataset)
        val_epoch_acc = val_running_corrects.double() / len(valid_dataset)
        
        print(f'\nTrain Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # Update learning rate
        scheduler.step(val_epoch_loss)
        
        # Save best model
        if val_epoch_acc > best_accuracy:
            best_accuracy = val_epoch_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
    
    print(f'\nTraining complete. Best validation accuracy: {best_accuracy:.4f}')
    print(f'Model saved to {Config.MODEL_SAVE_PATH}')

if __name__ == "__main__":
    main()
