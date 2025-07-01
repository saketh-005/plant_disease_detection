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
    
    # Path configuration for both Kaggle and local environments
    IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    if IS_KAGGLE:
        # Kaggle paths - assuming the dataset is uploaded as 'plant-disease-dataset' in Kaggle
        # The dataset will be available at /kaggle/input/plant-disease-dataset/plant_disease_dataset
        BASE_DIR = '/kaggle/working/'
        DATA_DIR = '/kaggle/input/plant-disease-dataset/plant_disease_dataset'
        OUTPUT_DIR = '/kaggle/working/'
    else:
        # Local development paths
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(BASE_DIR, "..", "dataset")
        OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Model and metadata paths
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "plant_disease_model.pth")
    CLASS_INDICES_PATH = os.path.join(OUTPUT_DIR, "class_indices.json")

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
    try:
        # Set up paths - check both possible directory structures
        possible_train_dirs = [
            os.path.join(Config.DATA_DIR, 'train'),
            os.path.join(Config.DATA_DIR, 'train_set'),
            os.path.join(Config.DATA_DIR, 'train_images')
        ]
        
        possible_valid_dirs = [
            os.path.join(Config.DATA_DIR, 'valid'),
            os.path.join(Config.DATA_DIR, 'val'),
            os.path.join(Config.DATA_DIR, 'validation'),
            os.path.join(Config.DATA_DIR, 'valid_set')
        ]
        
        # Find existing directories
        train_dir = next((d for d in possible_train_dirs if os.path.exists(d) and os.path.isdir(d)), None)
        valid_dir = next((d for d in possible_valid_dirs if os.path.exists(d) and os.path.isdir(d)), None)
        
        # If not found, check for dataset in the root directory
        if train_dir is None and os.path.exists(Config.DATA_DIR):
            # Check if the data is directly in the DATA_DIR
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(Config.DATA_DIR)):
                train_dir = Config.DATA_DIR
        
        # Verify directories exist
        if train_dir is None or not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"Training directory not found. Checked: {', '.join(possible_train_dirs)}"
                f"\nCurrent directory contents: {os.listdir(Config.DATA_DIR) if os.path.exists(Config.DATA_DIR) else 'Directory does not exist'}"
            )
            
        if valid_dir is None or not os.path.exists(valid_dir):
            print(f"Warning: Validation directory not found. Using training directory for validation.")
            valid_dir = train_dir
        
        # Print dataset info
        print("\n=== Dataset Information ===")
        print(f"Environment: {'Kaggle' if Config.IS_KAGGLE else 'Local'}")
        print(f"Base directory: {Config.DATA_DIR}")
        print(f"Training data: {train_dir}")
        print(f"Validation data: {valid_dir}")

        # Data augmentation and normalization for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(Config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Only normalization for validation
        valid_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.CenterCrop(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("\nLoading datasets...")
        # Load datasets
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        
        # If validation directory is same as training, split the training data
        if valid_dir == train_dir:
            print("Splitting training data into train/validation sets...")
            # Split dataset into train and validation (80/20)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # for reproducibility
            )
            # Apply validation transform to validation set
            valid_dataset.dataset.transform = valid_transform
        else:
            valid_dataset = ImageFolder(valid_dir, transform=valid_transform)
        
        # Verify we have data
        if len(train_dataset) == 0:
            raise ValueError("No training images found!")
        if len(valid_dataset) == 0:
            raise ValueError("No validation images found!")
        
        # Save class indices
        class_to_idx = train_dataset.dataset.class_to_idx if hasattr(train_dataset, 'dataset') else train_dataset.class_to_idx
        with open(Config.CLASS_INDICES_PATH, 'w') as f:
            json.dump(class_to_idx, f)
        
        # Create data loaders
        num_workers = min(4, os.cpu_count())  # Use up to 4 workers for data loading
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        print("\nDataset loaded successfully!")
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(valid_dataset):,}")
        print(f"Number of classes: {len(class_to_idx)}")
        print(f"Class labels: {', '.join(list(class_to_idx.keys())[:5])}...")
        
        return train_loader, valid_loader, class_to_idx
        
    except Exception as e:
        print(f"\nError loading datasets: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        if os.path.exists(Config.DATA_DIR):
            print(f"Contents of {Config.DATA_DIR}:")
            for item in os.listdir(Config.DATA_DIR):
                item_path = os.path.join(Config.DATA_DIR, item)
                print(f"- {item} ({'dir' if os.path.isdir(item_path) else 'file'})")
        else:
            print(f"Directory {Config.DATA_DIR} does not exist")
        raise

def main():
    print("\n" + "="*50)
    print(f"Starting training with {Config.EPOCHS} epochs")
    print(f"Batch size: {Config.BATCH_SIZE}, Learning rate: {Config.LEARNING_RATE}")
    print(f"Image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print("="*50 + "\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Load datasets
    train_loader, valid_loader, class_to_idx = get_datasets()
    
    # Initialize model
    model = SimpleCNN(num_classes=len(class_to_idx)).to(device)
    print(f"\nModel initialized on {device}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Training loop
    best_accuracy = 0.0  # Initialize best accuracy to 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*20} Epoch {epoch+1}/{Config.EPOCHS} {'='*20}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{Config.EPOCHS}')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            train_pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'acc': f'{train_acc:.2f}%'
            })
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(valid_loader, desc='Validating')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{val_loss/len(valid_loader):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # Calculate validation metrics
        val_loss = val_loss / len(valid_loader)
        val_acc = 100. * correct / total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f'Model saved with accuracy: {best_accuracy:.2f}%')
    
    print(f'\nTraining complete. Best validation accuracy: {best_accuracy:.4f}')
    print(f'Model saved to {Config.MODEL_SAVE_PATH}')

if __name__ == "__main__":
    main()
