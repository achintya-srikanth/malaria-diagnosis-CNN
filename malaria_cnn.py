import os
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import csv


# Configuration
class Config:
    img_size = 224
    batch_size = 32
    epochs = 30
    lr = 1e-4
    weight_decay = 1e-5
    train_images_dir = "/jet/home/srikanta/Spring2025/nndl-hw2/NNDL_HW2/dataset/train_images/"
    test_images_dir = "/jet/home/srikanta/Spring2025/nndl-hw2/NNDL_HW2/dataset/test_images"
    train_csv = "/jet/home/srikanta/Spring2025/nndl-hw2/NNDL_HW2/dataset/train_data.csv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing (as per paper)
class PlasmodiumPreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.sharp_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
    def __call__(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, (Config.img_size, Config.img_size))
        
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l, a, b))
        enhanced_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced_img

# Custom Dataset Class
class MalariaDataset(Dataset):
    def __init__(self, csv_file=None, images_dir=None, preprocessor=None, is_test=False):
        self.is_test = is_test
        if not is_test:
            self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.preprocessor = preprocessor
        self.image_names = os.listdir(images_dir) if is_test else self.data['image_name'].tolist()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_names[idx])
        img = self.preprocessor(img_name)
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        
        if self.is_test:
            return img_tensor, self.image_names[idx]
        return img_tensor, int(self.data.iloc[idx]['label'])

# CNN Model Architecture (7-layer CNN from Nature 2025 paper)
class PlasmodiumCNN(nn.Module):
    def __init__(self):
        super(PlasmodiumCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)  # Binary classification: parasitized or uninfected
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Training and Validation Loop
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    
    best_acc = 0.0
    
    for epoch in range(Config.epochs):
        model.train()
        
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        val_acc = validate_model(model, val_loader)
        
        print(f"Epoch {epoch + 1}/{Config.epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    
    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")

# Validation Function
def validate_model(model, val_loader):
    model.eval()
    
    corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            
            corrects += (preds == labels).sum().item()
    
    return corrects / len(val_loader.dataset)

# Main Function
def main():
    preprocessor = PlasmodiumPreprocessor()

    # Prepare datasets and loaders
    train_dataset_full = MalariaDataset(
        csv_file=Config.train_csv,
        images_dir=Config.train_images_dir,
        preprocessor=preprocessor,
    )
    
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
    
    test_dataset = MalariaDataset(
        images_dir=Config.test_images_dir,
        preprocessor=preprocessor,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
    
    model = PlasmodiumCNN().to(Config.device)
    
    train_model(model=model, train_loader=train_loader, val_loader=val_loader)

if __name__ == "__main__":
    main()

def generate_test_predictions(model_path, test_dir):
    # Load model
    model = PlasmodiumCNN().to(Config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create test dataset
    preprocessor = PlasmodiumPreprocessor()
    test_dataset = MalariaDataset(
        images_dir=test_dir,
        preprocessor=preprocessor,
        is_test=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)

    # Generate predictions
    predictions = []
    with torch.no_grad():
        for inputs, filenames in test_loader:
            inputs = inputs.to(Config.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(zip(filenames, preds.cpu().numpy()))

    # Save to CSV
    with open('test_predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['img_name', 'label'])
        for filename, label in predictions:
            writer.writerow([filename, int(label)])

    print(f"Predictions saved to test_predictions.csv")

# Run the function to generate test predictions
generate_test_predictions(
    model_path="best_model.pth",  # Path to your trained model
    test_dir=Config.test_images_dir
)
