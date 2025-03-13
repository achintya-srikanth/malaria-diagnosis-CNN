import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset
import torch.nn as nn

# Configuration
class Config:
    img_size = 224
    batch_size = 32
    test_images_dir = "/path_to/test_images"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing
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

# Dataset class for test data
class MalariaDataset(Dataset):
    def __init__(self, images_dir, preprocessor):
        self.images_dir = images_dir
        self.preprocessor = preprocessor
        self.image_names = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_names[idx])
        img = self.preprocessor(img_name)
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        return img_tensor, self.image_names[idx]

# Model architecture
class PlasmodiumCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Load model and extract embeddings
def extract_embeddings(model_path, test_dir):
    model = PlasmodiumCNN().to(Config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preprocessor = PlasmodiumPreprocessor()
    test_dataset = MalariaDataset(test_dir, preprocessor)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    embeddings = []
    filenames = []
    with torch.no_grad():
        for inputs, names in test_loader:
            inputs = inputs.to(Config.device)
            features = model.features(inputs)  # Extract features from the convolutional layers
            features = features.view(features.size(0), -1)  # Flatten the features
            embeddings.append(features.cpu().numpy())
            filenames.extend(names)

    return np.vstack(embeddings), filenames

# Apply t-SNE and plot
def visualize_embeddings(embeddings, filenames):
    labels = pd.read_csv("test_predictions.csv")['label'].values[:len(filenames)]
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 5))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='bwr', alpha=0.5)
    plt.title('t-SNE Visualization')
    plt.savefig('embedding.png')

# Main function
def main():
    model_path = "best_model.pth"
    test_dir = Config.test_images_dir
    embeddings, filenames = extract_embeddings(model_path, test_dir)
    visualize_embeddings(embeddings, filenames)

if __name__ == "__main__":
    main()
