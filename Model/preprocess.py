import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CovidDataset(Dataset):
    def __init__(self, image_paths, labels, target_size=(224, 224), transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size
        self.transform = transform
        self.valid_indices = []

        # Preprocess and filter out null/outlier images
        self._preprocess_images()

    def _preprocess_images(self):
        temp_paths = []
        temp_labels = []
        for i, img_path in enumerate(self.image_paths):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                img_resized = cv2.resize(
                    img, self.target_size, interpolation=cv2.INTER_AREA
                )
                variance = np.var(img_resized)
                if variance > 10:  # Threshold to filter out low-variance images
                    temp_paths.append(img_path)
                    temp_labels.append(self.labels[i])
                    self.valid_indices.append(i)
                else:
                    print(f"Removed outlier image: {img_path} (variance: {variance})")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        self.image_paths = temp_paths
        self.labels = temp_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# Define transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
