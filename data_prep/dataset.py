import torch
from torch.utils.data import Dataset
import cv2
import os
from glob import glob

class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset class for loading medical images from a directory structure.
    This version automatically discovers images and assigns labels based on subfolders.
    It expects a structure like:
    - train/
      - NORMAL/
        - img1.jpeg
        - img2.jpeg
      - PNEUMONIA/ (treated as the 'rare disease' class)
        - img3.jpeg
    """
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): Path to the directory containing class subfolders (e.g., '.../chest_xray/train').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Define the classes and their corresponding integer labels
        self.class_map = {"NORMAL": 0, "PNEUMONIA": 1}

        self._load_dataset()

    def _load_dataset(self):
        """Scans the directory to find all image paths and their corresponding labels."""
        for class_name, label in self.class_map.items():
            class_path = os.path.join(self.img_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"Warning: Directory not found for class '{class_name}' at {class_path}")
                continue
            
            # Use glob to find all image files (jpeg, png)
            paths = glob(os.path.join(class_path, "*.jpeg"))
            paths.extend(glob(os.path.join(class_path, "*.png")))
            
            self.image_paths.extend(paths)
            self.labels.extend([label] * len(paths))
            
        print(f"Loaded {len(self.image_paths)} images from {self.img_dir}")
        print(f"Class distribution: NORMAL={self.labels.count(0)}, PNEUMONIA={self.labels.count(1)}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # OpenCV loads images in BGR format, so we convert to RGB.
            image = cv2.imread(img_path)
            if image is None:
                raise IOError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return {'image': torch.zeros((3, 224, 224)), 'label': -1}
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        sample = {'image': image, 'label': label}
        return sample